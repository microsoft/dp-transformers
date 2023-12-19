# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train LLMs with DP using QLoRA'''

import datasets
import dp_transformers
import transformers
import sys
import logging
import torch
import ast
import linear
import data_utils

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Union
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    dataset_name: str = field(default="sst2", metadata={
        "help": "Dataset name in HuggingFace, e.g. 'sst2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    target_modules: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        params["target_modules"] = ast.literal_eval(params["target_modules"][0])
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    lora: LoraArguments


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = data_utils.ALL_DATASETS[args.model.dataset_name](tokenizer, args.model.sequence_len)

    if dataset.classes is not None:
        target_max_len = dataset.target_max_len()
        logger.info(f"Labels tokenized into max length: {target_max_len}")

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset.dataset = dataset.dataset.map(
            dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset", 
            remove_columns=dataset.dataset.column_names['train']
        )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name, quantization_config=bnb_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_args.gradient_checkpointing)

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset.dataset['train'],
        eval_dataset=dataset.dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=dataset.compute_metrics,
        preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
        privacy_args=privacy_args,
    )

    if hasattr(trainer.model._module, "config"):
        # The following is for GradSampleModule wrapping
        ignore_keys = getattr(trainer.model._module.config, "keys_to_ignore_at_inference", [])
    elif hasattr(trainer.model._module.module, "config"):
        # The following is for GradSampleModule and DPDDP wrapping
        ignore_keys = getattr(trainer.model._module.module.config, "keys_to_ignore_at_inference", [])
    else:
        ignore_keys = []

    try:
        # A workaround to avoid the following error:
        # AttributeError: 'GradSampleModule' object has no attribute 'gradient_checkpointing_enable'
        # inside Trainer _inner_training_loop. Already done by prepare_model_for_kbit_training
        trainer.args.gradient_checkpointing = False
        result = trainer.train(ignore_keys_for_eval=ignore_keys)
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

    if dataset.run_test:
        logger.info("Running test set evaluation after training")   
        test_metrics = dataset.compute_test_metrics(trainer)
        trainer.log(test_metrics)

    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()

    print_summary(result)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, LoraArguments))
    train_args, privacy_args, model_args, lora_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args))
