# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import datasets
import dp_transformers
import transformers
import opacus
import sys
import logging
import prv_accountant

from dataclasses import dataclass, field
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    lora_dim: int = field(default=0, metadata={
        "help": "LoRA dimension; 0 means LoRA is disabled"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })

    lora_dropout: float = field(default=0.0, metadata={
        "help": "Dropout probability for LoRA layers"
    })

    lora_alpha: int = field(default=32, metadata={
        "help": "LoRA attention alpha"
    })


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments


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

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load data
    dataset = datasets.load_dataset('reddit', split="train[:500000]").train_test_split(0.02, seed=args.train.seed)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    tokenizer.pad_token = -100 # Set a dummy pad token we don't use it anyway

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['content'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    if args.model.lora_dim > 0:
        model = convert_gpt2_attention_to_lora(
            model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    if args.model.lora_dim > 0:
        dp_transformers.register_grad_sampler_gpt2_lora()
    else:
        dp_transformers.register_grad_sampler_gpt2()

    if train_args.n_gpu > 1:
        model = dp_transformers.dp_utils.DifferentiallyPrivateDistributedDataParallel(model)

    sampling_probability = train_args.per_device_train_batch_size*train_args.world_size*train_args.gradient_accumulation_steps/len(dataset['train'])
    num_steps = int(train_args.num_train_epochs*(1/sampling_probability+1))
    if privacy_args.noise_multiplier is None: 
        noise_multiplier = dp_transformers.dp_utils.find_noise_multiplier(
            sampling_probability=sampling_probability,
            num_steps=num_steps,
            target_delta=1.0/len(dataset['train']),
            target_epsilon=privacy_args.target_epsilon
        )
    else:
        noise_multiplier = privacy_args.noise_multiplier
    if train_args.local_rank == 0:
        logger.info(f"The noise multiplier is set to be: {noise_multiplier}")

    privacy_engine = opacus.PrivacyEngine(module=model,
        batch_size=train_args.per_device_train_batch_size*train_args.gradient_accumulation_steps, sample_size=len(dataset['train']),
        max_grad_norm=privacy_args.per_sample_max_grad_norm, noise_multiplier=noise_multiplier, target_delta=1.0/len(dataset['train'])
    )
    # Privacy engine has already an accountant (`get_privacy_spent`) but it is overly pessimistic.
    # Hence, we use a more accurate accountant separately.
    privacy_acccountant = prv_accountant.Accountant(
        noise_multiplier=noise_multiplier,
        sampling_probability=sampling_probability,
        delta=1.0/len(dataset['train']),
        eps_error=0.1,
        max_compositions=num_steps
    )

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        callbacks=[
            dp_transformers.PrivacyEngineCallback(
                privacy_engine,
                lambda s: privacy_acccountant.compute_epsilon(s)[2]
            )
        ],
        data_collator=data_collator
    )

    try:
        trainer.train()
    finally:
        eps_prv = privacy_acccountant.compute_epsilon(privacy_engine.steps)[2]
        eps_rdp, alpha = privacy_engine.get_privacy_spent(1.0/len(dataset['train']))
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))
