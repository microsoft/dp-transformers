# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ parameter-efficient approach LoRA when lora_dim > 0)'''

import os
import sys
import logging
import shrike
from shrike.compliant_logging.exceptions import prefix_stack_trace
from shrike.compliant_logging.constants import DataCategory

shrike.compliant_logging.enable_compliant_logging(
        "SystemLog:",
        level="INFO",
        format="%(prefix)s%(levelname)s:%(name)s:%(message)s",
    )

class CompliantLoggerHack(shrike.compliant_logging.logging.CompliantLogger):

    def _log(
        self,
        level,
        msg,
        args=None,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        items=None,
        category=DataCategory.PRIVATE,
    ):
        super()._log(
            level,
            "SystemLog: " + msg,
            args,
            exc_info,
            extra,
            stack_info,
            stacklevel,
            items,
            category,
        )

logging.setLoggerClass(CompliantLoggerHack)
logger = logging.getLogger(__name__)
logger.info("Hello, world!", category=DataCategory.PUBLIC)

import datasets
import torch
import dp_transformers
import transformers

from dataclasses import dataclass, field
from transformers.training_args import ParallelMode
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora


@dataclass
class ModelArguments:
    training_data: str = field(default="./", metadata={
        "help": "Path to training data"
    })

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


@prefix_stack_trace(keep_message=True)
def main(args: Arguments):

    transformers.set_seed(args.train.seed)

    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # log_level = train_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}",
        category=DataCategory.PUBLIC
    )
    logger.info(f"Training/evaluation parameters {train_args}", category=DataCategory.PUBLIC)
    logger.info(f"Privacy parameters {privacy_args}", category=DataCategory.PUBLIC)

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load data
    # data_path = "C:\\Users\\huinan\\OneDrive - Microsoft\\Desktop\\dp-transformers\\examples\\nlg-reddit\\sample-level-dp\\tiny.csv"
    data_path_train = os.path.join(args.model.training_data, "train.csv")
    data_path_val = os.path.join(args.model.training_data, "val.csv")
    dataset = datasets.load_dataset('csv', data_files={'train': data_path_train, 'validation': data_path_val}) #.train_test_split(0.2, seed=args.train.seed)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    model.resize_token_embeddings(len(tokenizer))
    #initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    # label_column_names = [name for name in dataset["train"].column_names if "label" in name]
    # Tokenize data
    def preprocess_function(examples):
        batch = []
        # for t in range(len(examples['text'])):
        #     text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
        #     batch.append(text)
        for t in range(len(examples['Subject'])):
            text = f"Write an email with {examples['HasAttachments'][t]} attachments: {examples['Subject'][t]} END END END {examples['UniqueBody'][t]} {tokenizer.eos_token}" 
            batch.append(text)

        result = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=args.model.sequence_len)

        return result

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    if args.model.lora_dim > 0:
        model = convert_gpt2_attention_to_lora(
            model, r=args.model.lora_dim, lora_alpha=args.model.lora_alpha, lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}", category=DataCategory.PUBLIC)
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}", category=DataCategory.PUBLIC)

    model = model.cuda()
    model.train()

    if args.model.lora_dim > 0:
        from dp_transformers.grad_sample.lora import lora_layer
    else:
        from dp_transformers.grad_sample.transformers import conv_1d

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        privacy_args=privacy_args,
        tokenizer=tokenizer
    )

    try:
        train_result = trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

    if train_args.local_rank == 0 or train_args.local_rank == -1:
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        #model._module.module.config.save_pretrained(train_args.output_dir)
        #torch.save(model._module.module.transformer.state_dict(), os.path.join(train_args.output_dir, "pytorch_model.bin"))
        #trainer.save_state()

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))
