# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train MoE model series without DP'''

import datasets
import dp_transformers
import transformers
import sys
import logging

from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    data_dir: str

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
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

    # Load data
    dataset = datasets.load_dataset("sst2")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model.model_name, num_labels=2, id2label=id2label, 
        label2id=label2id, cache_dir=args.model.data_dir)
    model = model.to(train_args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name, use_fast=False)

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['sentence'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=None, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    data_collator = DataCollatorWithPadding(tokenizer, mlm=False)

    trainer = transformers.Trainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator
    )

    metrics = trainer.evaluate()
    trainer.train()

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, ModelArguments))
    train_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, model=model_args))
