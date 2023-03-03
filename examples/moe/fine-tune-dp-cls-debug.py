# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train MoE model series with DP'''

import os
import torch
import datasets
import dp_transformers
import transformers
import evaluate
import sys
import logging
import numpy as np
from typing import Optional, Tuple

from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
from modeling_switch_transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersModel 

logger = logging.getLogger(__name__)


class SwitchTransformersModelForSequenceClassification(SwitchTransformersForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=False,
            return_dict=return_dict
        )

        return outputs.loss, outputs.logits


@dataclass
class ModelArguments:
    data_dir: str

    task: str

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments


def data_process(args, tokenizer, dataset):
    
    if args.model.task == 'sst2':
        # Tokenize data
        with train_args.main_process_first(desc="tokenizing dataset"):
            dataset = dataset.map(
                lambda batch: tokenizer(batch['sentence'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
                batched=True, num_proc=None, desc="tokenizing dataset", remove_columns=[c for c in dataset['train'].column_names if c != 'label']
            )

        int_to_string_sentiment = {0: "negative", 1: "positive", -1: "unknown"}

        with train_args.main_process_first(desc="process labels"):
            dataset = dataset.map(
                lambda batch: {"labels": [tokenizer.convert_tokens_to_ids(int_to_string_sentiment[batch["label"]])]},
                batched=False, num_proc=None, desc="changing int to string for sentiment", remove_columns=['label']
            )
    elif args.model.task == 'SetFit/mnli':
        # Tokenize data
        with train_args.main_process_first(desc="tokenizing dataset"):
            dataset = dataset.map(
                lambda batch: tokenizer(batch['text1'] + batch['text2'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
                batched=False, num_proc=None, desc="tokenizing dataset", remove_columns=[c for c in dataset['train'].column_names if c != 'label']
            )

        with train_args.main_process_first(desc="process labels"):
            dataset = dataset.map(
                lambda batch: {"labels": [tokenizer.convert_tokens_to_ids(str(batch["label"]))]},
                batched=False, num_proc=None, desc="string label for inline prediction", remove_columns=['label']
            )
    else:
        raise ValueError(f"Unsupported task: {args.model.task}")
    
    return dataset


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

    # Load data
    #dataset = datasets.load_dataset(args.model.task, cache_dir=args.model.data_dir)
    #dataset = datasets.load_dataset(args.model.task)
    dataset = datasets.load_dataset('json', data_files={'train': os.path.join(args.model.data_dir, "train.json"),
                                                       'validation': os.path.join(args.model.data_dir, "val.json")},
                                                       )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name, use_fast=False)

    # Tokenize and process data
    dataset = data_process(args, tokenizer, dataset)

    # Load model
    #model = SwitchTransformersModelForSequenceClassification.from_pretrained(args.model.model_name, cache_dir=args.model.data_dir)
    config = AutoConfig.from_pretrained(args.model.model_name)
    config.dropout_rate = 0.0
    model = SwitchTransformersModelForSequenceClassification.from_pretrained(args.model.model_name, config=config)
    model = model.to(train_args.device)

    # for index, block in enumerate(model.encoder.block):
    #     if index % 2 == 1:
    #         block.layer[1].mlp.router.classifier.weight.requires_grad = False
    #         for expert in block.layer[1].mlp.experts:
    #             block.layer[1].mlp.experts[expert].wi.weight.requires_grad = False
    #             block.layer[1].mlp.experts[expert].wo.weight.requires_grad = False

    # for index, block in enumerate(model.decoder.block):
    #     if index % 2 == 1:
    #         block.layer[2].mlp.router.classifier.weight.requires_grad = False
    #         for expert in block.layer[2].mlp.experts:
    #             block.layer[2].mlp.experts[expert].wi.weight.requires_grad = False
    #             block.layer[2].mlp.experts[expert].wo.weight.requires_grad = False

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    #model = model.cuda()
    model.train()

    data_collator = DataCollatorWithPadding(tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        privacy_args=privacy_args,
        compute_metrics=compute_metrics
    )

    #metrics = trainer.evaluate()
    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))
