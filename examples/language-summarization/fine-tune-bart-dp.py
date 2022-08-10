# Train BART-type models with DP

import datasets
import dp_transformers
import transformers
import opacus
import nltk
import numpy as np
import sys
import torch
import logging
from typing import Optional
import prv_accountant

from dataclasses import dataclass, field
from transformers.training_args import ParallelMode

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str

    max_source_length: int = field(default=1024, metadata={
        "help": (
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        )
    })
    
    max_target_length: int = field(default=128, metadata={
        "help": (
            "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        )
    })

    cache_dir: Optional[str] = field(default=None, metadata={
        "help": "Where to store the pretrained models and datasets downloaded from huggingface.co"
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
    model = transformers.BartForConditionalGeneration.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    # Load data
    dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', cache_dir=args.model.cache_dir)
    dataset['train'] = dataset['train'] #.select(range(100))
    dataset['validation'] = dataset['validation'] #.select(range(100))
    dataset['test'] = dataset['test'] #.select(range(100))

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)

    def preprocess_function(examples):
    # HF preprocess method, removing pairs where at least one record is None
        text_column, summary_column = 'article', 'highlights' # specific to cnn_dailymail
        # text_column, summary_column = 'description', 'abstract' # specific to big_patent ds

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.model.max_source_length, padding="max_length", truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.model.max_target_length, padding="max_length", truncation=True)

        # Since we are padding here, replace all tokenizer.pad_token_id in the labels by -100 to ignore
        # padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing training dataset"):
        train_dataset = dataset["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True, num_proc=6, desc="tokenizing dataset", remove_columns=dataset["train"].column_names
        )

    with train_args.main_process_first(desc="tokenizing eval dataset"):
        eval_dataset = dataset["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True, num_proc=6, desc="tokenizing dataset", remove_columns=dataset["train"].column_names
        )

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    model.train()

    dp_transformers.register_grad_sampler_bart_cond()

    if train_args.parallel_mode == ParallelMode.DISTRIBUTED:
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

    data_collator = dp_transformers.DataCollatorForPrivateSeq2Seq(tokenizer)

    # Metric
    metric = datasets.load_metric("rouge")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            dp_transformers.PrivacyEngineCallback(
                privacy_engine,
                lambda s: privacy_acccountant.compute_epsilon(s)[2]
            )
        ],
        data_collator=data_collator,
        compute_metrics=compute_metrics
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
