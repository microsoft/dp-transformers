# Train T5 with DP

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import opacus
import prv_accountant
import torch
import transformers
from transformers.training_args import ParallelMode

import dp_transformers
from dp_transformers.module_modification import convert_t5_attention_nn_embedding_to_relative_position_embedding

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="t5-base", metadata={
        "help": "Model name in HuggingFace, e.g. 't5-base'"
    })

    dataset_name: str = field(default="cnn_dailymail", metadata={
        "help": "Dataset compatible with model's task (e.g. summarization)"
    })

    max_source_length: int = field(default=512, metadata={
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
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
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
        f"distributed training: {bool(train_args.local_rank != -1)}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model.model_name)

    # Fix for relative position embedding
    model = convert_t5_attention_nn_embedding_to_relative_position_embedding(
        model, train_args.per_device_train_batch_size)

    model = model.to(train_args.device)

    # Load data
    if model_args.dataset_name == "cnn_dailymail":
        dataset = datasets.load_dataset('cnn_dailymail', '3.0.0', cache_dir=args.model.cache_dir)
        text_column, summary_column = 'article', 'highlights'  # specific to cnn_dailymail
    elif model_args.dataset_name == "big_patent":
        dataset = datasets.load_dataset('big_patent', 'g', cache_dir=args.model.cache_dir)
        text_column, summary_column = 'description', 'abstract'  # specific to big_patent ds

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)

    def preprocess_function(examples):
        # HF preprocess method, removing pairs where at least one record is None
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
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True, num_proc=6, desc="tokenizing dataset", remove_columns=dataset["train"].column_names
        )

    with train_args.main_process_first(desc="tokenizing eval dataset"):
        eval_dataset = dataset["validation"].map(
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

    # Fix for tied embedding and t5 layer norm
    dp_transformers.register_grad_sampler_t5()

    if train_args.parallel_mode == ParallelMode.DISTRIBUTED:
        model = dp_transformers.dp_utils.DifferentiallyPrivateDistributedDataParallel(model)

    logger.info(f"world_size: {train_args.world_size}, "
                f"batch size: {train_args.per_device_train_batch_size * train_args.world_size * train_args.gradient_accumulation_steps}")

    sampling_probability = float(
        train_args.per_device_train_batch_size * train_args.world_size * train_args.gradient_accumulation_steps) / len(
        dataset['train'])
    num_steps = int(train_args.num_train_epochs * (1 / sampling_probability + 1))

    target_delta_calc = 1.0 / (len(dataset['train']) * math.log(len(dataset['train'])))
    logger.info(f"Sampling probability = {sampling_probability}, no. of steps = {num_steps}")

    if privacy_args.noise_multiplier is None:
        logger.info(f"Noise multiplier not provided, calculating.")
        noise_multiplier = dp_transformers.dp_utils.find_noise_multiplier(
            sampling_probability=sampling_probability,
            num_steps=num_steps,
            target_delta=target_delta_calc,
            target_epsilon=privacy_args.target_epsilon
        )
    else:
        noise_multiplier = privacy_args.noise_multiplier

    logger.info(f"The noise multiplier is set to be: {noise_multiplier}")

    privacy_engine = opacus.PrivacyEngine(module=model,
                                          batch_size=train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps,
                                          sample_size=len(dataset['train']),
                                          max_grad_norm=privacy_args.per_sample_max_grad_norm,
                                          noise_multiplier=noise_multiplier,
                                          target_delta=target_delta_calc
                                          )
    # Privacy engine has already an accountant (`get_privacy_spent`) but it is overly pessimistic.
    # Hence, we use a more accurate accountant separately.
    privacy_acccountant = prv_accountant.Accountant(
        noise_multiplier=noise_multiplier,
        sampling_probability=sampling_probability,
        delta=target_delta_calc,
        eps_error=0.1,
        max_compositions=num_steps
    )

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)

    train_args.prediction_loss_only = False
    train_args.predict_with_generate = True

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
        data_collator=data_collator
    )

    try:
        trainer.train()
    finally:
        eps_prv = privacy_acccountant.compute_epsilon(privacy_engine.steps)[2]
        eps_rdp, alpha = privacy_engine.get_privacy_spent(target_delta_calc)
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })


if __name__ == "__main__":
    try:
        arg_parser = transformers.HfArgumentParser(
            (dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
        train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()

        main(Arguments(train=train_args, privacy=privacy_args, model=model_args))

    except Exception as e:
        logger.error(str(e))
        logger.exception(e)
