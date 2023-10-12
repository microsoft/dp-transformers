# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import datasets
import torch
import opacus
from datasets import Dataset
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, TrainerCallback, TrainerState, TrainerControl, logging,
    DataCollatorForLanguageModeling, PreTrainedTokenizer, training_args, modeling_utils
)
from transformers.file_utils import is_sagemaker_mp_enabled, is_datasets_available
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from accelerate.optimizer import AcceleratedOptimizer


from dp_transformers import sampler, arguments
from dp_transformers.data import AuthorIndexedDataset

logger = logging.get_logger(__name__)


class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """
    def __init__(
        self,
        compute_epsilon: Callable[[], float],
        max_epsilon: float = float('inf')
    ) -> None:
        self.compute_epsilon = compute_epsilon
        self.max_epsilon = max_epsilon

    def on_substep_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        raise ValueError("Shouldn't be called for DP.")

    def on_step_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        optimizer.zero_grad()  # Opacus is bothered that HF does not call .zero_grad() on the optimizer

    def on_save(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(state, control)

    def on_evaluate(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(control)

    def _check_max_epsilon_exceeded(self, control: TrainerControl) -> TrainerControl:
        if self.compute_epsilon() > self.max_epsilon:
            logger.error("Max epsilon exceeded. Stopping training...")
            control.should_training_stop = True
        return control


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here
    """
    @contextmanager
    def no_sync(self):
        yield


def create_author_mapping(dataset: Dataset, author: str) -> Sequence[Sequence[int]]:
    """
    Creates a mapping from authors to samples in a dataset.
    """
    with dataset.formatted_as(type="pandas"):
        authors = pd.DataFrame(data={"author": dataset[author]})
        author_mapping = [g.index.values for _, g in authors.groupby("author")]
    return author_mapping


class OpacusDPTrainer(Trainer):
    """
    Wrapper to modify Huggingface Trainer to:
        (i) remove "loss = loss / self.args.gradient_accumulation_steps" operation in training_step
        as this is already handled by Opacus package.
        (ii) enable author-level DP training by modifing the sampler and the dataloader. In the case
        of sample-level DP, each sample can be represented by a unique author.
        (iii) wrap the optimizer with Opacus' DPOptimizer/DistributedDPOptimizer
    """
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module] = None,
        args: arguments.TrainingArguments = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        if author_mapping is None:
            author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not isinstance(train_dataset, AuthorIndexedDataset):
            train_dataset = AuthorIndexedDataset(
                dataset=train_dataset,
                author_index=author_mapping,
                rng=torch.Generator().manual_seed(args.seed)
            )

        if not self.privacy_args.disable_dp:
            if self.train_args.gradient_accumulation_steps > 1:
                raise NotImplementedError(
                    "DP currently doesn't support gradient accumulation via the Huggingface trainer. "
                    "Use --max_physical_per_device_train_batch_size which will automatically limit "
                    "the number of samples simulatenously processed."
                )
            callbacks = callbacks or []
            callbacks.append(DPCallback(compute_epsilon=self.compute_epsilon))

            # Wrap model in DDP and GradSampleModule
            if args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
                logger.info(f"Wrapping the model with DPDDP in distributed training.")
                model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(model)

            self.privacy_engine = opacus.PrivacyEngine(secure_mode=self.privacy_args.secure_mode)

        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks, **kwargs)

        if not self.privacy_args.disable_dp:
            super().create_optimizer()
            self.non_dp_optimizer = self.optimizer

            if self.privacy_args.noise_multiplier is None:
                self.dp_model, self.dp_optimizer, self.dp_train_dataloader = self.privacy_engine.make_private_with_epsilon(
                    module=model,
                    data_loader=super().get_train_dataloader(),
                    optimizer=self.non_dp_optimizer,
                    max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                    target_epsilon=self.privacy_args.target_epsilon,
                    target_delta=self.privacy_args.target_delta,
                    epochs=self.train_args.num_train_epochs
                )
            else:
                self.dp_model, self.dp_optimizer, self.dp_train_dataloader = self.privacy_engine.make_private(
                    madule=model,
                    data_loader=super().get_train_dataloader(),
                    optimizer=self.non_dp_optimizer,
                    max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                    noise_multiplier=self.privacy_args.noise_multiplier,
                    target_delta=self.privacy_args.target_delta
                )
            self.model = self.dp_model
            self.optimizer = self.dp_optimizer

            self.dp_train_dataloader = wrap_data_loader(
                data_loader=self.dp_train_dataloader, 
                max_batch_size=self.privacy_args.max_physical_per_device_train_batch_size,
                optimizer=self.dp_optimizer
            )
        else:
            self.dp_model = None
            self.dp_optimizer = None
            self.dp_train_dataloader = None


    def compute_epsilon(self) -> float:
        self.privacy_engine.get_epsilon(self.privacy_args.target_delta)

    def create_optimizer(self):
        if self.privacy_args.disable_dp:
            super().create_optimizer()
        else:
            self.optimizer = self.dp_optimizer

    def get_train_dataloader(self) -> DataLoader:
        if self.privacy_args.disable_dp:
            return super().get_train_dataloader()
        else:
            return self.dp_train_dataloader
 