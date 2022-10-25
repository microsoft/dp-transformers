# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import datasets
from datasets import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer, TrainerCallback, TrainerState, TrainerControl, logging,
    DataCollatorForLanguageModeling, PreTrainedTokenizer, training_args, modeling_utils
)
from transformers.file_utils import is_sagemaker_mp_enabled, is_datasets_available
import opacus
from prv_accountant import Accountant
from scipy import optimize
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union, Dict, Sequence

from . import sampler, arguments

logger = logging.get_logger(__name__)


class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """
    def __init__(
        self,
        noise_multiplier: float,
        target_delta: float,
        sampling_probability: float,
        compute_epsilon: Optional[Callable[[int], float]] = None,
        max_epsilon: float = float('inf')
    ) -> None:

        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.sampling_probability = sampling_probability
        self.accountant = opacus.accountants.RDPAccountant()

        self.max_epsilon = max_epsilon
        self.on_substep_end_was_called = False
        if not compute_epsilon:
            self.compute_epsilon = lambda _: self.accountant.get_epsilon(self.target_delta)
        else:
            self.compute_epsilon = compute_epsilon

    def on_substep_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        optimizer.signal_skip_step(do_skip=True)
        optimizer.step()
        optimizer.zero_grad()

        self.on_substep_end_was_called = True

    def on_step_begin(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs): 
        optimizer.signal_skip_step(do_skip=False)

    def on_step_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if not (
            args.gradient_accumulation_steps <= 1 or
            self.on_substep_end_was_called
        ):
            raise RuntimeError(
                "Gradient accumulation was specified but `on_substep_end` wasn't called. "
                "Make sure you're using a recent version of transformers (>=4.10.0) "
                "which has an appropriate callback in the trainer."
            )

        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        optimizer.zero_grad()  # Opacus is bothered that HF does not call .zero_grad() on the optimizer

        self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sampling_probability)

    def on_save(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(state, control)

    def on_evaluate(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(state, control)

    def _check_max_epsilon_exceeded(self, state: TrainerState, control: TrainerControl) -> TrainerControl:
        eps = self.compute_epsilon(state.global_step)
        if eps > self.max_epsilon:
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


def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float=0.1) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    def compute_epsilon(mu: float) -> float:
        acc = Accountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps,
            eps_error=eps_error/2
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]


    has_converged = False 
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error
    return bracket[1]


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
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        accountant_fn: callable = None,
        use_prv_accountant: bool = True,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        if author_mapping is None:
            author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
                num_samples=len(self.author_mapping),
            )

        # Wrap model in DDP and GradSampleModule
        if args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            logger.info(f"Wrapping the model with DPDDP in distributed training.")
            model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(model)

        model = GradSampleModule(model)

        # Instantiate alternative accountant
        if use_prv_accountant:
            self.privacy_accountant = Accountant(
                noise_multiplier=self.noise_multiplier,
                sampling_probability=self.sampling_probability,
                delta=self.privacy_params.target_delta,
                eps_error=0.1,
                max_compositions=self.num_steps
            )
            accountant_fn = lambda s: self.privacy_accountant.compute_epsilon(s)[2]
        else:
            accountant_fn = None

        # Set up callback for accounting and handling grad acc
        self.dp_callback = DPCallback(
            noise_multiplier=self.noise_multiplier,
            target_delta=self.privacy_params.target_delta,
            sampling_probability=self.sampling_probability,
            compute_epsilon=accountant_fn
        )
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=[self.dp_callback], **kwargs)

        self.get_rdp_epsilon = lambda: self.dp_callback.accountant.get_epsilon(self.privacy_params.target_delta)  # RDP epsilon
        self.get_accountant_epsilon = lambda: self.privacy_accountant.compute_epsilon(self.state.global_step)[2]

    @property
    def sampling_probability(self) -> float:
        return self.train_args.per_device_train_batch_size * self.train_args.world_size * \
            self.train_args.gradient_accumulation_steps / len(self.author_mapping)

    @property
    def num_steps(self) -> int:
        return int(self.train_args.num_train_epochs * (1 / self.sampling_probability + 1))

    def create_optimizer(self):
        _ = super().create_optimizer()

        if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            optimizer_generator = opacus.optimizers.DistributedDPOptimizer
        else:
            optimizer_generator = opacus.optimizers.DPOptimizer

        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
        )

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            raise NotImplementedError("DP currently doesn't support this")

        if self.use_cuda_amp or self.use_cpu_amp:
            raise NotImplementedError("DP currently doesn't support this.")
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Compared to the original HF implementation, we have to remove the loss scaling by the number of gradient
        # accumulation steps since opacus scales the gradients accordingly. However, we still need to scale the loss
        # that is returned in order for the logging to work correctly. Hence we scale the loss after the call to 
        # loss.backward()

        if self.use_cuda_amp or self.use_cpu_amp:
            raise NotImplementedError("DP currently doesn't support this")
        elif self.use_apex:
            raise NotImplementedError("DP currently doesn't support this")
        elif self.deepspeed:
            raise NotImplementedError("DP currently doesn't support this")
        else:
            loss.backward()

        return loss.detach()/self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        """
        Provides author sampler.
        """
        train_sampler = sampler.ShuffledAuthorSampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size
        )
        return train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use the author-level sampler from dp_transformers.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )