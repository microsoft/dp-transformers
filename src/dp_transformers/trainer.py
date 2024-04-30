import torch
import opacus

from typing import Dict, Optional, Sequence, Union, List
from transformers import Trainer, modeling_utils, TrainerCallback, DataCollator, training_args, logging
from opacus.utils.batch_memory_manager import wrap_data_loader
from torch.utils.data import DataLoader

from dp_transformers.data import AuthorIndexedDataset
from dp_transformers import arguments
from dp_transformers.callbacks import DPCallback


logger = logging.get_logger(__name__)


class DPTrainer(Trainer):
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module] = None,
        args: arguments.TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        if train_dataset is None:
            train_dataset = []

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

        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, callbacks=callbacks,
                         **kwargs)

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
                    epochs=self.train_args.num_train_epochs,
                    poisson_sampling=self.privacy_args.poisson_sampling,
                )
            else:
                self.dp_model, self.dp_optimizer, self.dp_train_dataloader = self.privacy_engine.make_private(
                    module=model,
                    data_loader=super().get_train_dataloader(),
                    optimizer=self.non_dp_optimizer,
                    max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
                    noise_multiplier=self.privacy_args.noise_multiplier,
                    poisson_sampling=self.privacy_args.poisson_sampling,
                )
            self.model = self.dp_model
            self.optimizer = self.dp_optimizer

            # Use the regular batch size if no max_physical_per_device_train_batch_size is provided
            max_batch_size = self.privacy_args.max_physical_per_device_train_batch_size or self.train_args.per_device_train_batch_size

            self.dp_train_dataloader = wrap_data_loader(
                data_loader=self.dp_train_dataloader, 
                max_batch_size=max_batch_size,
                optimizer=self.dp_optimizer
 
            )
            if data_collator is not None:
                self.dp_train_dataloader.collate_fn = DataCollatorWithEmptyWrapper.from_batch(
                    original_collator=data_collator,
                    batch=next(iter(super().get_train_dataloader()))
                )
        else:
            self.dp_model = None
            self.dp_optimizer = None
            self.dp_train_dataloader = None

    def compute_epsilon(self) -> float:
        if self.privacy_args.disable_dp:
            return float('inf')
        else:
            return self.privacy_engine.get_epsilon(self.privacy_args.target_delta)

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
 