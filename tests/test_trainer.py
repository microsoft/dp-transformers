import torch
import pytest

from tempfile import TemporaryDirectory
from transformers import set_seed, Trainer
from accelerate import PartialState, DistributedType

from dp_transformers.dp_utils import OpacusDPTrainer
from dp_transformers.arguments import PrivacyArguments, TrainingArguments

from utils import SimpleModule, create_dummy_data, compute_eval_loss


def test_distributed_non_dp_training_recovers_disabled_dp():
    """
    Ensure that if we use the DPTrainer but disable DP, we get the same results as non-DP training.
    """
    if PartialState().distributed_type != DistributedType.NO:
        pytest.skip("This test should only run in non-distributed mode")

    eval_data_size = 8
    train_data_size = 8
    dim = 10
    batch_size = 4
    max_steps = 16

    rng = torch.Generator().manual_seed(2032)
    model_non_dp = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(2032)
    model_disabled_dp = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(9130)
    train_data = create_dummy_data(train_data_size, dim, rng)

    rng = torch.Generator().manual_seed(32908)
    eval_data = create_dummy_data(eval_data_size, dim, rng)

    privacy_args = PrivacyArguments(
        disable_dp=True,
    )
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            output_dir=tmp_dir,
            max_steps=max_steps,
            use_cpu=True,
            remove_unused_columns=False,
            learning_rate=0.1,
        )
        set_seed(9230)
        trainer = OpacusDPTrainer(
            model=model_disabled_dp,
            args=train_args,
            train_dataset=train_data,
            privacy_args=privacy_args,
        )
        trainer.train()

    eval_loss_disabled_dp = compute_eval_loss(data=eval_data, model=model_disabled_dp)

    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            output_dir=tmp_dir,
            max_steps=max_steps,
            use_cpu=True,
            remove_unused_columns=False,
            learning_rate=0.1,
        )
        set_seed(9230)
        trainer = Trainer(
            model=model_non_dp,
            args=train_args,
            train_dataset=train_data,
        )
        trainer.train()

    eval_loss_non_dp = compute_eval_loss(data=eval_data, model=model_non_dp)

    assert eval_loss_disabled_dp == pytest.approx(eval_loss_non_dp)
