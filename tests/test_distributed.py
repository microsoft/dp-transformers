import pytest
import torch

from accelerate import PartialState, DistributedType
from tempfile import TemporaryDirectory
from transformers import set_seed, Trainer

from dp_transformers.dp_utils import OpacusDPTrainer
from dp_transformers.arguments import PrivacyArguments, TrainingArguments

from utils import SimpleModule, create_dummy_data, compute_eval_loss


@pytest.fixture(scope="module", autouse=True)
def initialize_dist():
    # Use CPU compatible backend to allow tests to run on machines without GPUs
    state = PartialState(cpu=True)
    yield


def skip_if_not_distributed():
    if PartialState().distributed_type != DistributedType.MULTI_CPU:
        pytest.skip("This test should only run in distributed mode")


def test_distributed_evaluation():
    skip_if_not_distributed()

    data_size = 8
    dim = 10

    rng = torch.Generator().manual_seed(2032)
    model = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(32908)
    data = create_dummy_data(data_size, dim, rng)

    eval_loss = compute_eval_loss(data=data, model=model)

    assert eval_loss == pytest.approx(3.023815393447876)


def test_distributed_non_dp_training_recovers_disabled_dp():
    skip_if_not_distributed()
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

    world_size = PartialState().num_processes
    assert batch_size % world_size == 0
    per_device_train_batch_size = batch_size // world_size

    privacy_args = PrivacyArguments(
        disable_dp=True,
    )
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
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
            per_device_train_batch_size=per_device_train_batch_size,
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


def test_dp_invariant_distribution():
    skip_if_not_distributed()
    eval_data_size = 8
    train_data_size = 8
    dim = 10
    batch_size = 4
    max_steps = 16

    rng = torch.Generator().manual_seed(2032)
    model_disabled_dp = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(9130)
    train_data = create_dummy_data(train_data_size, dim, rng)

    rng = torch.Generator().manual_seed(32908)
    eval_data = create_dummy_data(eval_data_size, dim, rng)

    world_size = PartialState().num_processes
    assert batch_size % world_size == 0
    per_device_train_batch_size = batch_size // world_size

    privacy_args = PrivacyArguments(
        disable_dp=False,
        noise_multiplier=0.0,
        per_sample_max_grad_norm=1000000,
        poisson_sampling=False,
    )
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
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

    eval_loss = compute_eval_loss(data=eval_data, model=model_disabled_dp)

    assert eval_loss == pytest.approx(3.2910091876983643)  # data from single CPU run
