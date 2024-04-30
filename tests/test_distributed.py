import pytest
import torch
import torch.distributed as dist

from accelerate import PartialState, DistributedType
from tempfile import TemporaryDirectory

from dp_transformers.dp_utils import OpacusDPTrainer
from dp_transformers.arguments import PrivacyArguments, TrainingArguments


@pytest.fixture(scope="module", autouse=True)
def initialize_dist():
    # Use CPU compatible backend to allow tests to run on machines without GPUs
    state = PartialState(cpu=True)
    yield


def test_distributed_operation():
    if PartialState().num_processes < 2:
        pytest.skip("Test requires at least 2 processes")
    rank = dist.get_rank()
    tensor = torch.tensor([rank], dtype=torch.int32)
    output = [torch.empty(1, dtype=torch.int32) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    assert torch.cat(output).tolist() == list(range(dist.get_world_size()))


def test_distributed_operation_2():
    if PartialState().num_processes < 2:
        pytest.skip("Test requires at least 2 processes")
    rank = dist.get_rank()
    tensor = torch.tensor([rank], dtype=torch.int32)
    output = [torch.empty(1, dtype=torch.int32) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    assert torch.cat(output).tolist() == list(range(dist.get_world_size()))


class SimpleModule(torch.nn.Module):
    def __init__(self, dim: int, rng: torch.Generator):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.fc.weight, generator=rng)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, input: torch.Tensor, labels: torch.Tensor):
        output = self.fc(input)
        loss = torch.nn.functional.cross_entropy(output, labels.view(-1))
        return {"loss": loss}


def create_dummy_data(size: int, dim: int, rng: torch.Generator):
    return [{
        "input": torch.randn(dim, dtype=torch.float32, generator=rng),
        "labels": torch.randint(0, dim, (1,), dtype=torch.int64, generator=rng)
    } for _ in range(size)]


def compute_eval_loss(data, model):
    privacy_args = PrivacyArguments(
        disable_dp=True,
    )
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=3,
            output_dir=tmp_dir,
            use_cpu=True,
            remove_unused_columns=False,
        )
        trainer = OpacusDPTrainer(
            model=model,
            args=train_args,
            privacy_args=privacy_args,
        )
        results = trainer.evaluate(eval_dataset=data)
    return results["eval_loss"]
 

def test_distributed_evaluation():
    data_size = 8
    dim = 10

    rng = torch.Generator().manual_seed(2032)
    model = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(32908)
    data = create_dummy_data(data_size, dim, rng)

    eval_loss = compute_eval_loss(data=data, model=model)

    assert eval_loss == pytest.approx(3.023815393447876)


def test_distributed_training():
    eval_data_size = 8
    train_data_size = 12
    dim = 10

    rng = torch.Generator().manual_seed(2032)
    model = SimpleModule(dim, rng)

    rng = torch.Generator().manual_seed(9130)
    train_data = create_dummy_data(train_data_size, dim, rng)

    rng = torch.Generator().manual_seed(32908)
    eval_data = create_dummy_data(eval_data_size, dim, rng)

    privacy_args = PrivacyArguments(
        disable_dp=False,
        noise_multiplier=0.1,
        per_sample_max_grad_norm=1.0,
        max_physical_per_device_train_batch_size=3,
        poisson_sampling=False,
    )
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=3,
            output_dir=tmp_dir,
            max_steps=3,
            use_cpu=True,
            remove_unused_columns=False,
        )
        trainer = OpacusDPTrainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            privacy_args=privacy_args,
        )
        trainer.train()

    eval_loss = compute_eval_loss(data=eval_data, model=model)


# Test's to implement
# - Disabling DP in DP Trainer yields same result as non-DP Trainer
# - Scaling number of processes gives the same results for DP Trainer
