import pytest
import torch
import torch.distributed as dist

from accelerate import PartialState
from tempfile import TemporaryDirectory

from dp_transformers.dp_utils import OpacusDPTrainer
from dp_transformers.arguments import PrivacyArguments, TrainingArguments


@pytest.fixture(scope="module", autouse=True)
def initialize_dist():
    # Use CPU compatible backend to allow tests to run on machines without GPUs
    state = PartialState(cpu=True)
    yield


def test_distributed_operation():
    rank = dist.get_rank()
    tensor = torch.tensor([rank], dtype=torch.int32)
    output = [torch.empty(1, dtype=torch.int32) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    assert torch.cat(output).tolist() == list(range(dist.get_world_size()))


def test_distributed_operation_2():
    rank = dist.get_rank()
    tensor = torch.tensor([rank], dtype=torch.int32)
    output = [torch.empty(1, dtype=torch.int32) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    assert torch.cat(output).tolist() == list(range(dist.get_world_size()))


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def test_distributed_module():
    batch_size = 8
    model = SimpleModule()
    for p in model.parameters():
        p.data.fill_(1)
    input = torch.tensor(range(batch_size*10), dtype=torch.float32).view(batch_size, 10)
    label = torch.tensor(range(batch_size), dtype=torch.int64)
    output = model(input)

    privacy_args = PrivacyArguments(
        disable_dp=True,
    )

    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=3,
            output_dir=tmp_dir,
        )

        trainer = OpacusDPTrainer(
            model = model,
            train_dataset = [input],
            args=train_args,
            privacy_args = privacy_args,
        )

    # compute loss
    loss = torch.nn.functional.cross_entropy(output, label)
    assert loss.item() == pytest.approx(2.3025851249694824)
    
