import torch

from tempfile import TemporaryDirectory
from transformers import TrainingArguments, Trainer


def create_dummy_data(size: int, dim: int, rng: torch.Generator):
    return [{
        "input": torch.randn(dim, dtype=torch.float32, generator=rng),
        "labels": torch.randint(0, dim, (1,), dtype=torch.int64, generator=rng)
    } for _ in range(size)]


def compute_eval_loss(data, model):
    with TemporaryDirectory() as tmp_dir:
        train_args=TrainingArguments(
            per_device_train_batch_size=3,
            output_dir=tmp_dir,
            use_cpu=True,
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=train_args)
        results = trainer.evaluate(eval_dataset=data)
    return results["eval_loss"]
 

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
