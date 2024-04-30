import torch


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
