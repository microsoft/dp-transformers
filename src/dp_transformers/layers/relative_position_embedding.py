import torch
from torch import nn


class RelativePositionEmbedding(nn.Embedding):
    training_batch_size: int

    def __init__(self, training_batch_size: int, num_embeddings: int, embedding_dim: int, **kwargs):
        self.training_batch_size = training_batch_size
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        return super().forward(x)
