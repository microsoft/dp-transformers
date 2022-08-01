# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from opacus.grad_sample import utils


def compute_tied_embedding_grad_sample(
    layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Embedding`` layer when input and output embeddings are tied.

    This is essentially the same code as native opacus provides, however, we accumulate grad samples
    rather than extending them. This is required if the model has input and output embeddings tied and
    therefore we compute two gradients for the same parameters which we need to add.

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    saved = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    batch_size = A.shape[batch_dim]
    index = (
        A.unsqueeze(-1)
        .expand(*A.shape, layer.embedding_dim)
        .reshape(batch_size, -1, layer.embedding_dim)
    )
    grad_sample = torch.zeros(
        batch_size, *layer.weight.shape, device=layer.weight.device
    )
    grad_sample.scatter_add_(1, index, B.reshape(batch_size, -1, layer.embedding_dim))
    torch.backends.cudnn.deterministic = saved

    utils.create_or_accumulate_grad_sample(layer.weight, grad_sample, layer)


def register_grad_sampler() -> None:
    utils.register_grad_sampler(nn.Embedding)(compute_tied_embedding_grad_sample)
