# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from opacus.grad_sample import utils
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n
from transformers.models.t5.modeling_t5 import T5LayerNorm


def compute_transformers_t5_layer_norm_grad_sample(
        layer: T5LayerNorm, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for T5LayerNorm

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    # layer norm should always be calculated in float32
    variance = A.to(torch.float32).pow(2).mean(-1, keepdim=True)
    t5_layer_norm = A * torch.rsqrt(variance + layer.variance_epsilon)

    utils.create_or_extend_grad_sample(
        layer.weight,
        sum_over_all_but_batch_and_last_n(
            t5_layer_norm * B,
            layer.weight.dim(),
        ),
        batch_dim,
    )


def register_grad_sampler() -> None:
    utils.register_grad_sampler(T5LayerNorm)(compute_transformers_t5_layer_norm_grad_sample)
