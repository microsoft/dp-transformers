# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from transformers.modeling_utils import Conv1D

from opacus.grad_sample import utils


def compute_transformers_conv1d_grad_sample(
    layer: Conv1D, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    gs = torch.einsum("n...i,n...j->nji", B, A).contiguous()
    utils.create_or_extend_grad_sample(
        layer.weight, gs, batch_dim
    )
    if layer.bias is not None:
        utils.create_or_extend_grad_sample(
            layer.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,  # pyre-ignore[6] We know layer.bias is not None
        )


def register_grad_sampler() -> None:
    utils.register_grad_sampler(Conv1D)(compute_transformers_conv1d_grad_sample)
