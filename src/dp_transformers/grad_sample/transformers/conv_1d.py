# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import torch
import torch.nn as nn
from opt_einsum import contract

from opacus.grad_sample.utils import register_grad_sampler

from transformers.modeling_utils import Conv1D

@register_grad_sampler(Conv1D)
def compute_transformers_conv1d_grad_sample(
    layer: Conv1D, A: torch.Tensor, B: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    ret = {}
    ret[layer.weight] = contract("n...i,n...j->nji", B, A).contiguous()
    if layer.bias is not None:
        ret[layer.bias] = contract("n...k->nk", B)
    return ret
