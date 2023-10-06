# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import torch
import torch.nn as nn
from opt_einsum import contract
from typing import List

from opacus.grad_sample.utils import register_grad_sampler

from transformers.modeling_utils import Conv1D


@register_grad_sampler(Conv1D)
def compute_transformers_conv1d_grad_sample(
    layer: Conv1D, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        ret[layer.weight] = contract("n...i,n...j->nji", backprops, activations).contiguous()
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops)
    return ret
