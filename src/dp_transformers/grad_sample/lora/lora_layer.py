# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import torch
import torch.nn as nn
from opt_einsum import contract

from opacus.grad_sample.utils import register_grad_sampler

from dp_transformers.layers.dp_merged_linear import Conv1DZeroInit

@register_grad_sampler(Conv1DZeroInit)
def compute_lora_conv_1d_zero_init_grad_sample(
    layer: Conv1DZeroInit, A: torch.Tensor, B: torch.Tensor) -> Dict[nn.Parameter, torch.Tensor]:

    ret = {}
    
    B = B.transpose(-2, -1)
    in_features = layer.in_features
    out_features = layer.out_features
    
    if layer.groups == 1:
        ret[layer.weight] = contract("nki,njk->nij", B, A).unsqueeze(-1)
    elif layer.groups == 2:
        gs1 = contract("nki,njk->nij", B[:, :, :out_features//2], A[:, :in_features, :])
        gs2 = contract("nki,njk->nij", B[:, :, out_features//2:], A[:, in_features:, :])
        ret[layer.weight] = torch.cat((gs1, gs2), 1).unsqueeze(-1)
    elif layer.groups == 3:
        gs1 = contract("nki,njk->nij", B[:, :, :out_features//3], A[:, :in_features, :])
        gs2 = contract("nki,njk->nij", B[:, :, out_features//3:2*out_features//3], A[:, in_features:2*in_features, :])
        gs3 = contract("nki,njk->nij", B[:, :, 2*out_features//3:], A[:, 2*in_features:3*in_features, :])
        ret[layer.weight] = torch.cat((gs1, gs2, gs3), 1).unsqueeze(-1)
    else:
        raise Exception("groups can only be 1, 2, or 3 in LoRA")
    
    return ret
