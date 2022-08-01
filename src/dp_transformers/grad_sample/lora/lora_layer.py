# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from dp_transformers.layers.dp_merged_linear import Conv1DZeroInit

from opacus.grad_sample import utils


def compute_lora_conv_1d_zero_init_grad_sample(
    layer: Conv1DZeroInit, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0) -> None:
    
    B = B.transpose(-2, -1)
    in_features = layer.in_features
    out_features = layer.out_features
    
    if layer.groups == 1:
        d_loraB_B = torch.einsum("nki,njk->nij", B, A).unsqueeze(-1)
    elif layer.groups == 2:
        gs1 = torch.einsum("nki,njk->nij", B[:, :, :out_features//2], A[:, :in_features, :])
        gs2 = torch.einsum("nki,njk->nij", B[:, :, out_features//2:], A[:, in_features:, :])
        d_loraB_B = torch.cat((gs1, gs2), 1).unsqueeze(-1)
    elif layer.groups == 3:
        gs1 = torch.einsum("nki,njk->nij", B[:, :, :out_features//3], A[:, :in_features, :])
        gs2 = torch.einsum("nki,njk->nij", B[:, :, out_features//3:2*out_features//3], A[:, in_features:2*in_features, :])
        gs3 = torch.einsum("nki,njk->nij", B[:, :, 2*out_features//3:], A[:, 2*in_features:3*in_features, :])
        d_loraB_B = torch.cat((gs1, gs2, gs3), 1).unsqueeze(-1)
    else:
        raise Exception("groups can only be 1, 2, or 3 in LoRA")
    
    utils.create_or_extend_grad_sample(layer.weight, d_loraB_B, batch_dim)


def register_grad_sampler() -> None:
    utils.register_grad_sampler(Conv1DZeroInit)(compute_lora_conv_1d_zero_init_grad_sample)
