# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This is a DP compatible implementation of MergedLinear from LoRA
import math
import torch
from torch import Tensor
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import Conv1D as TransformersConv1D

from torch.nn import functional as F

from typing import List

def identity(x):
    return x

class Conv1DZeroInit(nn.Module):
    def __init__(self, in_features: int, out_features: int, groups: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty((out_features, in_features,1), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(input, self.weight, groups=self.groups)


class DPMergedLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, **kwargs)
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features=in_features, out_features=r*sum(enable_lora), bias=False)
            self.lora_B = Conv1DZeroInit(in_features=r, out_features=out_features // len(enable_lora) * sum(enable_lora), groups=sum(enable_lora))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.linear.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.linear.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.linear.weight.data = self.linear.weight.data.T
            raise NotImplementedError()

    @staticmethod
    def from_transformers_conv1d(original_layer: TransformersConv1D,
                                 r: int = 0, lora_alpha: int = 1, 
                                 lora_dropout: float = 0., enable_lora: List[bool] = [False],
                                 fan_in_fan_out: bool = False, merge_weights: bool = True,
                                 **kwargs) -> "DPMergedLinear":
        lora_layer = DPMergedLinear(
            in_features = original_layer.weight.shape[0], out_features = original_layer.weight.shape[1],
            r = r, lora_alpha = lora_alpha, lora_dropout = lora_dropout, enable_lora = enable_lora,
            fan_in_fan_out = fan_in_fan_out, merge_weights = merge_weights, **kwargs
        )
        assert lora_layer.linear.weight.shape == original_layer.weight.T.shape
        lora_layer.linear.weight = torch.nn.parameter.Parameter(original_layer.weight.T)
        lora_layer.linear.bias = original_layer.bias
        return lora_layer


    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return self.linear(x)
        else:
            result = self.linear(x)
            if self.r > 0:
                after_A = self.lora_A(self.lora_dropout(x))
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A.reset_parameters()
            self.lora_B.reset_parameters()


def mark_only_lora_as_trainable(model: torch.nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError