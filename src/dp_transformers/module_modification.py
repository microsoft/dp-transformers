# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
import torch
from transformers import GPT2Model, GPT2PreTrainedModel
from typing import List

from dp_transformers.layers.dp_merged_linear import DPMergedLinear

def convert_gpt2_attention_to_lora(model: GPT2Model, r: int = 0, lora_alpha: int = 1, 
                                   lora_dropout: float = 0., enable_lora: List[bool] = [False],
                                   fan_in_fan_out: bool = False, merge_weights: bool = True,
                                   **kwargs) -> GPT2Model:
    if not isinstance(model, GPT2PreTrainedModel):
        raise TypeError("Requires a GPT2 model")

    if not hasattr(model, "h") and hasattr(model, "transformer"):
        warnings.warn("""It looks like you have a model with a classification or LM head. """
                      """If this is the case, pass `model.transformer` to `convert_gpt2_attention_to_lora` to avoid this warning. """, UserWarning)
        transformer = model.transformer
    else:
        transformer = model

    for h_i in transformer.h:
        h_i.attn.c_attn = DPMergedLinear.from_transformers_conv1d(
            original_layer = h_i.attn.c_attn, r = r, lora_alpha = lora_alpha,
            lora_dropout = lora_dropout, enable_lora = enable_lora, fan_in_fan_out = fan_in_fan_out,
            merge_weights = merge_weights, **kwargs)

    return model


def force_causal_attention(model: GPT2Model):
    """
    Force a GPT2 model to use causal attention

    Some variants of GPT2 may use bi-directional attention for the context.
    This can cause issues when training in an auto-regressive fashion. This function forces causal attention
    """
    if not isinstance(model, GPT2Model):
        raise TypeError("Requires a GPT2 model")

    if not hasattr(model, "h") and hasattr(model, "transformer"):
        warnings.warn("""It looks like you have a model with a classification or LM head. """
                      """If this is the case, pass `model.transformer` to `force_causal_attention` to avoid this warning. """, UserWarning)
        transformer = model.transformer
    else:
        transformer = model


    for h_i in transformer.h:
        h_i.attn.bias = torch.tril(h_i.attn.bias)

    return model


    