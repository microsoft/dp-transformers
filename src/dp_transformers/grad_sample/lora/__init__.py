# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .lora_layer import register_grad_sampler as _register_grad_sampler_lora_layer

def register_grad_sampler_gpt2_lora() -> None:
    """
    Register gradient samplers for LoRA's GPT-2 integration with Opacus's per-sample-gradient-computation hooks.

    This function needs to be called before the training is started and it will register
    all necessary methods globally.
    """
    _register_grad_sampler_lora_layer()