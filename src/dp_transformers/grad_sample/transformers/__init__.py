# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .conv_1d import register_grad_sampler as _register_grad_sampler_conv_1d
from .tied_embedding import register_grad_sampler as _register_grad_sampler_tied_embedding


def register_grad_sampler_gpt2() -> None:
    """
    Register gradient samplers for GPT-2 with Opacus's per-sample-gradient-computation hooks.

    This function needs to be called before the training is started and it will register
    all necessary methods globally.
    """
    _register_grad_sampler_conv_1d()
    _register_grad_sampler_tied_embedding()
