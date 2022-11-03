# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from opacus import PrivacyEngine
from opacus.dp_model_inspector import IncompatibleModuleException
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from dp_transformers.grad_sample.transformers import register_grad_sampler_gpt2, register_grad_sampler_t5


def test_gpt2_grad_sample_layers_registered():
    """
    Test whether all layers in GPT2 are registered in the grad sampler.
    """
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.train()

    engine = PrivacyEngine(model, sample_rate=1e-3, max_grad_norm=1e10, noise_multiplier=1.0)

    # We haven't registered the grad samples yet so make sure that it actually fails
    with pytest.raises(IncompatibleModuleException):
        engine.validator.validate(model)

    # Register the grad samples
    register_grad_sampler_gpt2()

    # Now make sure that it works
    engine.validator.validate(model)


def test_t5_grad_sample_layers_registered():
    """
    Test whether all layers in T5 are registered in the grad sampler.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    model.train()

    engine = PrivacyEngine(model, sample_rate=1e-3, max_grad_norm=1e10, noise_multiplier=1.0)

    # We haven't registered the grad samples yet so make sure that it actually fails
    with pytest.raises(IncompatibleModuleException):
        engine.validator.validate(model)

    # Register the grad samples
    register_grad_sampler_t5()

    # Now make sure that it works
    engine.validator.validate(model)
