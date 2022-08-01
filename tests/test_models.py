# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from transformers import AutoModelForCausalLM
from opacus import PrivacyEngine
from opacus.dp_model_inspector import IncompatibleModuleException

from dp_transformers.grad_sample.transformers import register_grad_sampler_gpt2


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
