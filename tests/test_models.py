# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from transformers import AutoModelForCausalLM
from opacus.validators import ModuleValidator
from opacus.validators.errors import UnsupportedModuleError


@pytest.mark.xfail(reason='functorch can deal with module in Opacus 1.2')
def test_gpt2_grad_sample_layers_registered():
    """
    Test whether all layers in GPT2 are registered in the grad sampler.
    """
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.train()

    validator = ModuleValidator()

    # We haven't registered the grad samples yet so make sure that it actually fails
    with pytest.raises(UnsupportedModuleError):
        validator.validate(model, strict=True)

    # Register the grad samples
    from dp_transformers.grad_sample.transformers import conv_1d

    # Now make sure that it works
    validator.validate(model, strict=True)
