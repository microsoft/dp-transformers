# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import transformers
import pytest
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
from dp_transformers.layers.dp_merged_linear import DPMergedLinear

def test_convert_gpt2_attention_to_lora():
    model = transformers.AutoModel.from_pretrained("distilgpt2")

    model = convert_gpt2_attention_to_lora(model, r=3)

    assert isinstance(model.h[0].attn.c_attn, DPMergedLinear)

def test_convert_gpt2_attention_to_lora_warning():
    model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")

    with pytest.warns(UserWarning):
        model = convert_gpt2_attention_to_lora(model, r=3)

    assert isinstance(model.transformer.h[0].attn.c_attn, DPMergedLinear)

