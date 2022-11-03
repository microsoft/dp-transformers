# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import transformers

from dp_transformers.layers.dp_merged_linear import DPMergedLinear
from dp_transformers.layers.relative_position_embedding import RelativePositionEmbedding
from dp_transformers.module_modification import convert_gpt2_attention_to_lora, \
    convert_t5_attention_nn_embedding_to_relative_position_embedding


def test_convert_gpt2_attention_to_lora():
    model = transformers.AutoModel.from_pretrained("distilgpt2")

    model = convert_gpt2_attention_to_lora(model, r=3)

    assert isinstance(model.h[0].attn.c_attn, DPMergedLinear)


def test_convert_gpt2_attention_to_lora_warning():
    model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")

    with pytest.warns(UserWarning):
        model = convert_gpt2_attention_to_lora(model, r=3)

    assert isinstance(model.transformer.h[0].attn.c_attn, DPMergedLinear)


def test_convert_t5_attention_nn_embedding_to_relative_position_embedding():
    model = transformers.AutoModel.from_pretrained("t5-base")

    model = convert_t5_attention_nn_embedding_to_relative_position_embedding(model, max_train_batch_size=16)

    assert isinstance(model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias, RelativePositionEmbedding)
    assert isinstance(model.decoder.block[0].layer[0].SelfAttention.relative_attention_bias, RelativePositionEmbedding)
