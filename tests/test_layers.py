# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
import numpy as np
import transformers

from typing import List

from loralib import MergedLinear
from transformers import set_seed
from torch.optim import SGD

from dp_transformers.layers.dp_merged_linear import DPMergedLinear, TransformersConv1D

torch.backends.cudnn.deterministic = True

class TestDPMergedLinear:
    @pytest.mark.parametrize("enable_lora", [[True, False, True], [True, False, False]])
    @pytest.mark.parametrize("r", [0, 3])
    @pytest.mark.parametrize("alpha", [1, 2])
    @pytest.mark.parametrize("dropout", [0, 0.1]) 
    @pytest.mark.parametrize("fan_in_fan_out", [False, pytest.param(True, marks=pytest.mark.xfail)])
    @pytest.mark.parametrize("merge_weights", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    def test_equivalent_to_loralib(self, enable_lora: List[bool], r: int,
                                   alpha: int, dropout: float, fan_in_fan_out: bool,
                                   merge_weights: bool, bias: bool):
        set_seed(193102)
        orig_ml = MergedLinear(
            in_features = 1024,
            out_features = 1024*len(enable_lora),
            r = r,
            lora_alpha = alpha, 
            lora_dropout = dropout,
            enable_lora = enable_lora,
            fan_in_fan_out = fan_in_fan_out,
            merge_weights = merge_weights,
            bias = bias
        )
        set_seed(193102)
        orig_ml.reset_parameters()

        set_seed(193102)
        dp_ml = DPMergedLinear(
            in_features = 1024,
            out_features = 1024*len(enable_lora),
            r = r,
            lora_alpha = alpha, 
            lora_dropout = dropout,
            enable_lora = enable_lora,
            fan_in_fan_out = fan_in_fan_out,
            merge_weights = merge_weights,
            bias = bias
        )
        set_seed(193102)
        dp_ml.reset_parameters()

        self.assert_output_same(orig_ml=orig_ml, dp_ml=dp_ml, seed=192302)

        opt_orig = SGD(orig_ml.parameters(), lr=1e-2)
        opt_dp = SGD(dp_ml.parameters(), lr=1e-2)

        x = torch.randn((16, 32, 1024))

        set_seed(193102)
        x_orig = orig_ml(x)
        set_seed(193102)
        x_dp = dp_ml(x)

        y = torch.randn((16, 32, 1024*len(enable_lora)))
        loss_orig = (x_orig.view(-1) - y.view(-1)).norm()
        loss_dp = (x_dp.view(-1) - y.view(-1)).norm()

        assert loss_orig.item() == pytest.approx(loss_dp.item())

        loss_orig.backward()
        loss_dp.backward()

        opt_orig.step()
        opt_dp.step()

        self.assert_output_same(orig_ml=orig_ml, dp_ml=dp_ml, seed=129)


    def assert_output_same(self, orig_ml: MergedLinear, dp_ml: DPMergedLinear, seed: int):
        x = torch.randn((16, 32, 1024))

        set_seed(seed)
        x_orig = orig_ml(x)
        set_seed(seed)
        x_dp = dp_ml(x)

        # Check output is the same
        np.testing.assert_array_almost_equal(x_orig.detach(), x_dp.detach())

    def test_from_transformer_conv1d(self):
        m:transformers.GPT2LMHeadModel = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")
        original_layer = m.transformer.h[0].attn.c_attn

        assert isinstance(original_layer, TransformersConv1D)

        lora_layer = DPMergedLinear.from_transformers_conv1d(original_layer, r = 3)

        # The original Conv1D is a transpose linear. So transpose here to make sure we can compare
        np.testing.assert_array_almost_equal(lora_layer.linear.weight.T.detach(), original_layer.weight.detach())

    def test_from_transformer_conv1d_gpt2medium(self):
        m:transformers.GPT2LMHeadModel = transformers.AutoModelForCausalLM.from_pretrained("gpt2-medium")
        original_layer = m.transformer.h[0].attn.c_attn

        assert isinstance(original_layer, TransformersConv1D)

        lora_layer = DPMergedLinear.from_transformers_conv1d(original_layer, r=3, enable_lora=[True, False, True])

        # The original Conv1D is a transpose linear. So transpose here to make sure we can compare
        np.testing.assert_array_almost_equal(lora_layer.linear.weight.T.detach(), original_layer.weight.detach())
