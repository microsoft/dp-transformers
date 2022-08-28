# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from opacus.tests.grad_samples.common import GradSampleHooks_test
from transformers.models.t5.modeling_t5 import T5LayerNorm

from dp_transformers.grad_sample.transformers.t5_layer_norm import register_grad_sampler


class TestConv1D(GradSampleHooks_test):
    def test_grad_sample(self):
        """
        Verify that our custom implementation of the grad sample for huggingface's T5LayerNorm
        layer works. We largely build on the test routines in opacus's library.
        """
        register_grad_sampler()
        x = torch.randn(1, 16)
        layer = T5LayerNorm(16)
        self.run_test(x, layer, batch_first=True)

        self.run_test(torch.randn(24, 16, 8), T5LayerNorm(8), batch_first=True)
