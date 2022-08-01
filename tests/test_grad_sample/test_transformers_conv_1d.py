# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from opacus.tests.grad_samples.common import GradSampleHooks_test

from transformers.modeling_utils import Conv1D

from dp_transformers.grad_sample.transformers.conv_1d import register_grad_sampler


class TestConv1D(GradSampleHooks_test):
    def test_grad_sample(self):
        """
        Verify that our custom implementation of the grad sample for huggingface's Conv1D
        layer works. We largely build on the test routines in opacus's library.
        """
        register_grad_sampler()
        x = torch.randn(16, 8)
        layer = Conv1D(4, 8)
        self.run_test(x, layer, batch_first=True)

        self.run_test(torch.randn(24, 8, 8), Conv1D(4, 8), batch_first=True)
