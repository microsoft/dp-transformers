# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dp_transformers.arguments import find_noise_multiplier
from prv_accountant import Accountant
import pytest

class TestFindNoiseMultiplier:
    def test_sensible_range(self):
        mu = find_noise_multiplier(2e-3, 10_000, 4.0, 1e-7)
        assert 0 < mu and mu < 2 # Check that mu is in a sensible interval

    def test_inverse(self):
        mu = find_noise_multiplier(2e-3, 10_000, 4.0, 1e-7)
        acc = Accountant(mu, 2e-3, 1e-7, 10_000, eps_error = 0.5)
        eps = acc.compute_epsilon(10_000)
        assert eps[2] == pytest.approx(4, abs=0.5)

    def test_robustness(self):
        with pytest.warns(None) as record:
            mu = find_noise_multiplier(
                sampling_probability=256/50_000,
                num_steps=int(50*50_000/256),
                target_epsilon=10.0,
                target_delta=1e-5
            )
        assert len(record) == 0

    def test_robustness_2(self):
        mu = find_noise_multiplier(
            sampling_probability=0.26058631921824105,
            num_steps=18800,
            target_delta=0.00011448277499759097,
            target_epsilon=4.0
        )
        # Just test that this doesn't cause a floating point overflow
        print(mu)