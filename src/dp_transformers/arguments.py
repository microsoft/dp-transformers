# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional

import numpy as np
from scipy import optimize
from transformers import TrainingArguments as HfTrainingArguments
from transformers import IntervalStrategy, logging
from dataclasses import dataclass, field
from datasets.utils import disable_progress_bar
from prv_accountant import Accountant

logger = logging.get_logger(__name__)


@dataclass
class PrivacyArguments:
    per_sample_max_grad_norm: Optional[float] = field(default=None, metadata={"help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=None, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=None, metadata={
        "help": "Target delta, defaults to 1/N"
    })
    disable_dp: bool = field(default=False, metadata={
        "help": "Disable DP training."
    })
    secure_mode: bool = field(default=False, metadata={
        "help": "Use secure mode for DP-SGD."
    })
    max_physical_per_device_train_batch_size: Optional[int] = field(default=None, metadata={
        "help": "Maximum physical batch size per device for training."
    })

    def __post_init__(self):
        if self.disable_dp:
            logger.warning("Disabling differentially private training...")
            self.noise_multiplier = 0.0
            self.per_sample_max_grad_norm = float('inf')
            self.target_epsilon = None
            if self.max_physical_per_device_train_batch_size is not None:
                raise ValueError("DP training is disabled, --max_physical_per_device_train_batch_size is not needed.")
        else:
            if bool(self.target_epsilon) == bool(self.noise_multiplier):
                raise ValueError("Exactly one of the arguments --target_epsilon and --noise_multiplier must be used.")
            if self.per_sample_max_grad_norm is None:
                raise ValueError("DP training requires --per_sample_max_grad_norm argument.")


@dataclass
class TrainingArguments(HfTrainingArguments):
    dry_run: bool = field(
        default=False,
        metadata={"help": "Option for reducing training steps (2) and logging intervals (1) for quick sanity checking of arguments."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.dry_run:
            logger.warning("--dry_run was specified. Reducing number of training steps to 2 and logging intervals to 1...")
            self.logging_steps = 1
            self.logging_strategy = IntervalStrategy.STEPS
            self.eval_steps = 1
            self.evaluation_strategy = IntervalStrategy.STEPS

            self.max_steps = 2

        if self.disable_tqdm:
            disable_progress_bar()


def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float=0.1) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    def compute_epsilon(mu: float) -> float:
        acc = Accountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps,
            eps_error=eps_error/2
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]

    has_converged = False 
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error

    return bracket[1]