from typing import Callable
from transformers import TrainerCallback, training_args, TrainerState, TrainerControl, logging


logger = logging.get_logger(__name__)


class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """
    def __init__(
        self,
        compute_epsilon: Callable[[], float],
        max_epsilon: float = float('inf')
    ) -> None:
        self.compute_epsilon = compute_epsilon
        self.max_epsilon = max_epsilon

    def on_substep_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        raise RuntimeError("Shouldn't be called for DP. Set --gradient_accumulation_steps to 1.")

    def on_step_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        optimizer.zero_grad()  # Opacus is bothered that HF does not call .zero_grad() on the optimizer

    def on_save(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(control)

    def on_evaluate(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(control)

    def _check_max_epsilon_exceeded(self, control: TrainerControl) -> TrainerControl:
        if self.compute_epsilon() > self.max_epsilon:
            logger.error("Max epsilon exceeded. Stopping training...")
            control.should_training_stop = True
        return control
