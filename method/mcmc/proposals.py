from abc import ABC, abstractmethod
from typing import Any, Literal

from textgrad.autograd.function import Module
from textgrad.optimizer.optimizer import Optimizer
from textgrad.variable import Variable

from method.tgext.algebra import sum
from method.losses import FactualityBasedLoss
from utils.factuality_score import evaluate_factuality


class Proposal(ABC):
    """Class representing a proposal distribution q."""

    @abstractmethod
    def propose(self, *args, **kwargs) -> None:
        """Sample parameters_new ~ q(parameters_new | parameters)."""
        pass

    @abstractmethod
    def logprobs(
        self, parameters_new: list[Variable], *args, **kwagrs
    ) -> tuple[str, str, list[list[float]]]:
        """Compute log probability q(parameters_new | parameters)."""
        pass

    @abstractmethod
    def propose_with_logprobs(self, *args, **kwargs) -> tuple[str, str, list[list[float]]]:
        """Sample parameters_new ~ q(parameters_new | parameters) and return q(parameters_new | parameters)."""
        pass


class TextGradProposal(Proposal):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        likelihood_loss: Module,
        prior_loss: Any | None = None,
        prior_losses_text: list[str] | None = None,
        sum_backward_mode: Literal["idempotent", "projection"] = "projection",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.likelihood_loss = likelihood_loss
        self.prior_losses = (
            [
                prior_loss(eval_system_prompt=prior_loss_text)
                for prior_loss_text in prior_losses_text
            ]
            if prior_loss and prior_losses_text
            else None
        )
        self.sum_backward_mode = sum_backward_mode

    def _propose_logprobs_helper(
        self, x: Variable, y: Variable, parameters_new: list[Variable] = None, **model_kwargs
    ) -> tuple[str, str, list[list[float]]]:
        self.optimizer.zero_grad()

        # Sample from the model
        res = self.model(x, **model_kwargs)
        # Get the final input/output pair from the model
        x_final_str = res["inputs_str"][-1]
        y_final = res["outputs"][-1]
        if type(self.likelihood_loss) == FactualityBasedLoss:
            loss_input = [x, y_final]
        else:
            loss_input = [x, y, y_final]
        print(f"loss_input: {loss_input}")
        if self.prior_losses:
            total_loss = sum(
                [
                    self.likelihood_loss(loss_input),
                    *[
                        prior_loss(parameter)
                        for prior_loss, parameter in zip(
                            self.prior_losses, self.optimizer.parameters
                        )
                    ],
                ],
                backward_mode=self.sum_backward_mode,
            )
        else:
            total_loss = self.likelihood_loss(loss_input)
        if type(self.likelihood_loss) == FactualityBasedLoss:
            factual_score_current = total_loss[1]
            total_loss = total_loss[0]
        total_loss.backward(engine=self.likelihood_loss.engine)

        parameters_new_logprobs = (
            self.optimizer.step()
            if parameters_new is None
            else self.optimizer.logprobs(parameters_new)
        )
        if type(self.likelihood_loss) == FactualityBasedLoss:
            return factual_score_current, x_final_str, y_final.value, parameters_new_logprobs
        return x_final_str, y_final.value, parameters_new_logprobs

    def propose_with_logprobs(
        self, x: Variable, y: Variable, **model_kwargs
    ) -> tuple[str, str, list[list[float]]]:
        return self._propose_logprobs_helper(x, y, **model_kwargs)

    def propose(self, x: Variable, y: Variable, **model_kwargs) -> None:
        self.propose_with_logprobs(x, y, **model_kwargs)

    def logprobs(
        self, parameters_new: list[Variable], x: Variable, y: Variable, **model_kwargs
    ) -> tuple[str, str, list[list[float]]]:
        return self._propose_logprobs_helper(x, y, parameters_new, **model_kwargs)
