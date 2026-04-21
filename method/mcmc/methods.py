import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from textgrad.engine import EngineLM
from textgrad.variable import Variable

from method.tgext.utils import standardize_engine
from utils.logprobs import extract_last_substring_logprobs

from .proposals import Proposal


class MCMCMethod(ABC):
    def __init__(self, proposal: Proposal) -> None:
        self.proposal = proposal
        self.history = []
        self.logger = logging.getLogger("MCMC")
        self.state = {}

    @property
    @abstractmethod
    def parameters_prev(self) -> list[Variable]:
        """Previous (stored) parameters"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> list[Variable]:
        """Current parameters"""
        pass

    @abstractmethod
    def store_state(self) -> None:
        """Store the required internal states in case of a sample rejection"""
        pass

    @abstractmethod
    def restore_state(self) -> None:
        """Restore the stored states after a sample rejection"""
        pass

    @abstractmethod
    def step(self, *args, **kwargs) -> None:
        """Compute a single step of the Markov Chain"""
        pass

    def sample_from_chain(
        self, dataloader, steps, burn_in, thinning, transform, **model_kwargs
    ) -> list[list[Variable]]:
        # Initialize the chain with the current parameters
        samples_params = [self.parameters]

        step = 0
        for example in dataloader:
            if step == steps:
                break

            if "unanswerable" in example[0] and example[0]["unanswerable"]:
                # in datasets where there are unanswerable questions, skip them for training
                continue

            x, y = transform(example)
            self.logger.info(f"x: {x.value}")
            self.logger.info(f"y: {y.value}")
            self.step(x, y, **model_kwargs)

            cur_params = self.parameters
            self.logger.info(f"Current parameters: {[p.value for p in cur_params]}")
            cur_accept_rate = sum([h["accept"] for h in self.history]) / len(self.history)
            self.logger.info(f"Current acceptance rate: {cur_accept_rate}")
            samples_params.append(cur_params)
            step += 1

        # Discard first `burn_in` samples and select every `thinning`th sample thereafter
        # E.g.: if steps=8, burn_in=4, thinning=2, then 8 steps will be taken after init,
        # making len(samples_params) == 9, and after the next line this will be
        # reduced to length 3 with indices 4, 6, and 8
        samples_params = samples_params[burn_in::thinning]

        return samples_params


class MetropolisHastings(MCMCMethod):
    def __init__(
        self,
        proposal: Proposal,
        likelihood_answer_regex: str,
        likelihood_param_addendum: str,
        likelihood_beta: int,
        prior_meta_prompts: list[str],
        engine: EngineLM | str | None = None,
    ):
        super().__init__(proposal)
        self.likelihood_answer_regex = likelihood_answer_regex
        self.likelihood_param_addendum = likelihood_param_addendum
        self.likelihood_beta = likelihood_beta
        self.prior_meta_prompts = prior_meta_prompts
        self.engine = standardize_engine(engine)

    @property
    def parameters_prev(self) -> list[Variable]:
        return self.state["optimizer"].parameters

    @property
    def parameters(self) -> list[Variable]:
        return self.proposal.optimizer.parameters

    def store_state(self) -> None:
        self.state["optimizer"] = deepcopy(self.proposal.optimizer)

    def restore_state(self) -> None:
        self.proposal.model.set_parameters(self.parameters_prev)
        self.proposal.optimizer = self.state["optimizer"]
        self.proposal.optimizer.parameters = self.proposal.model.parameters()

    def step(self, x: Variable, y: Variable, **model_kwargs) -> None:
        # Save all internal states in case the proposal is rejected
        self.store_state()
        parameters_prev = self.parameters_prev

        # ------------------------------ Proposal ------------------------------
        self.logger.info(f"Current parameters: {[p.value for p in parameters_prev]}")
        # This function call will update the internal value of the parameters
        x_final_prev, y_final_prev, parameters_new_logprobs = self.proposal.propose_with_logprobs(
            x, y, **model_kwargs
        )
        parameters_new = self.parameters
        self.logger.info(f"Proposed parameters: {[p.value for p in parameters_new]}")
        x_final_new, y_final_new, parameters_prev_logprobs = self.proposal.logprobs(
            parameters_prev, x, y, **model_kwargs
        )

        # Sum over all tokens of all parameters
        parameters_new_logprob = sum(sum(p_logprobs) for p_logprobs in parameters_new_logprobs)
        self.logger.info(f"log q(new | prev): {parameters_new_logprob}")
        # Sum over all tokens of all parameters
        parameters_prev_logprob = sum(sum(p_logprobs) for p_logprobs in parameters_prev_logprobs)
        self.logger.info(f"log q(prev | new): {parameters_prev_logprob}")

        # ------------------------------ Likelihood ------------------------------
        # If the predicted answer is in the right format, replace it with the true answer
        if re.search(self.likelihood_answer_regex, y_final_prev):
            y_prev = re.sub(self.likelihood_answer_regex, r"\g<1>" + str(y.value), y_final_prev)
            # Assume that the parameter used to generate the answer is the last one
            likelihood_parameter_prev = (
                parameters_prev[-1].value + "\n" + self.likelihood_param_addendum
            )
            likelihood_prev_logprobs = self.engine.logprobs(
                x_final_prev, y_prev, likelihood_parameter_prev
            )
            # Extract the logprobs of the tokens corresponding to the true answer
            likelihood_prev_logprobs = extract_last_substring_logprobs(
                string=y_prev,
                substring=y.value,
                tokens=likelihood_prev_logprobs.tokens,
                token_logprobs=likelihood_prev_logprobs.token_logprobs,
                logger=self.logger,
            )
            # Sum over all tokens
            likelihood_prev_logprob = sum(likelihood_prev_logprobs)
        else:  # We want to heavily penalize when answers are not in the right format
            likelihood_prev_logprob = -9999999.0
        self.logger.info(f"log p(y | x, params_prev) : {likelihood_prev_logprob}")
        # If the predicted answer is in the right format, replace it with the true answer
        if re.search(self.likelihood_answer_regex, y_final_new):
            y_new = re.sub(self.likelihood_answer_regex, r"\g<1>" + str(y.value), y_final_new)
            # Assume that the parameter used to generate the answer is the last one
            likelihood_parameter_new = (
                parameters_new[-1].value + "\n" + self.likelihood_param_addendum
            )
            likelihood_new_logprobs = self.engine.logprobs(
                x_final_new, y_new, likelihood_parameter_new
            )
            # Extract the logprobs of the tokens corresponding to the true answer
            likelihood_new_logprobs = extract_last_substring_logprobs(
                string=y_new,
                substring=y.value,
                tokens=likelihood_new_logprobs.tokens,
                token_logprobs=likelihood_new_logprobs.token_logprobs,
                logger=self.logger,
            )
            # Sum over all tokens
            likelihood_new_logprob = sum(likelihood_new_logprobs)
        else:  # We want to heavily penalize when answers are not in the right format
            likelihood_new_logprob = -9999999.0
        self.logger.info(f"log p(y | x, params_new) : {likelihood_new_logprob}")

        # ------------------------------ Prior ------------------------------
        prior_prev_logprobs = [
            self.engine.logprobs(prior_meta_prompt, parameter_prev.value).token_logprobs
            for prior_meta_prompt, parameter_prev in zip(self.prior_meta_prompts, parameters_prev)
        ]
        # Sum over all tokens of all parameters
        prior_prev_logprob = sum(sum(p_logprobs) for p_logprobs in prior_prev_logprobs)
        self.logger.info(f"log p(params_prev) : {prior_prev_logprob}")
        prior_new_logprobs = [
            self.engine.logprobs(prior_meta_prompt, parameter_new.value).token_logprobs
            for prior_meta_prompt, parameter_new in zip(self.prior_meta_prompts, parameters_new)
        ]
        # Sum over all tokens of all parameters
        prior_new_logprob = sum(sum(p_logprobs) for p_logprobs in prior_new_logprobs)
        self.logger.info(f"log p(params_new) : {prior_new_logprob}")

        # ------------------------------ Accept/Reject ------------------------------
        # Compute log acceptance ratio
        log_accept_ratio = (
            prior_new_logprob
            - prior_prev_logprob
            + self.likelihood_beta * likelihood_new_logprob
            - self.likelihood_beta * likelihood_prev_logprob
            + parameters_prev_logprob
            - parameters_new_logprob
        )
        accept = np.log(np.random.uniform()) <= log_accept_ratio
        if not accept:
            # Restore all internal states that may have changed as part of the proposal
            self.restore_state()

        self.logger.info(f"Log acceptance ratio: {log_accept_ratio}")
        self.logger.info(f"Proposal {'accepted' if accept else 'rejected'}")
        self.history.append(
            {
                "proposed": [p.value for p in parameters_new],
                "accept": accept,
                "previous": [p.value for p in parameters_prev],
            }
        )


class MetropolisHastingsFactuality(MCMCMethod):
    def __init__(
        self,
        proposal: Proposal,
        likelihood_answer_regex: str,
        likelihood_param_addendum: str,
        likelihood_beta: int,
        prior_meta_prompts: list[str],
        engine: EngineLM | str | None = None,
    ):
        super().__init__(proposal)
        self.likelihood_answer_regex = likelihood_answer_regex
        self.likelihood_param_addendum = likelihood_param_addendum
        self.likelihood_beta = likelihood_beta
        self.prior_meta_prompts = prior_meta_prompts
        self.engine = standardize_engine(engine)

    @property
    def parameters_prev(self) -> list[Variable]:
        return self.state["optimizer"].parameters

    @property
    def parameters(self) -> list[Variable]:
        return self.proposal.optimizer.parameters

    def store_state(self) -> None:
        self.state["optimizer"] = deepcopy(self.proposal.optimizer)

    def restore_state(self) -> None:
        self.proposal.model.set_parameters(self.parameters_prev)
        self.proposal.optimizer = self.state["optimizer"]
        self.proposal.optimizer.parameters = self.proposal.model.parameters()

    def step(self, x: Variable, y: Variable, **model_kwargs) -> None:
        # Save all internal states in case the proposal is rejected
        self.store_state()
        parameters_prev = self.parameters_prev

        # ------------------------------ Proposal ------------------------------
        self.logger.info(f"Current parameters: {[p.value for p in parameters_prev]}")
        # This function call will update the internal value of the parameters
        factual_score_prev, x_final_prev, y_final_prev, parameters_new_logprobs = (
            self.proposal.propose_with_logprobs(x, y, **model_kwargs)
        )
        if factual_score_prev == 0:
            factual_score_prev = 1e-200
        self.logger.info(
            f"Current factuality score: {factual_score_prev}, log: {np.log(factual_score_prev)}"
        )
        parameters_new = self.parameters
        self.logger.info(f"Proposed parameters: {[p.value for p in parameters_new]}")
        factual_score_new, x_final_new, y_final_new, parameters_prev_logprobs = (
            self.proposal.logprobs(parameters_prev, x, y, **model_kwargs)
        )
        if factual_score_new == 0:
            factual_score_new = 1e-200
        self.logger.info(
            f"New factuality score: {factual_score_new}, log: {np.log(factual_score_new)}"
        )

        # Sum over all tokens of all parameters
        parameters_new_logprob = sum(sum(p_logprobs) for p_logprobs in parameters_new_logprobs)
        self.logger.info(f"log q(new | prev): {parameters_new_logprob}")
        # Sum over all tokens of all parameters
        parameters_prev_logprob = sum(sum(p_logprobs) for p_logprobs in parameters_prev_logprobs)
        self.logger.info(f"log q(prev | new): {parameters_prev_logprob}")

        # ------------------------------ Prior ------------------------------
        prior_prev_logprobs = [
            self.engine.logprobs(prior_meta_prompt, parameter_prev.value).token_logprobs
            for prior_meta_prompt, parameter_prev in zip(self.prior_meta_prompts, parameters_prev)
        ]
        # Sum over all tokens of all parameters
        prior_prev_logprob = sum(sum(p_logprobs) for p_logprobs in prior_prev_logprobs)
        self.logger.info(f"log p(params_prev) : {prior_prev_logprob}")
        prior_new_logprobs = [
            self.engine.logprobs(prior_meta_prompt, parameter_new.value).token_logprobs
            for prior_meta_prompt, parameter_new in zip(self.prior_meta_prompts, parameters_new)
        ]
        # Sum over all tokens of all parameters
        prior_new_logprob = sum(sum(p_logprobs) for p_logprobs in prior_new_logprobs)
        self.logger.info(f"log p(params_new) : {prior_new_logprob}")

        # ------------------------------ Accept/Reject ------------------------------
        # Compute log acceptance ratio
        log_accept_ratio = (
            prior_new_logprob
            - prior_prev_logprob
            + self.likelihood_beta * np.log(factual_score_new)
            - self.likelihood_beta * np.log(factual_score_prev)
            + parameters_prev_logprob
            - parameters_new_logprob
        )
        accept = np.log(np.random.uniform()) <= log_accept_ratio
        if not accept:
            # Restore all internal states that may have changed as part of the proposal
            self.restore_state()

        self.logger.info(f"Log acceptance ratio: {log_accept_ratio}")
        self.logger.info(f"Proposal {'accepted' if accept else 'rejected'}")
        self.history.append(
            {
                "proposed": [p.value for p in parameters_new],
                "accept": accept,
                "previous": [p.value for p in parameters_prev],
            }
        )


class LangevinDynamics(MCMCMethod):
    @property
    def parameters_prev(self) -> list[Variable]:
        return self.state["parameters"]

    @property
    def parameters(self) -> list[Variable]:
        return self.proposal.optimizer.parameters

    def store_state(self) -> None:
        self.state["parameters"] = deepcopy(self.parameters)

    def restore_state(self) -> None:
        pass

    def step(self, x: Variable, y: Variable, **model_kwargs) -> None:
        # Here the internal states consist of the parameters for bookkeeping only
        self.store_state()
        parameters_prev = self.parameters_prev
        # This function call will update the internal value of the parameters
        self.proposal.propose(x, y, **model_kwargs)
        parameters_new = self.parameters
        self.history.append(
            {
                "proposed": [p.value for p in parameters_new],
                "accept": True,
                "previous": [p.value for p in parameters_prev],
            }
        )
