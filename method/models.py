from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal

from textgrad.autograd.function import Module
from textgrad.engine import EngineLM
from textgrad.engine.base import CachedEngine
from textgrad.variable import Variable

from .tgext.algebra import sum
from .tgext.llm_call import FormattedLLMCall, LLMCall
from .tgext.utils import standardize_engine


@dataclass
class ParamData:
    name: str
    value: str
    role: str
    addendum: str | None = None
    addendum_role: str | None = None
    input_roles: list[str] | None = None


class BaseModel(Module):
    def __init__(
        self,
        params_data: list[ParamData],
        engine: EngineLM | str = None,
        use_engine_cache: bool = False,
    ) -> None:
        self._params_data = params_data
        # We may be zeroing the engine cache, so we deepcopy the engine
        # to avoid unexpected side effects
        engine = deepcopy(engine)
        self.engine = standardize_engine(engine)
        self.use_engine_cache = use_engine_cache

    @abstractmethod
    def parameters(self) -> list[Variable]:
        # The last parameter in this list must be the one used to produce the model's final output
        pass

    @abstractmethod
    def set_parameters(self, parameters_new: list[Variable]) -> None:
        pass

    @property
    def params_data(self) -> list[ParamData]:
        """Keep this property up-to-date with the underlying parameters"""
        return [
            replace(p_data, value=p.value)
            for p_data, p in zip(self._params_data, self.parameters())
        ]

    def __reduce__(self):
        """Need to define custom pickling because engine cannot be pickled"""
        return self.__class__, (self.params_data, self.engine.model_string, self.use_engine_cache)


class SimpleModel(BaseModel):
    def __init__(
        self,
        params_data: list[ParamData],
        engine: EngineLM | str = None,
        use_engine_cache: bool = False,
        sum_backward_mode: Literal["idempotent", "projection"] = "projection",
    ) -> None:
        super().__init__(params_data, engine, use_engine_cache)

        assert len(params_data) == 1, f"{self.__class__} requires a single parameter"
        assert (
            params_data[0].name == "system_prompt"
        ), f"{self.__class__}'s only parameter must be a system prompt"
        system_prompt_data = params_data[0]

        self.system_prompt = Variable(
            system_prompt_data.value,
            requires_grad=True,
            role_description=system_prompt_data.role,
        )

        self.addendum = (
            Variable(
                system_prompt_data.addendum,
                requires_grad=False,
                role_description=system_prompt_data.addendum_role,
            )
            if system_prompt_data.addendum and system_prompt_data.addendum_role
            else None
        )

        assert (
            system_prompt_data.input_roles is None
        ), f"{self.__class__} only takes a single input, so input_roles is not needed"

        self.llm_call = LLMCall(self.engine, self.system_prompt)
        self.sum_backward_mode = sum_backward_mode

    def parameters(self) -> list[Variable]:
        # The last parameter in this list must be the one used to produce the model's final output
        return [self.system_prompt]

    def set_parameters(self, parameters_new: list[Variable]) -> None:
        assert len(parameters_new) == 1, f"Parameter list is incorrect length for {self.__class__}"
        self.system_prompt = parameters_new[0]

    def forward(self, x: Variable, **engine_kwargs) -> dict[str, list[str | Variable]]:
        if not self.use_engine_cache and isinstance(self.engine, CachedEngine):
            self.engine.cache.clear()

        if self.addendum:
            # Append the fixed addendum (e.g., formatting instructions) to the system prompt
            system_prompt = sum(
                [self.system_prompt, self.addendum], backward_mode=self.sum_backward_mode
            )
        else:
            system_prompt = self.system_prompt
        self.llm_call.system_prompt = system_prompt
        y_pred = self.llm_call(x, **engine_kwargs)
        return {"inputs_str": [x.value], "outputs": [y_pred]}

    def __reduce__(self):
        """Need to define custom pickling because engine cannot be pickled"""
        return self.__class__, (
            self.params_data,
            self.engine.model_string,
            self.use_engine_cache,
            self.sum_backward_mode,
        )


class SelfRefineModel(BaseModel):
    def __init__(
        self,
        params_data: list[ParamData],
        engine: EngineLM | str = None,
        num_refine_steps: int = 1,
        use_engine_cache: bool = False,
        sum_backward_mode: Literal["idempotent", "projection"] = "projection",
    ) -> None:
        super().__init__(params_data, engine, use_engine_cache)

        assert len(params_data) == 3, f"{self.__class__} requires 3 parameters"
        assert (
            params_data[0].name == "generation_system_prompt"
        ), f"{self.__class__}'s first parameter must be the generation system prompt"
        generation_system_prompt_data = params_data[0]
        assert (
            params_data[1].name == "feedback_system_prompt"
        ), f"{self.__class__}'s second parameter must be the feedback system prompt"
        feedback_system_prompt_data = params_data[1]
        assert (
            params_data[2].name == "refine_system_prompt"
        ), f"{self.__class__}'s third parameter must be the refine system prompt"
        refine_system_prompt_data = params_data[2]

        self.generation_system_prompt = Variable(
            generation_system_prompt_data.value,
            requires_grad=True,
            role_description=generation_system_prompt_data.role,
        )
        self.feedback_system_prompt = Variable(
            feedback_system_prompt_data.value,
            requires_grad=True,
            role_description=feedback_system_prompt_data.role,
        )
        self.refine_system_prompt = Variable(
            refine_system_prompt_data.value,
            requires_grad=True,
            role_description=refine_system_prompt_data.role,
        )

        self.generation_addendum = (
            Variable(
                generation_system_prompt_data.addendum,
                requires_grad=False,
                role_description=generation_system_prompt_data.addendum_role,
            )
            if generation_system_prompt_data.addendum
            and generation_system_prompt_data.addendum_role
            else None
        )
        self.feedback_addendum = (
            Variable(
                feedback_system_prompt_data.addendum,
                requires_grad=False,
                role_description=feedback_system_prompt_data.addendum_role,
            )
            if feedback_system_prompt_data.addendum and feedback_system_prompt_data.addendum_role
            else None
        )
        self.refine_addendum = (
            Variable(
                refine_system_prompt_data.addendum,
                requires_grad=False,
                role_description=refine_system_prompt_data.addendum_role,
            )
            if refine_system_prompt_data.addendum and refine_system_prompt_data.addendum_role
            else None
        )

        assert (
            generation_system_prompt_data.input_roles is None
        ), f"{self.__class__}'s generation only takes a single input, so input_roles is not needed"
        self.generation_llm_call = LLMCall(self.engine, self.generation_system_prompt)

        assert (
            feedback_system_prompt_data.input_roles is not None
            and len(feedback_system_prompt_data.input_roles) == 2
        ), f"{self.__class__}'s feedback requires two input_roles"
        self.feedback_input_roles = feedback_system_prompt_data.input_roles
        feedback_format_string_items = []
        for role in self.feedback_input_roles:
            feedback_format_string_items.append(f"## {role.capitalize()}\n{{{role}}}")
        feedback_format_string = "\n\n".join(feedback_format_string_items)
        feedback_fields = {role: None for role in self.feedback_input_roles}
        self.feedback_formatted_llm_call = FormattedLLMCall(
            self.engine, feedback_format_string, feedback_fields, self.feedback_system_prompt
        )

        assert (
            refine_system_prompt_data.input_roles is not None
            and len(refine_system_prompt_data.input_roles) == 3
        ), f"{self.__class__}'s refinement requires three input_roles"
        assert (
            refine_system_prompt_data.input_roles[:2] == self.feedback_input_roles
        ), f"{self.__class__}'s refinement's first two input_roles must match feedback's"
        self.refine_input_roles = refine_system_prompt_data.input_roles
        refine_format_string_items = []
        for role in self.refine_input_roles:
            refine_format_string_items.append(f"## {role.capitalize()}\n{{{role}}}")
        refine_format_string = "\n\n".join(refine_format_string_items)
        refine_fields = {role: None for role in self.refine_input_roles}
        self.refine_formatted_llm_call = FormattedLLMCall(
            self.engine, refine_format_string, refine_fields, self.refine_system_prompt
        )

        self.num_refine_steps = num_refine_steps
        self.sum_backward_mode = sum_backward_mode

    def parameters(self) -> list[Variable]:
        # The last parameter in this list must be the one used to produce the model's final output
        return [
            self.generation_system_prompt,
            self.feedback_system_prompt,
            self.refine_system_prompt,
        ]

    def set_parameters(self, parameters_new: list[Variable]) -> None:
        assert len(parameters_new) == 3, f"Parameter list is incorrect length for {self.__class__}"
        self.generation_system_prompt = parameters_new[0]
        self.feedback_system_prompt = parameters_new[1]
        self.refine_system_prompt = parameters_new[2]

    def forward(self, x: Variable, **engine_kwargs) -> dict[str, list[str | Variable]]:
        if not self.use_engine_cache and isinstance(self.engine, CachedEngine):
            self.engine.cache.clear()

        # ------------------------------ Set System Prompts ------------------------------
        if self.generation_addendum:
            # Append the fixed addendum (e.g., formatting instructions) to the system prompt
            generation_system_prompt = sum(
                [self.generation_system_prompt, self.generation_addendum],
                backward_mode=self.sum_backward_mode,
            )
        else:
            generation_system_prompt = self.generation_system_prompt
        self.generation_llm_call.system_prompt = generation_system_prompt

        if self.feedback_addendum:
            # Append the fixed addendum (e.g., formatting instructions) to the system prompt
            feedback_system_prompt = sum(
                [self.feedback_system_prompt, self.feedback_addendum],
                backward_mode=self.sum_backward_mode,
            )
        else:
            feedback_system_prompt = self.feedback_system_prompt
        self.feedback_formatted_llm_call.system_prompt = feedback_system_prompt

        if self.refine_addendum:
            # Append the fixed addendum (e.g., formatting instructions) to the system prompt
            refine_system_prompt = sum(
                [self.refine_system_prompt, self.refine_addendum],
                backward_mode=self.sum_backward_mode,
            )
        else:
            refine_system_prompt = self.refine_system_prompt
        self.refine_formatted_llm_call.system_prompt = refine_system_prompt

        # ------------------------------ Initial Prediction ------------------------------
        y_pred = self.generation_llm_call(x, **engine_kwargs)

        inputs_str, outputs = [x.value], [y_pred]
        for _ in range(self.num_refine_steps):
            # ------------------------------ Feedback ------------------------------
            for role, var in zip(self.feedback_input_roles, [x, y_pred]):
                var.set_role_description(role)
            x_feedback = {role: var for role, var in zip(self.feedback_input_roles, [x, y_pred])}
            x_feedback_str, y_feedback = self.feedback_formatted_llm_call(
                x_feedback, **engine_kwargs
            )

            # ------------------------------ Refinement ------------------------------
            for role, var in zip(self.refine_input_roles, [x, y_pred, y_feedback]):
                var.set_role_description(role)
            x_refine = {
                role: var for role, var in zip(self.refine_input_roles, [x, y_pred, y_feedback])
            }
            x_refine_str, y_refine = self.refine_formatted_llm_call(x_refine, **engine_kwargs)

            inputs_str.extend([x_feedback_str, x_refine_str])
            outputs.extend([y_feedback, y_refine])

            y_pred = y_refine

        return {"inputs_str": inputs_str, "outputs": outputs}

    def __reduce__(self):
        """Need to define custom pickling because engine cannot be pickled"""
        return self.__class__, (
            self.params_data,
            self.engine.model_string,
            self.num_refine_steps,
            self.use_engine_cache,
            self.sum_backward_mode,
        )


class EnsembleModel(Module):
    def __init__(self, models: list[Module]) -> None:
        self.models = models

    def forward(self, x: Variable, **model_kwargs) -> list[dict[str, list[str | Variable]]]:
        return [model.forward(x, **model_kwargs) for model in self.models]
