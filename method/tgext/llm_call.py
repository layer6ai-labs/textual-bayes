"""
Here we create our own LLMCall and FormattedLLMCall classes that add one small feature not
available in TextGrad: the ability to pass keyword arguments into the engine call. This
allows us to use API features like stochasticity or logprobs.

Most of this code is copied from textgrad.autograd.llm_ops
"""

from textgrad import logger
from textgrad.autograd.function import BackwardContext
from textgrad.autograd.llm_ops import FormattedLLMCall as TextGradFormattedLLMCall
from textgrad.autograd.llm_ops import LLMCall as TextGradLLMCall
from textgrad.defaults import VARIABLE_OUTPUT_DEFAULT_ROLE
from textgrad.variable import Variable


class LLMCall(TextGradLLMCall):
    def forward(
        self,
        input_variable: Variable,
        response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE,
        **engine_kwargs,
    ) -> Variable:
        """
        The LLM call. This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

        :param input_variable: The input variable (aka prompt) to use for the LLM call.
        :type input_variable: Variable
        :param response_role_description: Role description for the LLM response, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :param engine_kwargs: Keyword arguments to be passed to the engine call within
        :type engine_kwargs: dict
        :return: response sampled from the LLM
        :rtype: Variable

        :example:
        >>> from textgrad import Variable, get_engine
        >>> from textgrad.autograd.llm_ops import LLMCall
        >>> engine = get_engine("gpt-3.5-turbo")
        >>> llm_call = LLMCall(engine)
        >>> prompt = Variable("What is the capital of France?", role_description="prompt to the LM")
        >>> response = llm_call(prompt, engine=engine)
        # This returns something like Variable(data=The capital of France is Paris., grads=)
        """
        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(
            input_variable.value, system_prompt=system_prompt_value, **engine_kwargs
        )

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=(
                [self.system_prompt, input_variable] if self.system_prompt else [input_variable]
            ),
            role_description=response_role_description,
        )

        logger.info(
            f"LLMCall function forward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {input_variable.value}\nResponse: {response_text}"
            },
        )

        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                prompt=input_variable.value,
                system_prompt=system_prompt_value,
            )
        )

        return response


class FormattedLLMCall(TextGradFormattedLLMCall):
    def forward(
        self,
        inputs: dict[str, Variable],
        response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE,
        **engine_kwargs,
    ) -> tuple[str, Variable]:
        """The LLM call with formatted strings.
        This function will call the LLM with the input and return the response, also register the grad_fn for backpropagation.

        :param inputs: Variables to use for the input. This should be a mapping of the fields to the variables.
        :type inputs: dict[str, Variable]
        :param response_role_description: Role description for the response variable, defaults to VARIABLE_OUTPUT_DEFAULT_ROLE
        :type response_role_description: str, optional
        :param engine_kwargs: Keyword arguments to be passed to the engine call within
        :type engine_kwargs: dict
        :return: Formatted input string and sampled response from the LLM
        :rtype: type[str, Variable]
        """
        # First ensure that all keys are present in the fields
        assert set(inputs.keys()) == set(
            self.fields.keys()
        ), f"Expected fields {self.fields.keys()} but got {inputs.keys()}"

        input_variables = list(inputs.values())

        # Now format the string
        formatted_input_string = self.format_string.format(
            **{k: inputs[k].value for k in inputs.keys()}
        )

        # TODO: Should we allow default roles? It will make things less performant.
        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(
            formatted_input_string, system_prompt=system_prompt_value, **engine_kwargs
        )

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=(
                [self.system_prompt, *input_variables] if self.system_prompt else [*input_variables]
            ),
            role_description=response_role_description,
        )

        logger.info(
            f"LLMCall function forward",
            extra={
                "text": f"System:{system_prompt_value}\nQuery: {formatted_input_string}\nResponse: {response_text}"
            },
        )

        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                prompt=formatted_input_string,
                system_prompt=system_prompt_value,
            )
        )

        return formatted_input_string, response
