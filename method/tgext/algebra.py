from typing import Literal

from textgrad import Variable, logger
from textgrad.autograd.function import BackwardContext, Function
from textgrad.engine import EngineLM


def sum(
    variables: list[Variable], backward_mode: Literal["idempotent", "projection"] = "idempotent"
) -> Variable:
    """
    The forward pass of sum is simply a concatenation of the values of the variables.
    The backward pass can be one of the following:
        - An idempotent operation (this is the default).
        - A gradient projection operation (the projection is an LLM call to project the gradients).

    :param variables: The list of variables to be summed (concatenated).
    :type variables: list[Variable]
    :param backward_mode: The backward pass mode to be used. Can be either "idempotent" or "projection".
    :type backward_mode: Literal["idempotent", "projection"]
    :return: A new variable representing the sum of the input variables.
    :rtype: Variable
    """
    return Sum()(variables, backward_mode=backward_mode)


class Sum(Function):
    """
    The forward pass of sum is simply a concatenation of the values of the variables.
    The backward pass can be one of the following:
        - An idempotent operation (this is the default).
        - A gradient projection operation (the projection is an LLM call to project the gradients).
    """

    def forward(
        self,
        variables: list[Variable],
        backward_mode: Literal["idempotent", "projection"] = "idempotent",
    ) -> Variable:
        """
        Performs the forward pass of the sum (concatenation) operation.

        :param variables: The list of variables to be summed (concatenated).
        :type variables: list[Variable]
        :param backward_mode: The backward pass mode to be used. Can be either "idempotent" or "projection".
        :type backward_mode: Literal["idempotent", "projection"]
        :return: A new variable representing the sum of the input variables.
        :rtype: Variable
        """
        concat_values = "\n".join([v.get_value() for v in variables])
        role_descriptions = [v.get_role_description() for v in variables]
        role_descriptions = " & ".join(role_descriptions)

        sum = Variable(
            value=concat_values,
            role_description=f"a combination of the following variables: {role_descriptions}",
            predecessors=variables,
            requires_grad=any([v.requires_grad for v in variables]),
        )

        sum.set_grad_fn(
            BackwardContext(backward_fn=self.backward, sum=sum, backward_mode=backward_mode)
        )

        return sum

    def backward(
        self,
        sum: Variable,
        backward_mode: Literal["idempotent", "projection"],
        backward_engine: EngineLM,
    ):
        """
        Performs the backward pass of the sum operation.
        The backward pass can be one of the following:
            - An idempotent operation (this is the default).
            - A gradient projection operation (the projection is an LLM call to project the gradients).

        :param sum: The variable representing the sum.
        :type sum: Variable
        :param backward_mode: The backward pass mode to be used. Can be either "idempotent" or "projection".
        :type backward_mode: Literal["idempotent", "projection"]
        :param backward_engine: The backward engine used for backpropagation.
        :type backward_engine: EngineLM
        """
        children_variables = sum.predecessors
        for sum_gradient in sum.gradients:
            for variable in children_variables:
                if not variable.requires_grad:
                    continue

                if sum_gradient.value == "":
                    variable_gradient_value = ""
                else:
                    if backward_mode == "idempotent":
                        variable_gradient_value = self._grad_idempotent(
                            variable, children_variables, sum_gradient.value, backward_engine
                        )
                        logger.info(
                            f"Idempotent backward",
                            extra={
                                "v_gradient_value": variable_gradient_value,
                                "summation_role": sum.get_role_description(),
                            },
                        )
                    elif backward_mode == "projection":
                        variable_gradient_value = self._grad_projection(
                            variable, children_variables, sum_gradient.value, backward_engine
                        )
                        logger.info(
                            f"Projection backward",
                            extra={
                                "v_gradient_value": variable_gradient_value,
                                "summation_role": sum.get_role_description(),
                            },
                        )
                var_gradient = Variable(
                    value=variable_gradient_value,
                    role_description=f"feedback to {variable.get_role_description()}",
                )
                variable.gradients.add(var_gradient)

                # Propagate the context of the gradients from the sum to the variable
                if sum_gradient in sum.gradients_context:
                    variable.gradients_context[var_gradient] = {
                        "context": sum.gradients_context[sum_gradient]["context"],
                        "response_desc": sum.gradients_context[sum_gradient]["response_desc"],
                        "variable_desc": variable.get_role_description(),
                    }

                if sum._reduce_meta != []:
                    var_gradient._reduce_meta.extend(sum._reduce_meta)
                    variable._reduce_meta.extend(sum._reduce_meta)

    def _grad_idempotent(
        self,
        variable: Variable,
        variables_all: set[Variable],
        sum_gradient: str,
        backward_engine: EngineLM,
    ) -> str:
        # This is the default behaviour in TextGrad for the sum operation
        return f"Here is the combined feedback we got for this specific {variable.get_role_description()} and other variables: {sum_gradient}."

    def _grad_projection(
        self,
        variable: Variable,
        variables_all: set[Variable],
        sum_gradient: str,
        backward_engine: EngineLM,
    ) -> str:
        variable_role = variable.get_role_description()
        variable_all_roles = [v.get_role_description() for v in variables_all]
        projection_prompt = construct_grad_projection_prompt(
            sum_gradient, variable_role, variable_all_roles
        )
        return backward_engine(projection_prompt, system_prompt=GRAD_PROJECTION_SYSTEM_PROMPT)


GRAD_PROJECTION_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "You will receive global feedback that applies to a combination of variables. "
    "Your only job is to extract from this global feedback the feedback that applies to one of the variables. "
    "You will only have access to the role description of each variable to help you determine which feedback applies to the variable of interest so use your best judgement. "
    "When giving a response, only extract the feedback that applies to the variable of interest and nothing else. Do not think or explain. Do not add additional feedback."
)


GRAD_PROJECTION_PROMPT_TEMPLATE = (
    "Role descriptions for all variables:\n"
    "{variable_all_roles}\n\n"
    "Global feedback that applies to the combination of these variables:\n"
    "<FEEDBACK>{feedback}</FEEDBACK>\n\n"
    "Extract only the feedback that applies to the variable with role <ROLE>{variable_role}</ROLE>"
)


def construct_grad_projection_prompt(
    sum_gradient: str, variable_role: str, variable_all_roles: list[str]
) -> str:
    """
    Construct a prompt that projects the sum gradient on to a child variable.
    """
    variable_all_roles = "\n".join([f"<ROLE>{role}</ROLE>" for role in variable_all_roles])
    return GRAD_PROJECTION_PROMPT_TEMPLATE.format(
        feedback=sum_gradient, variable_role=variable_role, variable_all_roles=variable_all_roles
    )
