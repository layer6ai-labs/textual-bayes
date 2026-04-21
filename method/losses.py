import textgrad as tg
from textgrad.engine import EngineLM
from textgrad.variable import Variable
from utils.factuality_score import evaluate_factuality


class LikelihoodLoss(tg.loss.MultiFieldEvaluation):

    def __init__(
        self,
        evaluation_instruction: Variable | str,
        role_description: str,
        input_roles: list[str],
        engine: EngineLM | str = None,
        system_prompt: Variable = None,
    ):
        if isinstance(evaluation_instruction, str):
            evaluation_instruction = tg.Variable(
                evaluation_instruction,
                requires_grad=False,
                role_description=role_description,
            )

        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=input_roles,
            engine=engine,
            system_prompt=system_prompt,
        )


class FactualityBasedLoss(tg.loss.MultiFieldEvaluation):
    def __init__(
        self,
        evaluation_instruction: Variable | str,
        role_description: str,
        input_roles: list[str],
        engine: EngineLM | str = None,
        system_prompt: Variable = None,
        breakdown_prompt: str = None,
        factuality_model: str = "gpt-4o-mini",
    ):
        if isinstance(evaluation_instruction, str):
            evaluation_instruction = tg.Variable(
                evaluation_instruction,
                requires_grad=False,
                role_description=role_description,
            )

        super().__init__(
            evaluation_instruction=evaluation_instruction,
            role_descriptions=input_roles,
            engine=engine,
            system_prompt=system_prompt,
        )

        self.breakdown_prompt = breakdown_prompt
        self.factuality_model = factuality_model

    def forward(self, inputs, **kwargs) -> Variable:
        """Compute the factuality-based loss."""
        # Get subclaims from the predicted output
        x = inputs[0]
        y_pred = inputs[1]
        # Evaluate factuality
        factuality_score, annotation = evaluate_factuality(
            x.value, y_pred.value, self.factuality_model
        )
        # Create a new variable for the factuality score
        annotation = tg.Variable(
            annotation,
            requires_grad=False,
            role_description="factuality annotation",
        )
        loss = super().forward(
            inputs=[x, y_pred, annotation],
            **kwargs,
        )
        return (loss, factuality_score)
