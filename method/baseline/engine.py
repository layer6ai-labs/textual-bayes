import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


class Engine(ABC):
    """Abstract Base Class for language model engines."""

    @abstractmethod
    def generate(
        self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        """
        Generates text based on input messages and optionally returns log probabilities.
        Args:
            messages: A list of message dictionaries.
            logprobs: A boolean value that indicates if engine should return logprobs.
        Returns:
            A tuple containing:
            - A list of generated text strings (one for each choice).
            - An optional list containing log probability information for each choice.
        """
        pass


class OpenAIEngine(Engine):
    def __init__(self, model_name: str):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model_name

    def generate(
        self, messages: List[Dict[str, str]], logprobs: bool = False, **kwargs
    ) -> Tuple[List[str], Optional[List[List[float]]]]:
        ret = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            logprobs=logprobs,
            **kwargs,
        )

        texts = [choice.message.content for choice in ret.choices]
        logprobs_list = None
        if logprobs:
            logprobs_list = []
            for choice in ret.choices:
                l = [a.logprob for a in choice.logprobs.content]
                logprobs_list.append(l)
        return texts, logprobs_list
