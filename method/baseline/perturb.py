# Adapted from https://github.com/intuit-ai-research/SPUQ/tree/main

import logging
import random
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from method.baseline.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class PerturbedData:
    """Data structure holding a perturbed response, its logprobs, and the messages used."""

    response: str
    logprobs: Optional[List[float]]
    messages: List[Dict[str, str]]  # The full message history used for this specific generation


class ResponsePerturber(ABC):
    """Abstract base class for different response perturbation strategies."""

    @abstractmethod
    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        """
        Generates a list of 'n' perturbed responses for the given messages.

        Args:
            n: The number of perturbed responses to generate.
            engine: The generation engine instance.
            messages: The list of messages forming the conversation history/prompt.
        Returns:
            A list of PerturbedData objects, each containing a response,
            its logprobs, and the messages used.
        """
        pass


class SimpleSample(ResponsePerturber):
    """Generates multiple samples directly using the engine without perturbation."""

    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        responses, logprobs_list = engine.generate(
            messages=messages, n=n, logprobs=True, *args, **kwargs
        )

        results = []
        for i in range(n):
            response = responses[i]
            logger.info(f"Response {i}: {response}")
            logprobs = logprobs_list[i] if logprobs_list else None
            results.append(PerturbedData(response=response, logprobs=logprobs, messages=messages))
        return results


class TemperaturePerturber(ResponsePerturber):
    """Perturbs responses by sampling with varying temperatures."""

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0) -> None:
        assert 0 <= t_min <= t_max, "Temperature range must satisfy 0 <= t_min <= t_max"
        self.t_min = t_min
        self.t_max = t_max

    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        results = []

        def inner(i):
            temperature = self.t_min + np.random.random() * (self.t_max - self.t_min)
            generated_list, logprobs_list = engine.generate(
                messages=messages, n=1, temperature=temperature, logprobs=True, *args, **kwargs
            )
            response = generated_list[0]
            logprobs = logprobs_list[0] if logprobs_list else None
            return (
                i,
                temperature,
                PerturbedData(response=response, logprobs=logprobs, messages=messages),
            )

        with ThreadPoolExecutor(max_workers=min(32, n)) as exe:
            futures = [exe.submit(inner, i) for i in range(n)]
            for fut in as_completed(futures):
                i, temperature, perturbed = fut.result()
                logger.info(f"Response {i} with temp {temperature}: {perturbed.response}")
                results.append((i, perturbed))
        results = list(map(lambda x: x[1], sorted(results)))

        return results


class ParaphrasingPerturber(ResponsePerturber):
    """Perturbs the last user message by paraphrasing it before generation."""

    def __init__(self, ans_temp: float) -> None:
        """
        Initializes the paraphrasing perturber.
        """
        self.ans_temp = ans_temp
        # Updated command to ask for "a way" (singular) to paraphrase.
        self.cmd = (
            "Suggest a way to paraphrase the text in triple quotes above.\n"
            "If the original text is a question, please make sure that your answer is also a question.\n"
            "If the original text has answer options, please make sure your answer also has those options in the same order.\n"
            r"Answer should ONLY be the paraphrase and nothing else."
        )

    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        assert messages[-1]["role"] == "user", "Last element should be user input."
        original_content = messages[-1]["content"]

        paraphrase_user_prompt_content = f'"""\n{original_content}\n"""\n{self.cmd}'
        paraphrase_generation_messages = [
            {"role": "user", "content": paraphrase_user_prompt_content}
        ]

        all_paraphrases, _ = engine.generate(
            messages=paraphrase_generation_messages,
            n=n,
            logprobs=False,
            *args,
            **kwargs,
        )

        for i, x in enumerate(all_paraphrases):
            logger.info(f"Paraphrase {i}: {x}")
        results = []

        def inner(i):
            perturbed_messages = deepcopy(messages)
            perturbed_messages[-1]["content"] = all_paraphrases[i]
            generated_list, logprobs_list = engine.generate(
                messages=perturbed_messages,
                n=1,
                logprobs=True,
                temperature=self.ans_temp,
                *args,
                **kwargs,
            )
            response = generated_list[0]
            logprobs = logprobs_list[0] if logprobs_list else None
            return i, PerturbedData(
                response=response,
                logprobs=logprobs,
                messages=perturbed_messages,
            )

        with ThreadPoolExecutor(max_workers=min(32, n)) as exe:
            futures = [exe.submit(inner, i) for i in range(n)]
            for fut in as_completed(futures):
                i, perturbed = fut.result()
                logger.info(f"Responese {i}: {perturbed.response}")
                results.append((i, perturbed))
        results = list(map(lambda x: x[1], sorted(results)))

        return results


class SysMsgPerturber(ResponsePerturber):
    """Perturbs responses by prepending different system messages."""

    def __init__(self, ans_temp: float, system_messages: Optional[List[str]] = None) -> None:
        """
        Initializes the perturber with a list of system messages.

        Args:
            system_messages: A list of system message strings to choose from.
                             Defaults to a predefined list if None.
        """
        self.sys_msg = system_messages or [
            # Original from paper
            "you are a helpful assistant",
            "you are a question-answering assistant",
            "you are a nice assistant",
            # More to help with argmax decoding
            "You are an AI support tool.",
            "You are a friendly helper.",
            "You are here to assist users.",
            "You provide useful answers.",
            "You are a kind AI agent.",
            "You offer good information.",
            "You are a smart assistant.",
            "You help with many tasks.",
            "You are a reliable AI.",
            "You give clear responses.",
            "You are an able assistant.",
            "You try to be useful.",
            "You are a positive AI.",
            "You guide users well.",
            "You are an adept helper.",
            "You simplify complex things.",
            "You are a virtual guide.",
            "You aim to be accurate.",
        ]
        self.ans_temp = ans_temp

    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        results = []

        base_messages = messages
        shuffled = self.sys_msg[:]
        random.shuffle(shuffled)

        def inner(i):
            sys_msg = shuffled[i]  # if we repeat temp=0 will get the same result
            perturbed_messages = [{"role": "system", "content": sys_msg}] + base_messages
            generated_list, logprobs_list = engine.generate(
                messages=perturbed_messages,
                n=1,
                logprobs=True,
                temperature=self.ans_temp,
                *args,
                **kwargs,
            )
            response = generated_list[0]
            logger.info(f"Response {i} with sys prompt {sys_msg}: {response}")
            logprobs = logprobs_list[0] if logprobs_list else None
            return (
                i,
                sys_msg,
                PerturbedData(response=response, logprobs=logprobs, messages=perturbed_messages),
            )

        with ThreadPoolExecutor(max_workers=min(32, n)) as exe:
            futures = [exe.submit(inner, i) for i in range(n)]
            for fut in as_completed(futures):
                i, sys_msg, perturbed = fut.result()
                logger.info(f"Response {i} with sys prompt {sys_msg}: {perturbed.response}")
                results.append((i, perturbed))
        results = list(map(lambda x: x[1], sorted(results)))
        return results


class DummyTokenPerturber(ResponsePerturber):
    """Perturbs the last user message by adding dummy tokens/text."""

    def __init__(self, dummy_tokens: Optional[List[Dict[str, str]]] = None) -> None:
        """
        Initializes the perturber with a list of dummy tokens.

        Args:
            dummy_tokens: A list of dictionaries, each with 'text' and 'pos' ('before', 'after', 'both').
                          Defaults to a predefined list if None.
        """
        self.dummy_tokens = dummy_tokens or [
            {"text": "\n", "pos": "both"},
            {"text": "\t", "pos": "both"},
            {"text": " ", "pos": "both"},
            {"text": "...", "pos": "both"},
            {"text": " um, ", "pos": "before"},
            {"text": " uh, ", "pos": "before"},
            {"text": "?", "pos": "after"},
            {"text": "??", "pos": "after"},
            {"text": "\n\n", "pos": "both"},
            {"text": " um... ", "pos": "before"},
            {"text": " uh... ", "pos": "before"},
        ]

    def perturb_response(
        self,
        n: int,
        engine: Engine,
        messages: List[Dict[str, str]],
        *args: Any,
        **kwargs: Any,
    ) -> List[PerturbedData]:
        def inner(i):
            dummy = np.random.choice(self.dummy_tokens, 1)[0]
            perturbed_messages = deepcopy(messages)
            original_content = perturbed_messages[-1]["content"]
            text = dummy["text"]
            pos = dummy["pos"]
            if pos == "both":
                if np.random.random() > 0.5:
                    perturbed_messages[-1]["content"] = original_content + text
                else:
                    perturbed_messages[-1]["content"] = text + original_content
            elif pos == "before":
                perturbed_messages[-1]["content"] = text + original_content
            else:
                perturbed_messages[-1]["content"] = original_content + text

            generated_list, logprobs_list = engine.generate(
                messages=perturbed_messages, n=1, logprobs=True, *args, **kwargs
            )
            response = generated_list[0]
            logprobs = logprobs_list[0] if logprobs_list else None
            return (
                i,
                pos,
                text,
                PerturbedData(response=response, logprobs=logprobs, messages=perturbed_messages),
            )

        results = []

        with ThreadPoolExecutor(max_workers=min(32, n)) as exe:
            futures = [exe.submit(inner, i) for i in range(n)]
            for fut in as_completed(futures):
                i, pos, text, perturbed = fut.result()
                logger.info(f"Response {i} with  pos {pos} and dummy {text}: {perturbed.response}")
                results.append((i, perturbed))
        results = list(map(lambda x: x[1], sorted(results)))
        return results
