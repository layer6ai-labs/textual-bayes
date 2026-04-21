# Inspired by https://github.com/intuit-ai-research/SPUQ

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
from evaluate import load as hf_load
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from method.baseline.engine import Engine
from method.baseline.perturb import PerturbedData
from utils.semantic_uq import SemanticClustering, SemanticUQ

logger = logging.getLogger(__name__)


class ResponseUQ(ABC):
    """Abstract base class for response uncertainty quantification methods."""

    @abstractmethod
    def quantify_uncertainty(
        self,
        perturbed_responses: List[PerturbedData],
        **kwargs: Any,
    ) -> dict:
        """
        Calculates an uncertainty score based on a list of perturbed responses.

        Args:
            perturbed_responses: The list of generated responses and associated data.
        Returns:
            A single float representing the calculated uncertainty score (often confidence).
        """
        pass


class FrequencyUQ(ResponseUQ):
    """Simple class for aggregation based on frequency of answer, parsed by regex.
    Use with multi choice and/or simple datasets where the answer can be extracted via regex.
    """

    def __init__(self, answer_regex: str):
        self.answer_regex = re.compile(answer_regex)

    def quantify_uncertainty(
        self,
        perturbed_responses: List[Any],
        **kwargs: Any,
    ) -> dict:
        """
        Calculates uncertainty based on the frequency of the most common parsed answer.
        """
        parsed_answers = []
        for item in perturbed_responses:
            match = self.answer_regex.search(item.response)
            if match and match.groups():
                parsed_answers.append(match.group(1))
            else:
                parsed_answers.append(None)

        valid_answers = [ans for ans in parsed_answers if ans is not None]

        if not valid_answers:
            return {
                "uncertainty": 1,
                "confidence": 0,
            }

        answer_counts = Counter(valid_answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        confidence = most_common_count / len(perturbed_responses)
        return {
            "uncertainty": 1 - confidence,
            "confidence": confidence,
        }


class WeightedUQ(ResponseUQ):
    """Base class for UQ methods supporting input similarity weighting."""

    def __init__(self, weighted: bool = True):
        self.weighted = weighted
        self._weight_scorer = (
            rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if weighted else None
        )

    def _calculate_weight(
        self, inp0_turns: List[Dict[str, str]], inp_turns: List[Dict[str, str]]
    ) -> float:
        """Calculates weight based on input similarity (RougeL)."""
        if not self.weighted:
            return 1.0

        inp0 = "\n".join([turn["content"] for turn in inp0_turns])
        inp = "\n".join([turn["content"] for turn in inp_turns])
        score = self._weight_scorer.score(inp0, inp)["rougeL"].fmeasure
        return score


class InterUQ(WeightedUQ):
    """Base class for UQ methods calculating pairwise similarity between responses."""

    @abstractmethod
    def _calculate_similarity(self, a: str, b: str) -> float:
        """Subclasses must implement their specific similarity calculation."""
        pass

    def quantify_uncertainty(
        self,
        perturbed_responses: List[PerturbedData],
        **kwargs: Any,
    ) -> dict:
        """Calculates average pairwise similarity against the first response."""
        assert len(perturbed_responses) > 1, "quantify_uncertainty called with < 2 responses"

        sum_conf = 0.0
        sum_wt = 0.0

        inp0_turns = perturbed_responses[0].messages
        out0 = perturbed_responses[0].response

        for i in range(1, len(perturbed_responses)):
            inp_turns = perturbed_responses[i].messages
            out = perturbed_responses[i].response
            wt = self._calculate_weight(inp0_turns, inp_turns)
            if wt == 0:
                continue  # Avoid calc when wt is zero
            conf = self._calculate_similarity(a=out0, b=out)
            sum_conf += conf * wt
            sum_wt += wt
        confidence = sum_conf / sum_wt
        return {
            "uncertainty": 1 - confidence,
            "confidence": confidence,
        }


class RougeLUQ(InterUQ):
    """Calculates uncertainty based on RougeL similarity between perturbed responses."""

    def __init__(self, weighted: bool = True):
        super().__init__(weighted=weighted)
        self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self._weight_scorer = self._rouge_scorer if weighted else None

    def _calculate_similarity(self, a: str, b: str) -> float:
        return self._rouge_scorer.score(a, b)["rougeL"].fmeasure


class SbertUQ(InterUQ):
    """Calculates uncertainty based on SBERT cosine similarity between perturbed responses."""

    def __init__(self, weighted: bool = True):
        super().__init__(weighted=weighted)
        self._embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    def _calculate_similarity(self, a: str, b: str) -> float:
        embs = self._embedder.encode([a, b])
        norm = np.sqrt(np.sum(embs * embs, axis=-1))
        norm_embs = embs / norm.reshape(-1, 1)
        cos_sim = np.sum(norm_embs[0] * norm_embs[1], axis=-1)
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        similarity = (cos_sim + 1) / 2  # Transform [-1,1] to [0,1] for similarity.
        return similarity


class BertUQ(InterUQ):
    def __init__(self, weighted: bool = True, lang: str = "en"):
        super().__init__(weighted=weighted)
        self.lang = lang
        self._bertscore = hf_load("bertscore")

    def _calculate_similarity(self, a: str, b: str) -> float:
        results = self._bertscore.compute(
            predictions=[b], references=[a], lang=self.lang, verbose=False
        )
        return results["f1"][0]


class IntraUQ(WeightedUQ):
    """Base class for UQ methods asking an LLM about its confidence on a full conversation."""

    DEFAULT_CONF = 0.5

    def __init__(self, engine: Engine, weighted: bool = True):
        super().__init__(weighted=weighted)
        self.engine = engine

    def _prepare_and_call_engine(
        self,
        setup_prompt: str,
        request_prompt: str,
        messages: List[Dict[str, str]],
    ) -> str:
        """Asserts message structure: optional system, then user/assistant alternating, ends with assistant.
        Validates message structure, formats prompt, calls engine, returns verbalized response."""

        start_index = 0
        if messages[0]["role"] == "system":
            start_index = 1
        expected_role = "user"
        for i in range(start_index, len(messages)):
            actual_role = messages[i]["role"]
            if actual_role != expected_role:
                raise AssertionError(
                    f"Message at index {i} has role '{actual_role}', expected '{expected_role}'."
                )
            expected_role = "assistant" if expected_role == "user" else "user"

        if messages[-1]["role"] != "assistant":
            raise AssertionError("The last message must be from the 'assistant'.")

        prompt_turns = (
            [{"role": "system", "content": setup_prompt}]
            + messages[start_index:]
            + [{"role": "user", "content": request_prompt}]
        )

        verbalized_list, *_ = self.engine.generate(prompt_turns, temperature=0.0)
        return verbalized_list[0]

    @abstractmethod
    def _get_single_confidence(self, messages: List[Dict[str, str]]) -> float:
        """Subclasses must implement parsing of LLM response, using their specific templates."""
        pass

    def quantify_uncertainty(
        self,
        perturbed_responses: List[PerturbedData],
        **kwargs: Any,
    ) -> float:
        """Calculates average self-reported confidence across perturbed responses."""
        sum_conf = 0.0
        sum_wt = 0.0

        inp0_turns = perturbed_responses[0].messages if self.weighted else []

        def inner(i: int) -> Tuple[float, float]:
            item = perturbed_responses[i]
            inp_turns = item.messages
            wt = self._calculate_weight(inp0_turns, inp_turns)
            if wt == 0:
                return 0, 0  # Avoid calculation when weight is zero.
            conf = self._get_single_confidence(
                inp_turns + [{"role": "assistant", "content": item.response}]
            )
            return conf, wt

        n = len(perturbed_responses)
        with ThreadPoolExecutor(max_workers=min(32, n)) as exe:
            futures = [exe.submit(inner, i) for i in range(n)]
            for fut in as_completed(futures):
                conf, wt = fut.result()
                sum_conf += conf * wt
                sum_wt += wt
        confidence = sum_conf / sum_wt
        return {
            "uncertainty": 1 - confidence,
            "confidence": confidence,
        }


class Ask4ConfWordUQ(IntraUQ):
    """Calculates uncertainty by asking the LLM for its confidence ('Low', 'Medium', 'High') on a full conversation."""

    SETUP_PROMPT = "You previously answered a question. Your task now is to identify how certain you are of your answer using a word descriptor."
    REQUEST_PROMPT = "Provide your confidence in the answer using one word: Low, Medium, or High. Give ONLY the word, no other words or explanation.\nFor example:\nConfidence: Medium"

    WORD_MAP = {"low": 0.25, "medium": 0.5, "high": 0.75}

    def _get_single_confidence(self, messages: List[Dict[str, str]]) -> float:
        """Asks the LLM for confidence based on full history and parses the word response."""
        verbalized = self._prepare_and_call_engine(
            setup_prompt=self.SETUP_PROMPT,
            request_prompt=self.REQUEST_PROMPT,
            messages=messages,
        )
        verbalized_lower = verbalized.lower()

        for word, value in self.WORD_MAP.items():
            if re.search(r"\b" + word + r"\b", verbalized_lower):
                logger.info(f"Got confidence {word} from: '{verbalized}'")
                return value
        logger.debug(f"Could not get word confidence from LLM response: '{verbalized}'")
        return self.DEFAULT_CONF


class Ask4ConfNumUQ(IntraUQ):
    """Calculates uncertainty by asking the LLM for its confidence (0.0-1.0) on a full conversation."""

    SETUP_PORMPT = "You previously answered a question. Your task now is to identify how certain you are of your answer."
    REQUEST_PROMPT = "Provide the probability that your answer is correct. Give ONLY the probability, no other words or explanation.\nFor example:\nProbability: 0.85"

    FLOAT_REGEX = re.compile(r"([0-1](?:\.\d+)?)\b")

    def _get_single_confidence(self, messages: List[Dict[str, str]]) -> float:
        """Asks the LLM for confidence based on full history and parses the float response."""
        verbalized = self._prepare_and_call_engine(
            setup_prompt=self.SETUP_PORMPT,
            request_prompt=self.REQUEST_PROMPT,
            messages=messages,
        )
        matches = self.FLOAT_REGEX.findall(verbalized)
        if matches:
            conf = float(matches[0])
            logger.info(f"Got confidence {conf} from: '{verbalized}'")
            return max(0.0, min(1.0, conf))
        logger.info(f"Could not parse confidence number from LLM response: '{verbalized}'")
        return self.DEFAULT_CONF

    def _calculate_similarity(self, a: str, b: str) -> float:
        return self._rouge_scorer.score(a, b)["rougeL"].fmeasure


class LaplacianEigenvalueUQ(ResponseUQ):
    """Computes Laplacian Eigenvalue uncertainty U_EigV."""

    def __init__(
        self,
        llm_engine,
        entailment_model="deberta",
        strict_entailment=False,
    ):
        self.semantic_uq = SemanticUQ(
            entailment_model=entailment_model,
            strict_entailment=strict_entailment,
            llm_engine=llm_engine,
        )

    def quantify_uncertainty(
        self, perturbed_responses: List[PerturbedData], **kwargs: Any
    ) -> float:
        """Warning: The number if not bounded between [0,1]"""
        uncertainty = self.semantic_uq.get_laplacian_uncertainties(
            [x.response for x in perturbed_responses]
        )[0]

        return {"uncertainty": uncertainty}


class LaplacianDegreeUQ(ResponseUQ):
    """Computes Laplacian Degree uncertainty U_Deg."""

    def __init__(
        self,
        llm_engine,
        entailment_model="deberta",
        strict_entailment=False,
    ):
        self.semantic_uq = SemanticUQ(
            entailment_model=entailment_model,
            strict_entailment=strict_entailment,
            llm_engine=llm_engine,
        )

    def quantify_uncertainty(
        self, perturbed_responses: List[PerturbedData], **kwargs: Any
    ) -> float:
        """Warning: The number if not bounded between [0,1]"""
        uncertainty = self.semantic_uq.get_laplacian_uncertainties(
            [x.response for x in perturbed_responses]
        )[1]

        return {"uncertainty": uncertainty}


class SemanticVNEUq(ResponseUQ):
    """Computes Semantic Von Neumann Entropy using Heat Kernel."""

    def __init__(
        self,
        llm_engine,
        entailment_model="deberta",
        strict_entailment=False,
    ):
        self.semantic_uq = SemanticUQ(
            entailment_model=entailment_model,
            strict_entailment=strict_entailment,
            llm_engine=llm_engine,
        )

    def quantify_uncertainty(
        self, perturbed_responses: List[PerturbedData], **kwargs: Any
    ) -> float:
        """Warning: The number if not bounded between [0,1]"""
        uncertainty = self.semantic_uq.get_semantic_von_neumann_entropy(
            [x.response for x in perturbed_responses]
        )

        return {"uncertainty": uncertainty}


class SemanticEntropyUQ(ResponseUQ):
    def __init__(
        self,
        llm_engine,
        entailment_model="deberta",
        strict_entailment=False,
    ):
        self.semantic_uq = SemanticUQ(
            entailment_model=entailment_model,
            strict_entailment=strict_entailment,
            llm_engine=llm_engine,
        )

    def quantify_uncertainty(
        self, perturbed_responses: List[PerturbedData], use_logprobs=False, **kwargs: Any
    ) -> float:
        """Warning: The number if not bounded between [0,1]"""
        if use_logprobs:
            uncertainty = self.semantic_uq.get_semantic_entropy(
                [x.response for x in perturbed_responses],
                perturbed_responses[0].messages[1]["content"],
                return_num_sets=False,
                log_probs=[x.logprobs for x in perturbed_responses],
            )
        else:
            uncertainty = self.semantic_uq.get_semantic_entropy(
                [x.response for x in perturbed_responses],
                perturbed_responses[0].messages[1]["content"],
                return_num_sets=False,
            )
        return {"uncertainty": uncertainty}


class SemanticFrequencyUQ(ResponseUQ):
    def __init__(
        self,
        llm_engine,
        answer_regex,
        entailment_model="deberta",
        strict_entailment=False,
    ):
        self.sc = SemanticClustering(
            entailment_model=entailment_model,
            strict_entailment=strict_entailment,
            llm_engine=llm_engine,
            cuda=True,
        )
        self.answer_regex = re.compile(answer_regex)

    def quantify_uncertainty(
        self, perturbed_responses: List[PerturbedData], question: str, y_true: str, **kwargs: Any
    ) -> dict:
        """Warning: The number is not bounded between [0,1]"""

        answers_parsed = []
        for item in perturbed_responses:
            match = self.answer_regex.search(item.response)
            answers_parsed.append(match.group(1) if match else "<Not found>")

        semantic_ids = self.sc.get_semantic_ids(
            responses=answers_parsed,
            prompt=question,
            method="llm",
        )

        counter = Counter(semantic_ids)
        answers_len = len(answers_parsed)
        ys = {answer: count / answers_len for answer, count in counter.items()}
        majority_cluster = max(ys, key=ys.get)
        y_pred = answers_parsed[semantic_ids.index(majority_cluster)]
        y_conf = max(ys.values())
        equivalence = self.sc.get_correctness_by_llm(y_true, y_pred, question)
        logger.info("Answers from model\n" + "\n".join(answers_parsed))
        logger.info(f"Clustering results: {semantic_ids}")
        logger.info(f"Most repeated answer and groundtruth equivalence: {equivalence}")

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_conf": y_conf,
            "y_pred_cluster": majority_cluster,
            "abst_pred": -y_conf,
            "equivalence": equivalence,
        }
