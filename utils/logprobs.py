import copy
from logging import Logger
from typing import Dict, List

import numpy as np
from scipy.special import logsumexp
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def compute_prompt_key_logprobs_concat(
    prompt_template: str,
    prompt_dicts: List[Dict[str, str]],
    key: str,
    llm: LLM,
    sampling_params: SamplingParams,
) -> List[float]:
    """
    Computes the log probs for the part of a prompt associated with a given key.
    Assumes the key is located at the end of the prompt template.

    This method first tokenizes the prompt without the key and the key
    separately and then concatenates them so that the tokens where the
    key begins can be readily determinted based on length to extract logprobs.
    """

    tokenizer = llm.get_tokenizer()

    prompt_no_key_dicts = copy.deepcopy(prompt_dicts)
    for p in prompt_no_key_dicts:
        p[key] = ""
    prompts_no_key = [prompt_template.format(**p) for p in prompt_no_key_dicts]
    prompt_no_key_token_ids = tokenizer(prompts_no_key).input_ids

    prompts_key = [p[key] for p in prompt_dicts]
    prompt_key_token_ids = tokenizer(prompts_key).input_ids

    out = llm.generate(
        [
            TokensPrompt(prompt_token_ids=prompt_no_key_token_ids[i] + prompt_key_token_ids[i])
            for i in range(len(prompts_key))
        ],
        sampling_params,
        use_tqdm=False,
    )

    logprobs = []
    for i in range(len(prompts_key)):
        key_logprobs_dict = out[i].prompt_logprobs[-len(prompt_key_token_ids[i]) :]
        key_logprobs = [
            l.logprob for logprob_dict in key_logprobs_dict for id, l in logprob_dict.items()
        ]
        key_logprob = sum(key_logprobs)
        logprobs.append(key_logprob)

    return logprobs


def compute_prompt_key_logprobs_search(
    prompt_template: str,
    prompt_dicts: List[Dict[str, str]],
    key: str,
    llm: LLM,
    sampling_params: SamplingParams,
) -> List[float]:
    """
    Computes the log probs for the part of a prompt with a given key.
    Assumes the key is located at the end of the prompt template.

    This method first tokenizes the entire prompt and then searches for
    the first token in the tokenized prompt where the key begins to extract
    logprobs.
    """

    tokenizer = llm.get_tokenizer()

    prompts = [prompt_template.format(**p) for p in prompt_dicts]
    prompt_token_ids = tokenizer(prompts).input_ids

    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids[i]) for i in range(len(prompts))],
        sampling_params,
        use_tqdm=False,
    )

    prompt_no_key_dicts = copy.deepcopy(prompt_dicts)
    for p in prompt_no_key_dicts:
        p[key] = ""
    prompts_no_key = [prompt_template.format(**p) for p in prompt_no_key_dicts]
    prompt_no_key_token_ids = tokenizer(prompts_no_key).input_ids

    logprobs = []
    for i in range(len(prompts)):
        for prompt_key_start_idx in range(len(prompt_no_key_token_ids[i]) + 1, -1, -1):
            prompt_key_token_ids = prompt_token_ids[i][prompt_key_start_idx:]
            prompt_key = tokenizer.decode(prompt_key_token_ids)
            if prompt_key.endswith(prompt_dicts[i][key]):
                break
        if prompt_key_start_idx == 0:
            raise Exception(
                f"Something went wrong when trying to find the key {key} start index in the prompt {prompts[i]}"
            )

        key_logprobs_dict = out[i].prompt_logprobs[prompt_key_start_idx:]
        key_logprobs = [
            l.logprob for logprob_dict in key_logprobs_dict for id, l in logprob_dict.items()
        ]
        key_logprob = sum(key_logprobs)
        logprobs.append(key_logprob)

    return logprobs


def compute_prompt_key_logprobs(
    prompt_template: str,
    prompt_dicts: List[Dict[str, str]],
    key: str,
    llm: LLM,
    sampling_params: SamplingParams,
    method: str = "concat",
) -> List[float]:
    """
    Computes the log probs for the part of a prompt with a given key.
    Assumes the key is located at the end of the prompt template.

    This method will call the particular method for extracting logprobs
    from the prompt depending on the specified method.
    """
    sampling_params = copy.deepcopy(sampling_params)
    sampling_params.max_tokens = (
        1  # do not need to generate to get prompt logprobs, so set to smallest value
    )
    sampling_params.prompt_logprobs = 0  # to get the prompt logprobs of only the input tokens

    if method == "concat":
        return compute_prompt_key_logprobs_concat(
            prompt_template, prompt_dicts, key, llm, sampling_params
        )
    elif method == "search":
        return compute_prompt_key_logprobs_search(
            prompt_template, prompt_dicts, key, llm, sampling_params
        )
    else:
        raise NotImplementedError()


def compute_posterior_probs(
    likelihood_logprobs: np.ndarray, prior_logprobs: np.ndarray
) -> np.ndarray:
    """
    Computes the posterior probs from the logprobs
    Inputs:
        likelihood_logprobs: shape(batch_size, num_multi_choices)
        prior_logprobs: shape(batch_size, num_multi_choices)
    """
    # Compute posterior in logspace first for numerical stability
    posterior_logprobs_unnormalized = likelihood_logprobs + prior_logprobs
    posterior_logpartition = logsumexp(posterior_logprobs_unnormalized, axis=-1, keepdims=True)
    posterior_logprobs = posterior_logprobs_unnormalized - posterior_logpartition
    return np.exp(posterior_logprobs)


def extract_last_substring_logprobs(
    string: str,
    substring: str,
    tokens: list[str],
    token_logprobs: list[float],
    logger: Logger | None = None,
):
    """
    Extracts the log probabilities of tokens that overlap with the last occurrence of a specified substring in a string.

    Args:
        string (str): The original string composed of tokens.
        substring (str): The substring to find within the string.
        tokens (list[str]): A list of tokens that make up the string.
        token_logprobs (list[float]): A list of log probabilities corresponding to each token.

    Returns:
        list[float]: A list of log probabilities for tokens that overlap with the last occurrence of the substring.

    Raises:
        ValueError: If the substring is not found in the string.
    """
    # Find the span of the substring
    start_idx = string.rfind(substring)
    if start_idx == -1:
        raise ValueError(f"substring {substring} not found in string {string}")
    end_idx = start_idx + len(substring)

    extracted_tokens, extracted_logprobs = [], []
    char_idx = 0  # tracks our current position in `string`

    for token, logprob in zip(tokens, token_logprobs):
        token_start = char_idx
        token_end = char_idx + len(token)
        # Check for overlap between token span and subtring span
        if token_end > start_idx and token_start < end_idx:
            extracted_tokens.append(token)
            extracted_logprobs.append(logprob)
        char_idx = token_end  # move to the next token's start position

    if logger:
        logger.info(f"Extracted logprobs for: {''.join(extracted_tokens)}")

    return extracted_logprobs
