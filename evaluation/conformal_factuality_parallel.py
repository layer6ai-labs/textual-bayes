# https://github.com/tatsu-lab/conformal-factual-lm/blob/main/src/sayless.py

import io
import json
from math import ceil
import numpy as np
from openai import OpenAI
import os
import sys
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from textgrad.variable import Variable
from omegaconf import OmegaConf

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project root: {project_root}")
sys.path.append(project_root)
from method.models import EnsembleModel


import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Get logger for this module
log = logging.getLogger("ConformalFactuality")

# Load environment variables from a .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("Please set the OPENAI_API_KEY environment variable.")
# Default prompt to break into subclaims.
BREAKDOWN_PROMPT = (
    "Please breakdown the following input into a set of small, independent claims, "
    "and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n"
    "The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts "
    "and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult "
    "for anyone to know, like the birthdays of non-notable people. The input is: "
)
BREAKDOWN_SYSTEM_PROMPT = (
    "You are a helpful assistant whose job is to break down your inputs "
    "into a set of small claims so that a human can easily check each one. "
    "Make sure that each claim is small and non-overlapping."
)
DATASET_SYSTEM_PROMPT = "You are a helpful assistant write Bio for people."
GPT_client = OpenAI(api_key=OPENAI_API_KEY)

BASE_MODEL = "gpt-4"


class ConformalFactualityParallel:
    def __init__(
        self,
        sample_methods={"gpt-4": "gpt-4", "textual-bayes": None},
        system_prompt=DATASET_SYSTEM_PROMPT,
        base_model=BASE_MODEL,
    ):
        self.sample_methods = sample_methods
        self.DATASET_SYSTEM_PROMPT = system_prompt
        self.BASE_MODEL = base_model

    def process_one(self, data):
        # Create a new client instance for each process
        client = OpenAI(api_key=OPENAI_API_KEY)
        input = data["question"]
        result = {}
        if self.generate_original_answer:
            log.info(f"Generating original answer for {input}")
            result["original"] = self.query_gpt(
                client, input, self.BASE_MODEL, system_prompt=self.DATASET_SYSTEM_PROMPT
            )
        method_answers = {}
        for method in self.sample_methods:
            log.info(f"Generating alternative answers for {input} with {method}")
            method_answers[method] = self.get_alternate_outputs(
                input,
                self.n_samples,
                self.sample_methods[method],
                client=client,
                **self.model_kwargs,
            )
        result["alternatives"] = method_answers
        return result

    def prepare_for_conformal_factuality(
        self,
        dataset,
        n_samples=5,
        original_answers=None,
        alternative_answers=None,
        max_workers=None,
        **model_kwargs,
    ):
        log.info(f"Preparing for conformal factuality for dataset {len(dataset)} examples")
        self.generate_original_answer = False
        if original_answers is None:
            original_answers = []
            self.generate_original_answer = True
        if alternative_answers is None:
            alternative_answers = []

        # Store parameters as instance variables for process_one to access
        self.n_samples = n_samples
        self.model_kwargs = model_kwargs

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self.process_one, dataset),
                    total=len(dataset),
                    desc="Generating answers in parallel",
                )
            )

        # Collect results
        for result in results:
            if self.generate_original_answer:
                original_answers.append(result["original"])
            alternative_answers.append(result["alternatives"])

        return original_answers, alternative_answers

    def load_annotation(self, annotation_file):
        log.info(f"Loading the annotation from {annotation_file}")
        if os.path.exists(annotation_file):
            annotation = json.load(open(annotation_file))
            calculated_methods = [
                m.replace("frequency-score-", "")
                for m in annotation[0]["claims"][0].keys()
                if "frequency-score-" in m
            ]
            if set(calculated_methods) == set(self.sample_methods):
                log.info(f"Annotation file contains the same methods as the current methods.")
                return annotation
            else:
                log.warning(
                    f"Warning: The annotation file contains different methods than the current methods. Will create new"
                )
        else:
            log.info(f"Annotation file not found. Will create new.")
            return None

    def run_conformal_factuality(
        self,
        dataset,
        dataset_name,
        a=0.96,
        alphas=np.arange(0.05, 0.80, 0.05),
        n_samples=5,
        original_answer=None,
        alternative_answers=None,
        conformal_runs=1000,
        out_dir="outputs/conformal-factuality/",
        annotation_file=None,
        max_workers=None,
        **model_kwargs,
    ):
        """
        Run the conformal factuality task.
        Args:
            max_workers: Number of parallel processes to use for annotation. If None, uses os.cpu_count()
        """
        log.info(
            f"Running conformal factuality for dataset {dataset_name} with {len(dataset)} examples"
        )

        # Store out_dir as instance variable
        self.out_dir = out_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Define paths for intermediate outputs
        if annotation_file is None:
            annotation_file = f"{out_dir}/{dataset_name}-annotation.json"
        answers_file = f"{out_dir}/{dataset_name}-answers.json"

        # Try to load existing annotations
        if os.path.exists(annotation_file):
            log.info(f"Loading existing annotations from {annotation_file}")
            with open(annotation_file, "r") as f:
                saved_annotation = json.load(f)

            # Check which methods are already annotated
            existing_methods = set()
            if saved_annotation and "claims" in saved_annotation[0]:
                for claim in saved_annotation[0]["claims"]:
                    for key in claim.keys():
                        if key.startswith("frequency-score-"):
                            existing_methods.add(key.replace("frequency-score-", ""))

            log.info(f"Found existing annotations for methods: {existing_methods}")

            # Determine which methods need new annotations
            methods_to_annotate = set(self.sample_methods.keys()) - existing_methods

            if not methods_to_annotate:
                log.info("All methods already annotated, using saved annotations")
                annotation = saved_annotation
            else:
                log.info(f"Generating new annotations for methods: {methods_to_annotate}")
                # Use existing annotations as base
                annotation = saved_annotation
                # Generate new annotations for missing methods
                new_annotation = self.get_annotations_and_scores(
                    dataset,
                    n_samples,
                    original_answer,
                    alternative_answers,
                    max_workers=max_workers,
                    methods_to_process=list(methods_to_annotate),
                    **model_kwargs,
                )

                # Merge new annotations with existing ones
                for i, data in enumerate(annotation):
                    for method in methods_to_annotate:
                        for j, claim in enumerate(data["claims"]):
                            claim[f"frequency-score-{method}"] = new_annotation[i]["claims"][j][
                                f"frequency-score-{method}"
                            ]

                # Save the updated annotations
                log.info(f"Saving updated annotations to {annotation_file}")
                with open(annotation_file, "w") as f:
                    json.dump(annotation, f, indent=4)
        else:
            if os.path.exists(answers_file):
                log.info(f"Loading existing answers from {answers_file}")
                with open(answers_file, "r") as f:
                    saved_data = json.load(f)
                    original_answer = saved_data["original_answers"]
                    alternative_answers = saved_data["alternative_answers"]

            else:
                log.info("Generating new answers...")
                original_answer, alternative_answers = self.prepare_for_conformal_factuality(
                    dataset,
                    n_samples,
                    original_answer,
                    alternative_answers,
                    max_workers=max_workers,
                    **model_kwargs,
                )
                # Save the generated answers
                log.info(f"Saving answers to {answers_file}")
                with open(answers_file, "w") as f:
                    json.dump(
                        {
                            "original_answers": original_answer,
                            "alternative_answers": alternative_answers,
                        },
                        f,
                        indent=4,
                    )

            log.info("Generating new annotations...")
            annotation = self.get_annotations_and_scores(
                dataset,
                n_samples,
                original_answer,
                alternative_answers,
                max_workers=max_workers,
                **model_kwargs,
            )
            # Save the annotations
            log.info(f"Saving annotations to {annotation_file}")
            with open(annotation_file, "w") as f:
                json.dump(annotation, f, indent=4)

        results = {
            "method": [],
            "alpha": [],
            "run_id": [],
            "threshold": [],
            "removal_fraction": [],
            "empirical_factuality": [],
            "removal_precision": [],
            "removal_recall": [],
            "removal_fractions": [],
            "empirical_factualities": [],
        }
        for method in self.sample_methods:
            log.info(f"Running conformal factuality for method {method}")
            for data in annotation:
                for claim in data["claims"]:
                    try:
                        score = claim.get(f"frequency-score-{method}")
                        if score is not None:
                            claim[f"frequency-score-{method}"] = float(score)
                        else:
                            claim[f"frequency-score-{method}"] = 0.0
                    except (ValueError, TypeError):
                        log.warning(
                            f"Could not convert frequency score for method {method}, setting to 0.0"
                        )
                        claim[f"frequency-score-{method}"] = 0.0
                    if "noise" in claim and claim["noise"]:
                        claim[f"frequency-score-{method}"] += claim["noise"]
                    else:
                        claim[f"frequency-score-{method}"] += np.random.normal(0, 0.001)
            for run_id in tqdm(range(conformal_runs), desc="Conformal Runs Progress"):
                np.random.shuffle(annotation)
                calibration_set = annotation[: int(len(annotation) * 0.5)]
                test_set = annotation[int(len(annotation) * 0.5) :]
                test_set_copy = test_set.copy()
                for alpha in alphas:
                    removal_fractions = []
                    empirical_factualities = []
                    removal_precisions = []
                    removal_recalls = []
                    threshold = self.compute_threshold(calibration_set, alpha, method, a)
                    for i, data in enumerate(test_set):
                        all_subclaims = data["claims"]
                        accepted_subclaims = [
                            subclaim
                            for subclaim in all_subclaims
                            if subclaim[f"frequency-score-{method}"] > threshold
                        ]
                        removed_claims = [
                            claim for claim in all_subclaims if claim not in accepted_subclaims
                        ]
                        if run_id == 0:
                            test_set_copy[i][
                                f"accepted_subclaims_with_alpha_{alpha}"
                            ] = accepted_subclaims
                        removal_fraction = (len(all_subclaims) - len(accepted_subclaims)) / len(
                            all_subclaims
                        )
                        if len(accepted_subclaims) > 0:
                            empirical_factuality = sum(
                                [subclaim["factual"] for subclaim in accepted_subclaims]
                            ) / len(accepted_subclaims)
                            empirical_factuality = int(empirical_factuality >= a)
                        else:
                            empirical_factuality = 1

                        removal_precision = 1 - sum([claim["factual"] for claim in removed_claims])
                        if sum([claim["factual"] == 0 for claim in all_subclaims]) == 0:
                            removal_recall = 1
                        else:
                            removal_recall = sum(
                                claim["factual"] == 0 for claim in removed_claims
                            ) / sum([claim["factual"] == 0 for claim in all_subclaims])

                        removal_fractions.append(removal_fraction)
                        empirical_factualities.append(empirical_factuality)
                        removal_precisions.append(removal_precision)
                        removal_recalls.append(removal_recall)

                    removal_fraction = np.mean(removal_fractions)
                    empirical_factuality = np.mean(empirical_factualities)
                    removal_precision = np.mean(removal_precisions)
                    removal_recall = np.mean(removal_recalls)

                    results["method"].append(method)
                    results["alpha"].append(alpha)
                    results["run_id"].append(run_id)
                    results["threshold"].append(threshold)
                    results["removal_fraction"].append(removal_fraction)
                    results["empirical_factuality"].append(empirical_factuality)
                    results["removal_precision"].append(removal_precision)
                    results["removal_recall"].append(removal_recall)
                    results["removal_fractions"].append(removal_fractions)
                    results["empirical_factualities"].append(empirical_factualities)
        df = pd.DataFrame(results)
        return df

    def process_annotation(self, args):
        try:
            data, original_answer, alternative_answer_method = args
            question = data["question"]
            data_annotation = {"prompt": question, "original_answer": original_answer}
            golden_answer = data.get("golden_answer", None)
            subclaims = self.get_subclaims(original_answer, claim_llm="gpt-4o-mini")
            subclaims_annotation = self.get_entailment_label(
                subclaims, question, golden_answer=golden_answer
            )

            # Process only specified methods if methods_to_process is set
            methods_to_process = (
                self.methods_to_process if hasattr(self, "methods_to_process") else None
            )
            methods = methods_to_process if methods_to_process else self.sample_methods.keys()

            for method in methods:
                alternative_answer = alternative_answer_method[method]
                frequency_scores = self.get_frequency_score(
                    subclaims, alternative_answer, **self.model_kwargs
                )
                for i, subclaim in enumerate(subclaims_annotation):
                    subclaim[f"frequency-score-{method}"] = frequency_scores[i]
                    subclaim["noise"] = np.random.normal(0, 0.001)
            data_annotation["claims"] = subclaims_annotation
            return data_annotation
        except Exception as e:
            log.error(f"Error in process_annotation: {e}")
            # Return a minimal valid annotation to prevent process pool from breaking
            return {
                "prompt": data["question"],
                "original_answer": original_answer,
                "claims": [{"subclaim": "error", "factual": 0, "source": "error"}],
            }

    def get_annotations_and_scores(
        self,
        dataset,
        n_samples,
        original_answers,
        alternative_answers,
        max_workers=None,
        methods_to_process=None,
        **model_kwargs,
    ):
        """
        Parallelized version of annotation and scoring.
        Args:
            max_workers: Number of parallel processes to use. If None, uses os.cpu_count()
            methods_to_process: List of methods to process. If None, process all methods.
        """
        log.info(f"Getting annotations and scores for dataset {len(dataset)} examples")
        log.info(f"max_workers: {max_workers}")
        if methods_to_process:
            log.info(f"Processing methods: {methods_to_process}")
        else:
            log.info("Processing all methods")

        # Store model_kwargs as instance variable for process_annotation to access
        self.model_kwargs = model_kwargs
        self.methods_to_process = methods_to_process
        args_list = [
            (data, original_answers[idx], alternative_answers[idx])
            for idx, data in enumerate(dataset)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            annotation = list(
                tqdm(
                    executor.map(self.process_annotation, args_list),
                    total=len(args_list),
                    desc="Parallel Processing",
                )
            )
        return annotation

    def compute_threshold(self, annotation, alpha, method, a=0.7):
        partial_entailment_scores = []
        for data in annotation:
            scores = [subclaim[f"frequency-score-{method}"] for subclaim in data["claims"]]
            threshold_set = sorted(scores, reverse=True)
            partial_entailment_score = -1000
            for threshold in threshold_set:
                if partial_entailment_score == -1000:
                    accepted_subclaims = [
                        subclaim
                        for subclaim in data["claims"]
                        if subclaim[f"frequency-score-{method}"] >= threshold
                    ]
                    if len(accepted_subclaims) == 0:
                        entailed_fraction = 1
                    else:
                        entailed_fraction = sum(
                            [subclaim["factual"] for subclaim in accepted_subclaims]
                        ) / len(accepted_subclaims)
                    if entailed_fraction < a:
                        partial_entailment_score = threshold
            partial_entailment_scores.append(partial_entailment_score)
        quantile_target_index = ceil((len(partial_entailment_scores) + 1) * (1 - alpha))
        threshold = sorted(partial_entailment_scores)[quantile_target_index - 1]
        return threshold

    def get_frequency_score(self, subclaims, alternative_answer, **model_kwargs):
        # ... existing code ...
        final_scores = [0.0] * len(subclaims)
        subclaims = [{"id": i, "subclaim": claim} for i, claim in enumerate(subclaims)]
        for output in alternative_answer:
            counting_prompt = f""" You will get a list of claims and piece of text. 
                For each claim, score whether the text supports, contradicts, 
                or is unrelated to the claim. Directly return a jsonl, where each line is 
                {{"id":[CLAIM_ID], "score":[SCORE]}}. Each line should be sdeperated by newline. 
                Directly return the jsonl with 
                no explanation or other formatting. For the [SCORE], return 1 for supports, 
                -1 for contradicts, and 0 for unrelated. The claims are:\n{subclaims}\n\n
                The text is:\n{output}"""
            output = self.query_gpt(
                GPT_client, counting_prompt, model="gpt-4", max_tokens=5000, temperature=0
            )
            output = output.replace("```jsonl\n", "")
            output = output.replace("```", "")
            output = output.replace("\\", "\\\\")
            output = output.replace("```json", "")
            try:
                for i, line in enumerate(output.splitlines()):
                    scores = json.loads(line)
                    idx = int(scores["id"])
                    final_scores[idx] += float(scores["score"])
            except Exception as ex:
                log.error(ex)
                log.error("======Failed to parse as jsonl 3======")
                log.error(output)
        return final_scores

    def get_subclaims(
        self,
        output,
        claim_llm="gpt-4o-mini",
        breakdown_prompt=BREAKDOWN_PROMPT,
        max_tokens=8000,
        temperature=0,
    ):
        # ... existing code ...
        output = self.query_gpt(
            GPT_client,
            prompt=breakdown_prompt + output,
            model=claim_llm,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=BREAKDOWN_SYSTEM_PROMPT,
        )
        output = output.replace("```jsonl\n", "")
        output = output.replace("\\", "\\\\")
        subclaims = output.replace("```", "")
        try:
            subclaims = [json.loads(line) for line in subclaims.splitlines() if line]
            return subclaims
        except Exception as ex:
            log.error(ex)
            log.error("======Failed to parse as jsonl 1======")
            log.error(subclaims)
            return None

    def get_entailment_label(
        self, claims, question, entailment_model="gpt-4o-mini", golden_answer=None
    ):
        subclaims_annotation = []
        claims_list = [claim["subclaim"] for claim in claims]
        claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims_list)])

        if golden_answer:
            prompt = f"""Given the question: {question}, and the response: {golden_answer}, 
                        verify if each of these claims is supported by the response.
                        Claims:
                        {claims_text}
                        
                        Return your answer as a JSON array, where each element is an object with these keys:
                        {{"subclaim": "[CLAIM]", "supported": 1 or 0, "reason": "explanation"}}
                        
                        Format your response as a valid JSON array only, with no additional text or formatting.
                        Example format:
                        [
                            {{"subclaim": "claim 1", "supported": 1, "reason": "explanation"}},
                            {{"subclaim": "claim 2", "supported": 0, "reason": "explanation"}}
                        ]"""
        else:
            prompt = f"""Please verify if each of these claims is factual.
                        Claims:
                        {claims_text}
                        
                        Return your answer as a JSON array, where each element is an object with these keys:
                        {{"subclaim": "[CLAIM]", "factual": 1 or 0, "source": "source or explanation"}}
                        
                        Format your response as a valid JSON array only, with no additional text or formatting.
                        Example format:
                        [
                            {{"subclaim": "claim 1", "factual": 1, "source": "source"}},
                            {{"subclaim": "claim 2", "factual": 0, "source": "source"}}
                        ]"""

        try:
            response = GPT_client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search_preview", "search_context_size": "low"}],
                input=prompt,
            )
            response_content = response.output_text
        except Exception as e:
            log.error(f"Error in web search: {e}")
            # Fallback to regular GPT call if web search fails
            response_content = self.query_gpt(
                GPT_client, prompt, entailment_model, max_tokens=5000, temperature=0
            )

        # Clean up the response content
        response_content = response_content.replace("```jsonl\n", "")
        response_content = response_content.replace("\\", "\\\\")
        response_content = response_content.replace("```", "")
        response_content = response_content.replace("json", "")
        response_content = response_content.strip()

        try:
            # Try parsing as a JSON array first
            try:
                results = json.loads(response_content)
                if isinstance(results, list):
                    for result in results:
                        if "supported" in result:
                            factual = result["supported"]
                            source = result.get("reason", golden_answer)
                        else:
                            factual = result["factual"]
                            source = result.get("source", "no source provided")

                        subclaims_annotation.append(
                            {
                                "subclaim": result["subclaim"],
                                "factual": int(factual),
                                "source": source,
                            }
                        )
                else:
                    raise ValueError("Response is not a JSON array")
            except json.JSONDecodeError:
                # If JSON array parsing fails, try parsing line by line
                for line in response_content.splitlines():
                    if not line.strip():
                        continue
                    # Clean up the line
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        data = json.loads(line)
                        if "supported" in data:
                            factual = data["supported"]
                            source = data.get("reason", golden_answer)
                        else:
                            factual = data["factual"]
                            source = data.get("source", "no source provided")

                        subclaims_annotation.append(
                            {
                                "subclaim": data["subclaim"],
                                "factual": int(factual),
                                "source": source,
                            }
                        )
        except Exception as ex:
            log.error(ex)
            log.error("======Failed to parse as jsonl 2======")
            log.error(response_content)
            # Create default responses for all claims
            for claim in claims_list:
                subclaims_annotation.append(
                    {"subclaim": claim, "factual": 0, "source": "parsing_error"}
                )

        return subclaims_annotation

    def get_alternate_outputs(self, prompt, n_samples, model, client=None, **model_kwargs):
        # Use provided client or create new one
        if client is None:
            client = GPT_client
        alternate_outputs = []
        if type(model) == EnsembleModel:
            for _ in range(n_samples):
                if type(prompt) == str:
                    prompt = Variable(
                        prompt, requires_grad=False, role_description="factscore question"
                    )
                output = [answer["outputs"][-1].value for answer in model(prompt, **model_kwargs)]
                log.info(f"Generated alternative answer for {prompt}")
                alternate_outputs.append(output)
        elif type(model) == str:
            if type(prompt) == Variable:
                prompt = prompt.value
            alternate_outputs = self.query_gpt(
                client,
                prompt,
                model,
                n_samples=n_samples,
                temperature=1,
                system_prompt=self.DATASET_SYSTEM_PROMPT,
            )
        return alternate_outputs

    def conformal_factuality_removal(
        self, prompt, model, original_output, alternative_outputs, threshold, method, n_samples=5
    ):
        # ... existing code ...
        subclaims = self.get_subclaims(GPT_client, original_output, claim_llm="gpt-4o-mini")
        frequency_scores = self.get_frequency_score(subclaims, alternative_outputs, method=method)
        for i, subclaim in enumerate(subclaims):
            subclaim[f"frequency-score-{method}"] = frequency_scores[i]
        accepted_subclaims = [
            subclaim for subclaim in subclaims if subclaim[f"frequency-score-{method}"] > threshold
        ]
        merged_output = self.merge_subclaims(GPT_client, accepted_subclaims, model, prompt)
        return merged_output, (accepted_subclaims, subclaims)

    def default_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [f"{i}: {subclaim['subclaim']}" for i, subclaim in enumerate(subclaims)]
        )
        return (
            "You will receive an instruction and a set of facts that are true. "
            "Construct an answer using ONLY the facts provided, and try to use all facts as long as possible. "
            "If no facts are given, reply to the instruction acknowledging that you don't have enough information to fully respond.\n\n"
            f"The facts:\n{claim_string}\n\n"
            f"The instruction:\n{prompt}"
        )

    def merge_subclaims(
        self, subclaims, prompt, merge_llm="gpt-4o-mini", create_merge_prompt=default_merge_prompt
    ):
        prompt = create_merge_prompt(subclaims, prompt)
        output = (
            self.query_gpt(prompt, merge_llm, max_tokens=5000, temperature=0)
            if subclaims
            else "Abstain."
        )
        return output

    def query_gpt(
        self, client, prompt, model, max_tokens=5000, temperature=0, n_samples=1, system_prompt=None
    ):
        if type(prompt) == Variable:
            prompt = prompt.value
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            if type(system_prompt) == Variable:
                system_prompt = system_prompt.value
            messages.insert(0, {"role": "system", "content": system_prompt})
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n_samples,
        )
        if n_samples > 1:
            return [choice.message.content for choice in completion.choices]
        else:
            return completion.choices[0].message.content

    def plot_result(self, results, exp_name, errorbar=True):
        # ... existing code ...
        methods = results["method"].unique()
        plt.figure(figsize=(8, 6))
        for method in methods:
            method_results = results[results["method"] == method]
            grouped_results = (
                method_results.groupby("alpha")["empirical_factuality"]
                .agg(["mean", "std"])
                .reset_index()
            )
            x = 1 - grouped_results["alpha"]
            y = grouped_results["mean"]
            yerr = grouped_results["std"]
            # if errorbar:
            #     plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label=f"{method} Empirical Factuality")
            # else:
            #     plt.plot(x, y, linestyle='-', label=f"{method} Empirical Factuality")
            plt.plot(x, y, linestyle="-", label=f"{method} Empirical Factuality")
        plt.plot(x, x, linestyle="--", label="Ideal Factuality", color="grey")
        plt.xlabel("1 - Alpha")
        plt.ylabel("Average Empirical Factuality")
        plt.title("Empirical Factuality vs 1 - Alpha")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.out_dir}/factuality_{exp_name}.png")
        plt.savefig(f"{self.out_dir}/factuality_{exp_name}.pdf", bbox_inches="tight")
        log.info(f"Plot saved as {self.out_dir}/factuality_{exp_name}.png")
        plt.show()
        plt.figure(figsize=(8, 6))
        for method in methods:
            method_results = results[results["method"] == method]
            grouped_removal = (
                method_results.groupby("alpha")["removal_fraction"]
                .agg(["mean", "std"])
                .reset_index()
            )
            x_removal = 1 - grouped_removal["alpha"]
            y_removal = grouped_removal["mean"]
            yerr_removal = grouped_removal["std"]
            yerr_removal = yerr_removal * 1.96 / np.sqrt(len(yerr_removal))
            if errorbar:
                plt.errorbar(
                    x_removal,
                    y_removal,
                    yerr=yerr_removal,
                    fmt="o-",
                    capsize=5,
                    label=f"{method} Removal Rate",
                )
            else:
                plt.plot(x_removal, y_removal, linestyle="-", label=f"{method} Removal Rate")
        plt.xlabel("1 - Alpha")
        plt.ylabel("Average Removal Rate")
        plt.title("Removal Rate vs 1 - Alpha")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.out_dir}/removal_rate_{exp_name}.png")
        plt.savefig(f"{self.out_dir}/removal_rate_{exp_name}.pdf", bbox_inches="tight")
        log.info(f"Plot saved as {self.out_dir}/removal_rate_{exp_name}.png")
        plt.show()

        # Add new plot for removal rate with std from individual fractions
        plt.figure(figsize=(8, 6))
        for method in methods:
            method_results = results[results["method"] == method]

            # Group by alpha and calculate mean and standard error
            grouped = method_results.groupby("alpha")
            x_values = []
            y_values = []
            yerr_values = []

            for alpha, group in grouped:
                x_values.append(1 - alpha)
                # Calculate mean and std from individual removal fractions
                all_removal_fractions = []
                for fractions in group["removal_fractions"]:
                    all_removal_fractions.extend(fractions)
                y_values.append(np.mean(all_removal_fractions))
                yerr_values.append(
                    1.96 * np.std(all_removal_fractions) / np.sqrt(len(all_removal_fractions))
                )

            # Sort by x values for proper line plotting
            sorted_indices = np.argsort(x_values)
            x_values = np.array(x_values)[sorted_indices]
            y_values = np.array(y_values)[sorted_indices]
            yerr_values = np.array(yerr_values)[sorted_indices]

            # Plot with error bars
            plt.errorbar(
                x_values,
                y_values,
                yerr=yerr_values,
                fmt="o-",
                capsize=5,
                label=f"{method} Removal Rate",
                markersize=6,
                linewidth=2,
                capthick=2,
            )

        plt.xlabel("1 - Alpha")
        plt.ylabel("Average Removal Rate")
        plt.title("Removal Rate vs 1 - Alpha (with Individual Std Error)")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.out_dir}/removal_rate_std_{exp_name}.png")
        plt.savefig(f"{self.out_dir}/removal_rate_std_{exp_name}.pdf", bbox_inches="tight")
        log.info(f"Plot saved as {self.out_dir}/removal_rate_std_{exp_name}.png")
        plt.show()


if __name__ == "__main__":
    # Example usage
    # with io.open("datasets/fact_score/fact_score_names.txt", "r") as fopen:
    #     factscore_names = fopen.readlines()[:]
    # dataset_name = "factscore_sample"
    # dataset = [{"question": f"Tell me about {name.strip()}"} for name in factscore_names]
    dataset_name = "factscore"
    # Read JSONL file line by line
    dataset = []
    with open("datasets/fact_score/factscore_test.jsonl", "r") as f:
        for i, line in enumerate(f):
            dataset.append(json.loads(line.strip()))
    sample_model = "gpt-4"  # Replace with your actual model
    conformal_factuality = ConformalFactualityParallel(sample_methods={"gpt-4": "gpt-4"})
    results = conformal_factuality.run_conformal_factuality(
        dataset,
        dataset_name,
        a=0.96,
        alphas=np.arange(0.05, 0.80, 0.05),
        conformal_runs=1000,
        out_dir="outputs/factuality_baseline/",
    )
    results.to_csv(f"outputs/factuality_baseline/{dataset_name}-results.csv", index=False)
    conformal_factuality.plot_result(results, exp_name=dataset_name, errorbar=True)
    print(results)
