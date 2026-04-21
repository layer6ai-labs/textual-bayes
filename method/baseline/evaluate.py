import logging
import re
from collections import Counter

import numpy as np

from evaluation.metrics import accuracy, compute_auc_roc, expected_calibration_error, get_f1_scores

logger = logging.getLogger("Evaluation")


def baseline_qasper(results):
    results_answerable = [result for result in results if result["abst_true"] == 0]
    y_true = np.stack([result["y_true"] for result in results_answerable])
    y_pred = np.array([result["y_pred"] for result in results_answerable])
    y_conf = np.array([result["y_conf"] for result in results_answerable])
    equivalence = np.array([result["equivalence"] for result in results_answerable])
    f1, _ = get_f1_scores(y_true, y_pred)
    acc = np.sum(equivalence) / y_true.shape[0]
    ece = expected_calibration_error(equivalence, y_conf, num_bins=10)
    logger.info(
        f"\n"
        f"ECE: {ece:.4f}\n{'*' * 70}\n"
        f"ACC: {acc:.4f}\n{'*' * 70}\n"
        f"F1: {f1:.4f}\n{'*' * 70}\n"
    )

    # abstention related metrics
    abst_true = np.stack([result["abst_true"] for result in results])
    abst_pred = np.stack([result["abst_pred"] for result in results])
    abst_roc = compute_auc_roc(abst_true, abst_pred)
    logger.info(f"\nAbst Roc: {abst_roc:.4f}\n{'*' * 70}\n")


def baseline_simpleQA(results):
    y_true = np.stack([result["y_true"] for result in results])
    y_conf = np.array([result["y_conf"] for result in results])
    equivalence = np.array([result["equivalence"] for result in results])
    acc = np.sum(equivalence) / y_true.shape[0]
    ece = expected_calibration_error(equivalence, y_conf, num_bins=10)
    logger.info(f"\n" f"ECE: {ece:.4f}\n{'*' * 70}\n" f"ACC: {acc:.4f}\n{'*' * 70}\n")


def baseline_multi_choice(
    examples,
    answer_regex,
    choices,
):
    results = []
    for i, example in enumerate(examples):
        answers_parsed = []
        for answer in example["responses"]:
            match = re.search(answer_regex, answer)
            if match:
                value = match.group(1)
                answers_parsed.append(value)
            else:
                answers_parsed.append(None)
        # Amalgamate LLM textual outputs into a probability vector
        y_probs = np.array(
            [np.mean([answer == letter for answer in answers_parsed]) for letter in choices]
        )

        # Handle corner cases where textual output isn't structured correctly
        if y_probs.sum() == 0:
            logger.info("zero y_probs.")
            y_probs = np.array([1 / len(choices) for _ in choices])
        elif y_probs.sum() < 1.0:
            logger.info("y_probs sum to less than 1.")
            y_probs /= y_probs.sum()

        results.append(
            {
                "y_true": choices.index(example["ground_truth"]),
                "y_probs": y_probs,
                "y_conf": example["confidence"],
                "answers_full": example["responses"],
                "answers_parsed": answers_parsed,
            }
        )
        logger.info(
            f"Validation Example: #{i + 1}\n Question: {example['raw_input_example']}\nPrediction: {y_probs}\nAnswer: {example['ground_truth']}"
        )

    y_true = np.stack([result["y_true"] for result in results])
    y_probs = np.array([result["y_probs"] for result in results])
    y_pred = np.argmax(y_probs, axis=-1)  # Model's top answer
    y_conf = np.stack([result["y_conf"] for result in results])

    # Compute metrics
    ece = expected_calibration_error(y_true == y_pred, y_conf, num_bins=10)
    acc = accuracy(y_true, y_pred)
    logger.info(f"\nECE: {ece:.2f}\n{'*' * 70}\nACC: {acc:.2f}\n{'*' * 70}\n")


def baseline_int(
    examples,
    answer_regex,
):
    results = []
    for i, example in enumerate(examples):
        answers_parsed = []
        for answer in example["responses"]:
            match = re.search(answer_regex, answer)
            if match:
                value = int(match.group(1))
                answers_parsed.append(value)
            else:
                answers_parsed.append(None)

            # Amalgamate LLM textual outputs into a probability vector
            counter = Counter(answers_parsed)
            answers_len = len(answers_parsed)
            ys = {answer: count / answers_len for answer, count in counter.items()}

            results.append(
                {
                    "y_true": int(example["ground_truth"]),
                    "y_pred": max(ys, key=ys.get),
                    # "y_conf": max(ys.values()),
                    "y_conf": example["confidence"],
                    "answers_parsed": answers_parsed,
                    "answers_full": example["responses"],
                }
            )

            logger.info(
                f"Validation Example: #{i + 1}\n Question: {example['raw_input_example']}\nPrediction: {ys}\nAnswer: {example['ground_truth']}"
            )

    y_true = np.stack([result["y_true"] for result in results])
    y_pred = np.array([result["y_pred"] for result in results])
    y_conf = np.array([result["y_conf"] for result in results])

    # Compute metrics
    ece = expected_calibration_error(y_true == y_pred, y_conf, num_bins=10)
    acc = accuracy(y_true, y_pred)
    # Brier Score cannot be computed for GSM8K
    logger.info(f"\nECE: {ece:.2f}\n{'*' * 70}\nACC: {acc:.2f}\n")
