import re
import string
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score


# adapted based on https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
def expected_calibration_error(correctness, y_conf, num_bins=10):
    """
    Input:
        correctness: shape(num_data) indicates the correctness of each datapoint (can be hard 0/1 or soft probabilities)
        y_conf: shape(num_data) indicating the probability assigned to the predicted output
        n_bins (optional): number of bins to compute
    Output:
        ECE
    """
    assert all(y_conf > 0) and all(y_conf <= 1)

    correct = np.array(correctness, dtype=np.float32)

    b = np.linspace(start=0, stop=1.0, num=num_bins + 1)
    bins = np.digitize(y_conf, bins=b, right=True)  # array of bin indices per datapoint
    # note: bin indices go from 1 : (0, 0.1] until 10: (0.9: 1.0]

    o = 0
    for b in range(1, num_bins + 1):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - y_conf[mask]))

    return o / y_conf.shape[0]


def accuracy(y_true, y_pred):
    """
    Input:
        y_true: shape(num_data) which indicates the correct index of correct answer
        y_pred: shape(num_data) which indicates the predicted index of correct answer
    Output:
        accuracy
    """
    correct = (y_pred == y_true).astype(np.float32)
    return np.sum(correct) / y_true.shape[0]


def brier_score(y_true, y_probs):
    """
    Input:
        y_true: shape(num_data) which indicates the index of correct answer (starting from 0)
        y_probs: shape(num_data, num_multi_choices) which for each question it returns the probability of each answer
    Output:
        Brier score
    """
    assert np.isclose(np.sum(y_probs, axis=1), 1).all()

    brier = (
        1
        + (np.sum(y_probs**2) - 2 * np.sum(y_probs[np.arange(y_probs.shape[0]), y_true]))
        / y_true.shape[0]
    )
    return brier


def compute_auc_roc(y_true, y_probs):
    """
    Computes the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

    Parameters:
    y_true (list or array): True binary labels (0 or 1).
    y_probs (list or array): Predicted probabilities for the positive class.

    Returns:
    float: AUC-ROC score
    """
    return roc_auc_score(y_true, y_probs)


def compute_abstention_rate(y_true, y_pred, abstention_label=1):
    """
    Computes the abstention rate.

    Parameters:
    y_true (list or array): True labels.
    y_pred (list or array): Predicted labels, where abstentions are marked with a specific label (default: -1).

    Returns:
    float: Abstention rate (proportion of abstained predictions).
    """
    abstentions = sum(1 for y in y_pred if y == abstention_label)
    return abstentions / len(y_pred)


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(ground_truth, prediction):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.

    Parameters:
        ground_truth (str): The ground truth answer.
        prediction (str): The predicted answer.
    Returns:
        float: The token F1 score between the ground truth and prediction.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_f1_scores(ground_truths, predictions):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.

    Parameters:
        ground_truths (list of str): The list of ground truth answers.
        predictions (list of str): The list of predicted answers.
    Returns:
        float: The average F1 score across all predictions.
        list of float: The list of F1 scores for each prediction-ground truth pair.
    """
    f1_score_list = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        f1_score = token_f1_score(prediction, ground_truth)
        f1_score_list.append(f1_score)
    avg_f1_score = np.array(f1_score_list).mean()

    return avg_f1_score, f1_score_list
