"""Metrics for a single (model, split) run."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn import metrics as skm

HEADLINE_METRICS = ("f1", "pr_auc", "mcc", "recall", "precision")
LOWER_IS_BETTER = frozenset({"fpr", "fnr", "log_loss", "fit_seconds", "predict_seconds"})


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, is_probability: bool
) -> dict[str, Any]:
    """Compute every threshold-based and ranking metric for one run.

    Args:
        y_true: True 0/1 labels.
        y_pred: Predicted 0/1 labels.
        y_score: Continuous anomaly scores (higher = more anomalous).
        is_probability: True if ``y_score`` is a probability, enabling log loss.

    Returns:
        A dictionary of metric name to value (NaN where undefined).
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    result: dict[str, Any] = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": _safe(skm.balanced_accuracy_score, y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, zero_division=0),
        "specificity": _ratio(tn, tn + fp),
        "f1": skm.f1_score(y_true, y_pred, zero_division=0),
        "f2": skm.fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": _safe(skm.matthews_corrcoef, y_true, y_pred),
        "cohen_kappa": _safe(skm.cohen_kappa_score, y_true, y_pred),
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    result.update(ranking_metrics(y_true, y_score, is_probability))
    return {key: _to_builtin(value) for key, value in result.items()}


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, is_probability: bool) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC and (for probabilities) log loss.

    Args:
        y_true: True 0/1 labels.
        y_score: Continuous anomaly scores.
        is_probability: Whether log loss applies.

    Returns:
        ``roc_auc``, ``pr_auc`` and ``log_loss`` (NaN when undefined).
    """
    single_class = len(np.unique(y_true)) < 2
    roc = float("nan") if single_class else skm.roc_auc_score(y_true, y_score)
    pr = float("nan") if single_class else skm.average_precision_score(y_true, y_score)
    loss = float("nan")
    if is_probability and not single_class:
        clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
        loss = skm.log_loss(y_true, clipped, labels=[0, 1])
    return {"roc_auc": roc, "pr_auc": pr, "log_loss": loss}


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return the confusion matrix as ``(tn, fp, fn, tp)``, robust to a single class.

    Args:
        y_true: True 0/1 labels.
        y_pred: Predicted 0/1 labels.

    Returns:
        Integer counts.
    """
    matrix = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find the score threshold that maximises F1 on the given (training) data.

    Args:
        y_true: True 0/1 labels of the train portion.
        y_score: Scores of the train portion.

    Returns:
        The threshold; 0.5 if the labels contain a single class.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = skm.precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def tuned_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute the tuned-threshold variant of the headline metrics on the test portion.

    Args:
        y_true: True 0/1 test labels.
        y_score: Test scores.
        threshold: Threshold tuned on the train portion.

    Returns:
        ``threshold_tuned`` plus ``*_tuned`` precision, recall, F1, MCC and balanced accuracy.
    """
    y_pred = (y_score >= threshold).astype(np.int8)
    return {
        "threshold_tuned": float(threshold),
        "precision_tuned": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall_tuned": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1_tuned": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "mcc_tuned": float(_safe(skm.matthews_corrcoef, y_true, y_pred)),
        "balanced_accuracy_tuned": float(_safe(skm.balanced_accuracy_score, y_true, y_pred)),
    }


def _safe(func: Any, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Call a metric, returning 0.0 instead of warnings/errors on degenerate inputs.

    Args:
        func: scikit-learn metric function.
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        The metric value, or 0.0 when it is undefined.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = float(func(y_true, y_pred))
        except ValueError:
            return 0.0
    return 0.0 if np.isnan(value) else value


def _ratio(numerator: int, denominator: int) -> float:
    """Divide, returning 0.0 for a zero denominator.

    Args:
        numerator: Top of the fraction.
        denominator: Bottom of the fraction.

    Returns:
        The ratio.
    """
    return float(numerator) / denominator if denominator else 0.0


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to plain Python numbers for JSON.

    Args:
        value: Any value.

    Returns:
        A JSON-serialisable value.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
