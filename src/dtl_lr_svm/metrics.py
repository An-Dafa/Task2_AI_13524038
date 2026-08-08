"""Evaluation utilities for binary classification.

The competition metric is macro F1.  The functions in this module are written
with NumPy so the same evaluator can be used for both from-scratch models and
scikit-learn baselines without hiding the metric calculation behind a model
library.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _as_binary_vector(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} tidak boleh kosong.")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} hanya boleh berisi label biner 0 dan 1.")
    return array.astype(np.int64, copy=False)


def binary_confusion_counts(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
) -> dict[str, int]:
    """Return TN, FP, FN, and TP counts for labels {0, 1}."""
    true = _as_binary_vector(y_true, "y_true")
    pred = _as_binary_vector(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise ValueError("y_true dan y_pred harus memiliki panjang yang sama.")

    return {
        "tn": int(np.sum((true == 0) & (pred == 0))),
        "fp": int(np.sum((true == 0) & (pred == 1))),
        "fn": int(np.sum((true == 1) & (pred == 0))),
        "tp": int(np.sum((true == 1) & (pred == 1))),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def binary_classification_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
) -> dict[str, float | int | list[list[int]]]:
    """Calculate accuracy, per-class metrics, macro F1, and confusion matrix.

    The matrix layout follows scikit-learn: rows are actual classes and columns
    are predicted classes, i.e. ``[[TN, FP], [FN, TP]]``.
    """
    counts = binary_confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = (counts[key] for key in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp

    precision_0 = _safe_divide(tn, tn + fn)
    recall_0 = _safe_divide(tn, tn + fp)
    f1_0 = _safe_divide(2 * precision_0 * recall_0, precision_0 + recall_0)

    precision_1 = _safe_divide(tp, tp + fp)
    recall_1 = _safe_divide(tp, tp + fn)
    f1_1 = _safe_divide(2 * precision_1 * recall_1, precision_1 + recall_1)

    accuracy = _safe_divide(tn + tp, total)
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    return {
        **counts,
        "support_0": int(tn + fp),
        "support_1": int(fn + tp),
        "accuracy": accuracy,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": macro_recall,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_search(
    y_true: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    thresholds: Iterable[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate macro F1 over candidate probability/decision thresholds.

    This helper is intentionally model-agnostic.  For Logistic Regression,
    ``scores`` can be probabilities.  For linear SVM, ``scores`` can be decision
    values and the threshold grid may include negative values.
    """
    true = _as_binary_vector(y_true, "y_true")
    score_array = np.asarray(scores, dtype=float).reshape(-1)
    if true.shape != score_array.shape:
        raise ValueError("y_true dan scores harus memiliki panjang yang sama.")
    if not np.isfinite(score_array).all():
        raise ValueError("scores mengandung NaN atau nilai tak hingga.")

    if thresholds is None:
        thresholds_array = np.linspace(0.05, 0.95, 91)
    else:
        thresholds_array = np.asarray(list(thresholds), dtype=float)
    if thresholds_array.size == 0:
        raise ValueError("Daftar threshold tidak boleh kosong.")

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds_array:
        predictions = (score_array >= threshold).astype(np.int64)
        metrics = binary_classification_metrics(true, predictions)
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(metrics["macro_f1"]),
                "f1_0": float(metrics["f1_0"]),
                "f1_1": float(metrics["f1_1"]),
                "precision_1": float(metrics["precision_1"]),
                "recall_1": float(metrics["recall_1"]),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "f1_1", "threshold"], ascending=[False, False, True]
    ).reset_index(drop=True)
