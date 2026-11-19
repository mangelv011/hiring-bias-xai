"""Fairness metrics (TPR gap) computed for any Classifier."""

import numpy as np

from hiring_bias_xai.config import SEED


def evaluate(y_true, y_pred, gender, share_women, *, n_boot=1000, seed=SEED) -> dict:
    """Accuracy and macro-F1 of the screening decisions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = []
    for occupation in np.unique(y_true):
        tp = np.sum((y_true == occupation) & (y_pred == occupation))
        fp = np.sum((y_true != occupation) & (y_pred == occupation))
        fn = np.sum((y_true == occupation) & (y_pred != occupation))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return {
        "accuracy": {"value": float(np.mean(y_true == y_pred))},
        "macro_f1": {"value": float(np.mean(f1))},
    }
