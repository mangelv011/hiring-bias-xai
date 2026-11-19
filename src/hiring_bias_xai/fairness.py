"""Fairness metrics (TPR gap) computed for any Classifier."""

import numpy as np

from hiring_bias_xai.config import SEED


def evaluate(y_true, y_pred, gender, share_women, *, n_boot=1000, seed=SEED) -> dict:
    """Accuracy of the screening decisions."""
    accuracy = np.mean(np.asarray(y_true) == np.asarray(y_pred))
    return {"accuracy": {"value": float(accuracy)}}
