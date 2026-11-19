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
    gender = np.asarray(gender)
    per_occupation = {}
    for occupation in np.unique(y_true):
        tpr = {}
        for group in ("female", "male"):
            of_group = (y_true == occupation) & (gender == group)
            tpr[f"tpr_{group}"] = {"value": float(np.mean(y_pred[of_group] == occupation))}
            tpr[f"n_{group}"] = int(of_group.sum())
        tpr["tpr_gap"] = {"value": tpr["tpr_female"]["value"] - tpr["tpr_male"]["value"]}
        per_occupation[str(occupation)] = tpr
    shares = [share_women[occupation] for occupation in per_occupation]
    gaps = [tpr["tpr_gap"]["value"] for tpr in per_occupation.values()]
    return {
        "per_occupation": per_occupation,
        "gap_share_pearson": {"value": float(np.corrcoef(gaps, shares)[0, 1])},
        "rms_tpr_gap": {"value": float(np.sqrt(np.mean(np.square(gaps))))},
        "accuracy": {"value": float(np.mean(y_true == y_pred))},
        "macro_f1": {"value": float(np.mean(f1))},
    }
