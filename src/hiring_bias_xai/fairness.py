"""Fairness metrics (TPR gap) computed for any Classifier."""

import numpy as np

from hiring_bias_xai.config import SEED


def _metrics(true, pred, female, shares):
    """Metrics of the bios `true`, `pred` and `female` select, as numpy values.

    `true` and `pred` are occupation codes 0..K-1 (K = an occupation absent from `true`);
    `female` is a boolean array. An occupation x gender cell without bios gives NaN.
    """
    k = len(shares)
    hit = true == pred
    n_true = np.bincount(true, minlength=k)
    n_pred = np.bincount(pred, minlength=k + 1)[:k]
    true_positive = np.bincount(true[hit], minlength=k)
    cell = 2 * true + ~female
    n_cell = np.bincount(cell, minlength=2 * k).reshape(k, 2)
    hit_cell = np.bincount(cell[hit], minlength=2 * k).reshape(k, 2)
    with np.errstate(all="ignore"):
        f1 = np.where(n_true > 0, 2 * true_positive / (n_true + n_pred), np.nan)
        tpr = hit_cell / n_cell
        gap = tpr[:, 0] - tpr[:, 1]
        return {
            "accuracy": hit.mean(),
            "macro_f1": np.nanmean(f1),
            "tpr_female": tpr[:, 0],
            "tpr_male": tpr[:, 1],
            "tpr_gap": gap,
            "rms_tpr_gap": np.sqrt(np.mean(gap**2)),
            "gap_share_pearson": np.corrcoef(gap, shares)[0, 1],
        }


def _summary(value, resampled):
    ci_low, ci_high = np.nanpercentile(resampled, [2.5, 97.5])
    return {"value": float(value), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def evaluate(y_true, y_pred, gender, share_women, *, n_boot=1000, seed=SEED) -> dict:
    """Accuracy, macro-F1, TPR gap per occupation, RMS TPR gap and gap-share correlation.

    Every metric is `{"value", "ci_low", "ci_high"}`, a 95% percentile bootstrap interval from
    `n_boot` resamples of bios. A resample whose occupation x gender cell is empty gives NaN
    for the metrics that depend on it and is ignored for them.
    """
    y_true, y_pred, gender = np.asarray(y_true), np.asarray(y_pred), np.asarray(gender)
    occupations = np.unique(y_true)
    true = np.searchsorted(occupations, y_true)
    pred = np.searchsorted(occupations, y_pred).clip(max=len(occupations) - 1)
    pred = np.where(occupations[pred] == y_pred, pred, len(occupations))
    female = gender == "female"
    shares = np.array([share_women[occupation] for occupation in occupations])

    n = len(true)
    point = _metrics(true, pred, female, shares)
    rng = np.random.default_rng(seed)
    runs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        runs.append(_metrics(true[idx], pred[idx], female[idx], shares))
    resampled = {name: np.array([run[name] for run in runs]) for name in point}

    per_occupation = {}
    for i, occupation in enumerate(occupations):
        in_occupation = true == i
        per_occupation[str(occupation)] = {
            name: _summary(point[name][i], resampled[name][:, i])
            for name in ("tpr_female", "tpr_male", "tpr_gap")
        } | {
            "n_female": int(np.sum(in_occupation & female)),
            "n_male": int(np.sum(in_occupation & ~female)),
        }
    return {
        name: _summary(point[name], resampled[name])
        for name in ("accuracy", "macro_f1", "rms_tpr_gap", "gap_share_pearson")
    } | {"per_occupation": per_occupation}
