"""Activation patching: how much of the effect of a gender swap one layer's states restore."""

import numpy as np

from hiring_bias_xai.config import SEED
from hiring_bias_xai.models import log_odds

N_BOOT = 1000


def restored_share(clean: np.ndarray, corrupted: np.ndarray, patched: np.ndarray) -> np.ndarray:
    """Per bio, (patched - corrupted) / (clean - corrupted) of the true-class log-odds.

    1 means the patch brings the corrupted run back to the clean one, 0 that it changes nothing;
    a patch can overshoot, so the share can leave [0, 1]. Raises ValueError if a bio has the same
    clean and corrupted value, where the share is undefined.
    """
    effect = np.asarray(clean) - np.asarray(corrupted)
    if (effect == 0).any():
        raise ValueError("clean and corrupted log-odds must differ in every bio")
    return (np.asarray(patched) - np.asarray(corrupted)) / effect


def eligible(
    swap_change: np.ndarray, removal_change: np.ndarray, quantile: float = 0.95
) -> tuple[np.ndarray, float]:
    """Which bios a swap moves by more than the removal control moves any bio, and the bar.

    The bar is the `quantile` of |`removal_change`| over all bios; a bio is eligible when its
    |`swap_change`| is above it. Both changes are in the true-class log-odds.
    """
    bar = float(np.quantile(np.abs(removal_change), quantile))
    return np.abs(swap_change) > bar, bar


def summarise(shares: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean and median of the restored shares of the bios, each with a 95% bootstrap interval.

    The interval is the 2.5th and 97.5th percentile over `n_boot` resamples of the bios.
    """
    shares = np.asarray(shares, dtype=float)
    n = len(shares)
    drawn = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    summary = {"n": n}
    for name, function in (("mean", np.mean), ("median", np.median)):
        low, high = np.percentile(function(shares[drawn], axis=1), [2.5, 97.5])
        summary |= {
            name: float(function(shares)),
            f"{name}_ci_low": float(low),
            f"{name}_ci_high": float(high),
        }
    return summary


def true_class_log_odds(logits: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per row of `logits`, the log-odds of the column in `index` (one column per row)."""
    index = np.asarray(index)
    result = np.empty(len(index))
    for column in np.unique(index):
        rows = index == column
        result[rows] = log_odds(logits[rows], column)
    return result
