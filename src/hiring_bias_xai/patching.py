"""Activation patching: how much of the effect of a gender swap one layer's states restore."""

import numpy as np


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
