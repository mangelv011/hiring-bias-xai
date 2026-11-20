"""Effects of a counterfactual edit on a screener: flip rate, log-odds change, flip direction."""

from collections.abc import Mapping, Sequence

import numpy as np

from hiring_bias_xai.config import SEED

from hiring_bias_xai.config import SEED


def moves_toward_new_gender(
    gender: Sequence[str],
    before: Sequence[str],
    after: Sequence[str],
    share_women: Mapping[str, float],
) -> np.ndarray:
    """Per bio, whether the new prediction has a share of women that moves toward the new gender.

    `gender` is the bio's label before the swap, so the new gender is the other one. Toward the
    male gender means a lower share of women than the old prediction, toward the female gender a
    higher one; equal shares are not toward. `share_women` is the training share (the one F1
    uses); a prediction that is not in it raises KeyError.
    """
    change = np.array(
        [share_women[new] - share_women[old] for old, new in zip(before, after, strict=True)]
    )
    to_male = np.asarray(gender) == "female"
    return np.where(to_male, change < 0, change > 0)


N_BOOT = 1000


def _interval(resampled: np.ndarray) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles, ignoring the resamples that gave NaN; NaN if all did."""
    if np.isnan(resampled).all():
        return float("nan"), float("nan")
    low, high = np.nanpercentile(resampled, [2.5, 97.5])
    return float(low), float(high)


def summarise(
    flip: np.ndarray,
    change: np.ndarray,
    toward: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Flip rate, mean log-odds change and share of flips toward the new gender, per bio.

    `flip` says whether the prediction changed, `change` is the log-odds of the true occupation
    after the edit minus before, `toward` (swap only) whether the new prediction moves toward the
    new gender. Each value comes with the 95% percentile interval of `n_boot` bootstrap
    resamples of the bios (`_ci_low`, `_ci_high`). The toward share is over the flips and NaN,
    like its interval, when there are none or no `toward` is given.
    """
    flip = np.asarray(flip, dtype=bool)
    change = np.asarray(change, dtype=float)
    n = len(flip)
    drawn = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    summary = {"n": n, "n_flips": int(flip.sum())}
    for name, value, resampled in (
        ("flip_rate", flip.mean(), flip[drawn].mean(axis=1)),
        ("mean_log_odds_change", change.mean(), change[drawn].mean(axis=1)),
    ):
        low, high = _interval(resampled)
        summary |= {name: float(value), f"{name}_ci_low": low, f"{name}_ci_high": high}
    value, low, high = np.nan, np.nan, np.nan
    if toward is not None and flip.any():
        moved = flip & np.asarray(toward, dtype=bool)
        value = moved.sum() / flip.sum()
        with np.errstate(invalid="ignore"):
            low, high = _interval(moved[drawn].sum(axis=1) / flip[drawn].sum(axis=1))
    return summary | {"toward_share": value, "toward_share_ci_low": low, "toward_share_ci_high": high}
