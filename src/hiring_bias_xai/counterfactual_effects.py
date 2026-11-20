"""Effects of a counterfactual edit on a screener: flip rate, log-odds change, flip direction."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

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
DIRECTION_BASIS = "training share of women, the share F1 uses (not independent of F1)"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson score interval of a share of `successes` in `n` trials."""
    share = successes / n
    scale = 1 + z**2 / n
    centre = (share + z**2 / (2 * n)) / scale
    half = z * np.sqrt(share * (1 - share) / n + z**2 / (4 * n**2)) / scale
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


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
    return summary | {
        "toward_share": value,
        "toward_share_ci_low": low,
        "toward_share_ci_high": high,
    }


def hand_check_sample(ids: Sequence[str], n: int, seed: int = SEED) -> list[str]:
    """`n` distinct ids drawn at random from `ids`, in the order they have in `ids`."""
    if len(ids) < n:
        raise ValueError(f"cannot draw {n} bios from only {len(ids)}")
    chosen = np.random.default_rng(seed).choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(chosen)]


def hand_check_summary(checks: pd.DataFrame) -> dict:
    """The incoherent share of a hand check with its Wilson interval.

    `checks` has one row per bio: `coherent` ("yes" or "no") and, for every "no", a `reason`.
    """
    if not checks["coherent"].isin(["yes", "no"]).all():
        raise ValueError("coherent must be yes or no for every bio")
    incoherent = checks["coherent"] == "no"
    if (checks.loc[incoherent, "reason"].fillna("").str.strip() == "").any():
        raise ValueError("every incoherent bio needs a reason")
    n, k = len(checks), int(incoherent.sum())
    low, high = wilson_interval(k, n)
    return {"n": n, "n_incoherent": k, "incoherent_share": k / n, "ci_low": low, "ci_high": high}


def table(
    scored: pd.DataFrame, share_women: Mapping[str, float], screener: str, n_boot: int = N_BOOT
) -> pd.DataFrame:
    """One row per occupation (and "pooled"), subset (all, unflagged) and condition (swap, removal).

    `scored` has one row per bio: occupation, gender, flagged, the predictions `original`,
    `swapped` and `removed`, and the log-odds of the true occupation `lo_original`, `lo_swapped`
    and `lo_removed`.
    """
    rows = []
    for occupation in (*sorted(scored["occupation"].unique()), "pooled"):
        group = scored if occupation == "pooled" else scored[scored["occupation"] == occupation]
        for subset, bios in (("all", group), ("unflagged", group[~group["flagged"]])):
            for condition, edited in (("swap", "swapped"), ("removal", "removed")):
                flip = (bios[edited] != bios["original"]).to_numpy()
                change = (bios[f"lo_{edited}"] - bios["lo_original"]).to_numpy()
                swap = condition == "swap"
                toward = (
                    moves_toward_new_gender(
                        bios["gender"], bios["original"], bios[edited], share_women
                    )
                    if swap
                    else None
                )
                row = {"screener": screener, "occupation": occupation, "subset": subset}
                row |= {"condition": condition} | summarise(flip, change, toward, n_boot=n_boot)
                rows.append(row | {"direction_basis": DIRECTION_BASIS if swap else ""})
    return pd.DataFrame(rows)
