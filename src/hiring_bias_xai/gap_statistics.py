"""Statistics behind the gap-share correlation and the per-occupation TPR gaps.

`fairness.evaluate` bootstraps bios; here the units are the 28 occupations (correlations) or the
bios within an occupation (permutation null, per-occupation tests).
"""

import numpy as np
from scipy import stats

from hiring_bias_xai.config import SEED

Z_95 = stats.norm.ppf(0.975)


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    """95% interval of a Pearson correlation `r` over `n` points, by the Fisher z transform."""
    with np.errstate(divide="ignore"):
        z = np.arctanh(r)
    half_width = Z_95 / np.sqrt(n - 3)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def holm(p_values) -> list[float]:
    """Holm-adjusted p-values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = (len(p) - np.arange(len(p))) * p[order]
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return adjusted.tolist()


def two_proportion_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided p of the pooled two-proportion z-test; 1 when the pooled rate is 0 or 1."""
    pooled = (hits_a + hits_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (hits_a / n_a - hits_b / n_b) / se
    return float(2 * stats.norm.sf(abs(z)))


def _by_occupation(y_true, y_pred, gender):
    """Occupation names (sorted) and, for each, its bios' hit flags and female flags."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    female = np.asarray(gender) == "female"
    names = np.unique(y_true)
    cells = [(y_true == name) for name in names]
    return names.tolist(), [(y_pred[c] == name, female[c]) for name, c in zip(names, cells, strict=True)]


def _rms(cells) -> float:
    gaps = [hit[female].mean() - hit[~female].mean() for hit, female in cells]
    return float(np.sqrt(np.mean(np.square(gaps))))


def rms_permutation_test(y_true, y_pred, gender, *, n_perm=1000, seed=SEED) -> dict:
    """RMS TPR gap against a null where gender carries no information about the hits.

    The null shuffles the gender labels among the bios of each occupation and keeps the
    predictions fixed. `p_value` is (1 + permutations with an RMS gap at least the observed) /
    (1 + n_perm).
    """
    _, cells = _by_occupation(y_true, y_pred, gender)
    observed = _rms(cells)
    rng = np.random.default_rng(seed)
    null = np.array(
        [_rms([(hit, rng.permutation(female)) for hit, female in cells]) for _ in range(n_perm)]
    )
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": float((1 + np.sum(null >= observed)) / (1 + n_perm)),
        "n_permutations": n_perm,
    }


def occupation_tests(y_true, y_pred, gender) -> dict:
    """Per occupation: TPR gap, raw p of the women-versus-men z-test and its Holm-adjusted p.

    Holm runs over all occupations. The gaps are descriptive; a claim that one occupation's gap
    is non-zero rests on `p_holm` only.
    """
    names, cells = _by_occupation(y_true, y_pred, gender)
    rows = []
    for hit, female in cells:
        hits_f, n_f = int(hit[female].sum()), int(female.sum())
        hits_m, n_m = int(hit[~female].sum()), int((~female).sum())
        rows.append((hits_f / n_f - hits_m / n_m, two_proportion_p(hits_f, n_f, hits_m, n_m)))
    adjusted = holm([p for _, p in rows])
    return {
        name: {"tpr_gap": gap, "p_raw": p, "p_holm": p_holm}
        for name, (gap, p), p_holm in zip(names, rows, adjusted, strict=True)
    }


def _pearson_rows(x, y):
    """Pearson correlation of each row of `x` with the same row of `y`; NaN for a constant row."""
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    with np.errstate(all="ignore"):
        return (x * y).sum(axis=-1) / np.sqrt((x**2).sum(axis=-1) * (y**2).sum(axis=-1))


def _spearman_rows(x, y):
    return _pearson_rows(stats.rankdata(x, axis=-1), stats.rankdata(y, axis=-1))


def correlations(gap, share, *, n_boot=1000, seed=SEED) -> dict:
    """Pearson and Spearman correlation of the TPR gaps with the shares of women.

    Pearson has a Fisher-z interval and a bootstrap interval; Spearman has a bootstrap one. The
    bootstrap resamples the occupations (the (gap, share) points) with replacement and reads the
    2.5th and 97.5th percentiles; a resample whose points all share one value is ignored.
    """
    gap, share = np.asarray(gap, dtype=float), np.asarray(share, dtype=float)
    n = len(gap)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    pearson, spearman = _pearson_rows(gap, share), _spearman_rows(gap, share)
    fisher_low, fisher_high = fisher_ci(pearson, n)
    boot_pearson = np.nanpercentile(_pearson_rows(gap[idx], share[idx]), [2.5, 97.5])
    boot_spearman = np.nanpercentile(_spearman_rows(gap[idx], share[idx]), [2.5, 97.5])
    return {
        "pearson": {
            "value": float(pearson),
            "fisher_ci_low": fisher_low,
            "fisher_ci_high": fisher_high,
            "bootstrap_ci_low": float(boot_pearson[0]),
            "bootstrap_ci_high": float(boot_pearson[1]),
        },
        "spearman": {
            "value": float(spearman),
            "ci_low": float(boot_spearman[0]),
            "ci_high": float(boot_spearman[1]),
        },
    }


def leave_one_out(gap, share, names) -> dict:
    """Lowest and highest Pearson r when one occupation is left out, and which one it was."""
    gap, share = np.asarray(gap, dtype=float), np.asarray(share, dtype=float)
    n = len(gap)
    rows = np.array([np.delete(np.arange(n), i) for i in range(n)])
    r = _pearson_rows(gap[rows], share[rows])
    lowest, highest = int(np.argmin(r)), int(np.argmax(r))
    return {
        "min": {"r": float(r[lowest]), "left_out": names[lowest]},
        "max": {"r": float(r[highest]), "left_out": names[highest]},
    }
