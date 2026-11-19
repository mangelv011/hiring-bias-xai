"""Statistics behind the gap-share correlation and the per-occupation TPR gaps.

`fairness.evaluate` bootstraps bios; here the units are the 28 occupations (correlations) or the
bios within an occupation (permutation null, per-occupation tests).
"""

import numpy as np
from scipy import stats

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
