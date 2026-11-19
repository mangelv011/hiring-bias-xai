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
