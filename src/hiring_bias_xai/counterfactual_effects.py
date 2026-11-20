"""Effects of a counterfactual edit on a screener: flip rate, log-odds change, flip direction."""

from collections.abc import Mapping, Sequence

import numpy as np


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
    change = np.array([share_women[new] - share_women[old] for old, new in zip(before, after, strict=True)])
    to_male = np.asarray(gender) == "female"
    return np.where(to_male, change < 0, change > 0)
