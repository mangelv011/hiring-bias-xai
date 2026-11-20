"""Checks of the Shapley explanations: deletion curves, seed and removal stability, leave-one-out.

A deletion curve removes words from a bio in an order and follows the main model's (or the
baseline model's) call as more words go. The words are removed with `unk`, the removal that the
attributions did not use, so the check is not circular for the main model.
"""

from typing import NamedTuple

import numpy as np


def words_to_remove(n_words: int, percent: int) -> int:
    """How many of `n_words` words go at `percent` percent: the share, rounded half up."""
    return (percent * n_words + 50) // 100


class Orders(NamedTuple):
    """The orders in which the words of one bio are removed, as word indices."""

    morf: np.ndarray
    lerf: np.ndarray
    random: np.ndarray


def removal_orders(values: np.ndarray, n_random: int, rng: np.random.Generator) -> Orders:
    """MoRF (largest value first), LeRF (smallest first) and `n_random` random orders.

    Ties go to the word that comes first in the bio.
    """
    n_words = len(values)
    return Orders(
        morf=np.argsort(-values, kind="stable"),
        lerf=np.argsort(values, kind="stable"),
        random=np.stack([rng.permutation(n_words) for _ in range(n_random)]),
    )
