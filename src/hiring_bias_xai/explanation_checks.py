"""Checks of the Shapley explanations: deletion curves, seed and removal stability, leave-one-out.

A deletion curve removes words from a bio in an order and follows the main model's (or the
baseline model's) call as more words go. The words are removed with `unk`, the removal that the
attributions did not use, so the check is not circular for the main model.
"""

import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hiring_bias_xai.attribution import remove_words
from hiring_bias_xai.config import SEED
from hiring_bias_xai.data import word_core
from hiring_bias_xai.models import Classifier, log_odds


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


class Curve(NamedTuple):
    """One bio's deletion curve. Rows: MoRF, LeRF, then the random orders; columns: the steps.

    `agreement` is 1 where the call is still the call on the full bio, 0 where it changed;
    `drop` is the fall in the log-odds of the target occupation from the full bio.
    """

    agreement: np.ndarray
    drop: np.ndarray


def deletion_curve(
    classifier: Classifier,
    words: Sequence[str],
    values: np.ndarray,
    target: int,
    percents: Sequence[int],
    n_random: int,
    rng: np.random.Generator,
) -> Curve:
    """The classifier's call and score as the words of a bio go, with `unk` removal.

    At each step in `percents`, `words_to_remove` words are replaced by `[UNK]`, in the order of
    MoRF (by `values`), LeRF and `n_random` random orders. All texts go to the classifier in one
    call, each distinct text once. `target` is the column of the occupation whose log-odds fall.
    """
    n_words = len(words)
    orders = removal_orders(values, n_random, rng)
    all_orders = [orders.morf, orders.lerf, *orders.random]
    full = " ".join(words)
    row_of_text = {full: 0}
    rows = np.empty((len(all_orders), len(percents)), dtype=int)
    for i, order in enumerate(all_orders):
        for j, percent in enumerate(percents):
            removed = np.zeros(n_words, dtype=bool)
            removed[order[: words_to_remove(n_words, percent)]] = True
            text = remove_words(words, removed, "unk")
            rows[i, j] = row_of_text.setdefault(text, len(row_of_text))
    logits = classifier.logits(list(row_of_text))
    score = log_odds(logits, target)
    return Curve(
        agreement=(logits.argmax(axis=1)[rows] == logits[0].argmax()).astype(float),
        drop=score[0] - score[rows],
    )


N_BOOT = 1000


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float | np.ndarray], n: int, n_boot: int = N_BOOT, seed=SEED
):
    """The 95% bootstrap interval of `statistic`, resampling `n` bios with replacement.

    `statistic` gets the resampled bio indices and returns a number or an array; the bounds are
    the 2.5th and 97.5th percentiles, per entry. A resample whose statistic is NaN is ignored.
    """
    rng = np.random.default_rng(seed)
    resampled = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_boot)])
    low, high = np.nanpercentile(resampled, [2.5, 97.5], axis=0)
    return low, high


TOP = 10


def rank_agreement(one: np.ndarray, other: np.ndarray, top: int = TOP) -> tuple[float, float]:
    """Spearman correlation over all words, and top-`top` overlap, of two value vectors.

    The overlap is the share of the `top` words with the largest absolute value in `one` (all the
    words, if the bio has fewer) that are also among the largest in `other`; ties go to the word
    that comes first. The correlation is NaN when it is undefined: under two words, or all
    values equal in one vector.
    """
    shown = min(top, len(one))
    largest_one = set(np.argsort(-np.abs(one), kind="stable")[:shown])
    largest_other = set(np.argsort(-np.abs(other), kind="stable")[:shown])
    overlap = len(largest_one & largest_other) / shown
    if len(one) < 2 or np.ptp(one) == 0 or np.ptp(other) == 0:
        return float("nan"), overlap
    return float(spearmanr(one, other).statistic), overlap


def drop_without(
    classifier: Classifier, words: Sequence[str], groups: Sequence[Sequence[int]], target: int
) -> np.ndarray:
    """For each group of word indices, the log-odds of the full bio minus the log-odds of the bio
    with that group deleted. The full bio and all the groups go to the classifier in one call."""
    texts = [" ".join(words)]
    for group in groups:
        removed = np.zeros(len(words), dtype=bool)
        removed[list(group)] = True
        texts.append(remove_words(words, removed, "delete"))
    score = log_odds(classifier.logits(texts), target)
    return score[0] - score[1:]


def loo_values(classifier: Classifier, words: Sequence[str], target: int) -> np.ndarray:
    """Leave-one-out occlusion: the value of each word is the fall in the log-odds of the target
    occupation when that word alone is deleted."""
    return drop_without(classifier, words, [[i] for i in range(len(words))], target)


def stratified_sample(subset: pd.DataFrame, n: int, seed) -> np.ndarray:
    """Positions in `subset` of `n` bios: equal shares from each occupation and gender group.

    The groups are visited in sorted (occupation, gender) order. Each gets `n // groups` bios,
    and the first `n % groups` groups one more. Each draws without replacement from one
    generator seeded with `seed`. A group too small for its share raises, so the sample never
    silently has fewer than `n` bios. The positions are sorted.
    """
    groups = subset.groupby(["occupation", "gender"]).indices
    base, extra = divmod(n, len(groups))
    rng = np.random.default_rng(seed)
    chosen = []
    for rank, (name, positions) in enumerate(sorted(groups.items())):
        size = base + (rank < extra)
        if len(positions) < size:
            raise ValueError(f"{name} has {len(positions)} bios, fewer than its share of {size}")
        chosen.append(rng.choice(positions, size=size, replace=False))
    return np.sort(np.concatenate(chosen))


MIN_UNITS = 15
IDENTIFIER = re.compile(r"[0-9]{6,}")


class Case(NamedTuple):
    """One bio's Shapley and leave-one-out values for its true occupation."""

    bio_id: str
    words: list[str]
    shapley: np.ndarray
    loo: np.ndarray


def _eligible(case: Case) -> bool:
    """At least `MIN_UNITS` word units, no run of six digits (it may be an identifier), and a top
    Shapley word (largest absolute value, first on ties) that occurs at least twice, matched
    without case and without punctuation at its ends."""
    if len(case.words) < MIN_UNITS or IDENTIFIER.search(" ".join(case.words)):
        return False
    cores = [word_core(word) or word for word in case.words]
    top = int(np.argmax(np.abs(case.shapley)))
    return cores.count(cores[top]) >= 2


def disagreement_example(cases: Sequence[Case]) -> tuple[Case, float] | None:
    """The eligible bio with the lowest Spearman correlation between its Shapley and
    leave-one-out values, with that correlation; the first such bio on a tie. None if no bio
    qualifies or every correlation is undefined."""
    scored = [
        (rank_agreement(case.shapley, case.loo)[0], case) for case in cases if _eligible(case)
    ]
    scored = [(spearman, case) for spearman, case in scored if not np.isnan(spearman)]
    if not scored:
        return None
    spearman, case = min(scored, key=lambda pair: pair[0])  # min keeps the first on a tie
    return case, spearman
