"""Which words make up the score gap between women's and men's bios of one occupation.

Work on the bios whose true occupation is `o`, with the target of the Shapley values equal to
`o`. A word is the `data.word_core` of a unit (a unit with no letter or digit keeps its text);
a bio's value for a word is the sum over its occurrences, and a word absent from a bio counts 0.

- `I(w, o)`, word importance: mean value over the bios of `o` that contain `w`; reported only
  for words in at least `MIN_BIOS` bios of `o`.
- `D(w, o)`, gap contribution: mean over women's bios minus mean over men's bios, absent = 0.
- Efficiency makes the contributions add up: the sum of D over all words is the women's mean
  minus the men's mean of `score(full bio) - score(empty bio)`.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from hiring_bias_xai import data
from hiring_bias_xai.attribution import Attribution
from hiring_bias_xai.config import SEED
from hiring_bias_xai.models import log_odds

MIN_BIOS = 20
TOP = 10
N_BOOT = 1000
COLUMNS = [
    "word",
    "n_bios",
    "importance",
    "contribution",
    "contribution_ci_low",
    "contribution_ci_high",
    "freq_female",
    "freq_male",
    "value_female",
    "value_male",
    "usage",
    "weight",
    "gendered",
    "top",
]


class Decomposition(NamedTuple):
    table: pd.DataFrame  # one row per reported word, largest |contribution| first
    summary: dict


class Bio(NamedTuple):
    gender: str
    values: dict[str, float]  # Shapley value of each word for the target, summed over occurrences
    score: float  # log-odds score of the full bio
    empty: float  # log-odds score of the empty bio


def _matrices(bios: list[Bio], vocabulary: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Value of each word (column) in each bio (row), 0 where absent, and whether it is present.

    A word can be present with value exactly 0, so presence is tracked on its own.
    """
    column = {word: i for i, word in enumerate(vocabulary)}
    values = np.zeros((len(bios), len(vocabulary)))
    present = np.zeros(values.shape, dtype=bool)
    for row, bio in enumerate(bios):
        for word, value in bio.values.items():
            values[row, column[word]] = value
            present[row, column[word]] = True
    return values, present


def _resampled_means(matrix: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Mean over the rows of `matrix` in `n_boot` resamples of its rows with replacement."""
    n = len(matrix)
    drawn = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n))
    np.add.at(counts, (np.arange(n_boot)[:, None], drawn), 1)
    return counts @ matrix / n


def _bootstrap(values: np.ndarray, full: np.ndarray, female: np.ndarray, n_boot: int, seed):
    """Per resample, the contribution of every word and the score gap of the full bios.

    Each gender is resampled on its own, with replacement; the generator draws women first.
    """
    rng = np.random.default_rng(seed)
    together = np.column_stack([values, full])
    women = _resampled_means(together[female], n_boot, rng)
    men = _resampled_means(together[~female], n_boot, rng)
    difference = women - men
    return difference[:, :-1], difference[:, -1]


def _interval(draws: np.ndarray) -> dict:
    """The 2.5th and 97.5th percentiles, ignoring the draws that are not a number."""
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high)}


def _share(part, total):
    """`part / total`, NaN where the total is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total == 0, np.nan, part / total)


def _value(number) -> float | None:
    """A float for JSON: None for NaN."""
    return None if np.isnan(number) else float(number)


def _top_share(contribution: np.ndarray, top: int) -> np.ndarray:
    """Per row, the sum of the `top` largest |contribution| over the row's total."""
    top = min(top, contribution.shape[1])
    largest = np.argpartition(-np.abs(contribution), top - 1, axis=1)[:, :top]
    part = np.take_along_axis(contribution, largest, axis=1).sum(axis=1)
    return _share(part, contribution.sum(axis=1))


def decompose(
    bios: list[Bio],
    *,
    min_bios: int = MIN_BIOS,
    top: int = TOP,
    n_boot: int = N_BOOT,
    seed=SEED,
) -> Decomposition:
    """The words of one occupation with their contribution to the women-minus-men score gap.

    `bios` are all the bios of the occupation, both genders. The table has the words in at least
    `min_bios` bios and the `top` words of largest |contribution| (flagged in `top`), largest
    |contribution| first. Importance is left empty for a word in fewer than `min_bios` bios.
    """
    vocabulary = sorted({word for bio in bios for word in bio.values})
    values, present = _matrices(bios, vocabulary)
    female = np.array([bio.gender == "female" for bio in bios])
    contribution = values[female].mean(axis=0) - values[~female].mean(axis=0)
    total = contribution.sum()
    n_bios = present.sum(axis=0)
    frequent = n_bios >= min_bios
    importance = np.where(frequent, values.sum(axis=0) / n_bios, np.nan)  # n_bios >= 1
    order = np.argsort(-np.abs(contribution), kind="stable")
    is_top = np.zeros(len(vocabulary), dtype=bool)
    is_top[order[:top]] = True
    columns = {
        "word": vocabulary,
        "n_bios": n_bios,
        "importance": importance,
        "contribution": contribution,
        "top": is_top,
    }
    for gender, rows in (("female", female), ("male", ~female)):
        count = present[rows].sum(axis=0)
        columns[f"freq_{gender}"] = count / rows.sum()
        with np.errstate(invalid="ignore"):  # a word absent from the group has no mean value
            columns[f"value_{gender}"] = np.where(
                count > 0, values[rows].sum(axis=0) / count, np.nan
            )
    # D = f_F m_F - f_M m_M = (f_F - f_M) mean(m) + mean(f) (m_F - m_M); a missing m is 0 here,
    # because its frequency is 0 too.
    f_female, f_male = columns["freq_female"], columns["freq_male"]
    m_female, m_male = np.nan_to_num(columns["value_female"]), np.nan_to_num(columns["value_male"])
    columns["usage"] = (f_female - f_male) * (m_female + m_male) / 2
    columns["weight"] = (f_female + f_male) / 2 * (m_female - m_male)
    gendered = np.array([data.is_gendered_word(word) for word in vocabulary], dtype=bool)
    columns["gendered"] = gendered
    full = np.array([bio.score for bio in bios])
    empty = np.array([bio.empty for bio in bios])
    score_gap = full[female].mean() - full[~female].mean()
    exact = (full - empty)[female].mean() - (full - empty)[~female].mean()
    boot_contribution, boot_score_gap = _bootstrap(values, full, female, n_boot, seed)
    low, high = np.percentile(boot_contribution, [2.5, 97.5], axis=0)
    columns["contribution_ci_low"], columns["contribution_ci_high"] = low, high
    table = pd.DataFrame(columns)[COLUMNS]
    kept = table.iloc[order][frequent[order] | is_top[order]].reset_index(drop=True)
    boot_total = boot_contribution.sum(axis=1)
    decomposed_gap = {"value": float(total), **_interval(boot_total)}
    fixed = _interval(_share(boot_contribution[:, is_top].sum(axis=1), boot_total))
    summary = {
        "n_female": int(female.sum()),
        "n_male": int((~female).sum()),
        "score_gap": {"value": float(score_gap), **_interval(boot_score_gap)},
        "decomposed_gap": decomposed_gap,
        "gap_unstable": bool(decomposed_gap["ci_low"] <= 0 <= decomposed_gap["ci_high"]),
        "top_share": {
            "value": _value(_share(contribution[is_top].sum(), total)),
            **_interval(_top_share(boot_contribution, top)),
            **{f"fixed_{key}": value for key, value in fixed.items()},
        },
        "top_words": [vocabulary[i] for i in order[:top]],
        "gendered_share": {
            "value": _value(_share(contribution[gendered].sum(), total)),
            **_interval(_share(boot_contribution[:, gendered].sum(axis=1), boot_total)),
        },
        "empty_score_drift": float(score_gap - total),
        "decomposition_error": float(abs(total - exact)),
    }
    return Decomposition(kept, summary)


def bio_from_attribution(saved: Attribution, gender: str) -> Bio:
    """The per-word values of one stored attribution, with its full and empty scores."""
    values: dict[str, float] = {}
    for word, value in zip(saved.words, saved.values, strict=True):
        key = data.word_core(word) or word
        values[key] = values.get(key, 0.0) + float(value)
    scores = log_odds(saved.logits, saved.target)[saved.index]
    return Bio(gender, values, float(scores[0, -1]), float(scores[0, 0]))
