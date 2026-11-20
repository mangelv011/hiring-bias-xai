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
from hiring_bias_xai.models import log_odds


MIN_BIOS = 20
TOP = 10


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


def decompose(bios: list[Bio], *, min_bios: int = MIN_BIOS, top: int = TOP) -> Decomposition:
    """The words of one occupation with their contribution to the women-minus-men score gap.

    `bios` are all the bios of the occupation, both genders. The table has the words in at least
    `min_bios` bios and the `top` words of largest |contribution| (flagged in `top`), largest
    |contribution| first. Importance is left empty for a word in fewer than `min_bios` bios.
    """
    vocabulary = sorted({word for bio in bios for word in bio.values})
    values, present = _matrices(bios, vocabulary)
    female = np.array([bio.gender == "female" for bio in bios])
    contribution = values[female].mean(axis=0) - values[~female].mean(axis=0)
    n_bios = present.sum(axis=0)
    frequent = n_bios >= min_bios
    importance = np.where(frequent, values.sum(axis=0) / n_bios, np.nan)  # n_bios >= 1
    order = np.argsort(-np.abs(contribution), kind="stable")
    is_top = np.zeros(len(vocabulary), dtype=bool)
    is_top[order[:top]] = True
    table = pd.DataFrame(
        {
            "word": vocabulary,
            "n_bios": n_bios,
            "importance": importance,
            "contribution": contribution,
            "top": is_top,
        }
    )
    kept = table.iloc[order][frequent[order] | is_top[order]].reset_index(drop=True)
    full = np.array([bio.score for bio in bios])
    empty = np.array([bio.empty for bio in bios])
    score_gap = full[female].mean() - full[~female].mean()
    exact = (full - empty)[female].mean() - (full - empty)[~female].mean()
    summary = {
        "n_female": int(female.sum()),
        "n_male": int((~female).sum()),
        "score_gap": {"value": float(score_gap)},
        "decomposed_gap": {"value": float(contribution.sum())},
        "empty_score_drift": float(score_gap - contribution.sum()),
        "decomposition_error": float(abs(contribution.sum() - exact)),
    }
    return Decomposition(kept, summary)


def _key(word: str) -> str:
    return data.word_core(word) or word


def bio_from_attribution(saved: Attribution, gender: str) -> Bio:
    """The per-word values of one stored attribution, with its full and empty scores."""
    values: dict[str, float] = {}
    for word, value in zip(saved.words, saved.values, strict=True):
        key = _key(word)
        values[key] = values.get(key, 0.0) + float(value)
    scores = log_odds(saved.logits, saved.target)[saved.index]
    return Bio(gender, values, float(scores[0, -1]), float(scores[0, 0]))
