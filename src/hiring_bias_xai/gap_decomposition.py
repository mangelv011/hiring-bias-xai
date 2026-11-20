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

from hiring_bias_xai import data
from hiring_bias_xai.attribution import Attribution
from hiring_bias_xai.models import log_odds


class Bio(NamedTuple):
    gender: str
    values: dict[str, float]  # Shapley value of each word for the target, summed over occurrences
    score: float  # log-odds score of the full bio
    empty: float  # log-odds score of the empty bio


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
