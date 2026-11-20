"""F2: which words drove the main model's screening decision, for two showcase-subset bios.

Both cases come from the saved Shapley values of the main model, with values recomputed for the
occupation it predicted. The cases are picked by rules, not by hand:

- Error case: among the bios the main model gets wrong, the one where the gendered words add the
  most to the log-odds of the predicted (wrong) occupation: the largest sum of their Shapley
  values, and that sum must be above zero.
- Correct case: among the bios it gets right, the one where gendered words carry the largest share
  of the total absolute Shapley value.

A tie goes to the bio that comes first in the subset. Bios with a run of 6 or more digits are
never picked, because they may hold an identifier (a licence or a registry number). A bio is also
left out when the stored full-bio logits predict another occupation than the saved prediction
does (the stored bio is the words rejoined with single spaces).
"""

import re
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from hiring_bias_xai import attribution, data

SCREENER = "deberta"
IDENTIFIER = re.compile(r"[0-9]{6,}")


class Case(NamedTuple):
    bio_id: str
    words: list[str]
    true: str
    predicted: str
    values: np.ndarray  # Shapley value of each word for the predicted occupation


def is_gendered(word: str) -> bool:
    """Whether the word is one of the gendered words `data.scrub` removes (the list of F2)."""
    return data._GENDERED.search(word) is not None


def _gendered_mask(case: Case) -> np.ndarray:
    return np.array([is_gendered(word) for word in case.words], dtype=bool)


def _eligible(case: Case) -> bool:
    return IDENTIFIER.search(" ".join(case.words)) is None


def gendered_push(case: Case) -> float:
    """Sum of the Shapley values of the gendered words, for the predicted occupation."""
    return float(case.values[_gendered_mask(case)].sum())


def gendered_share(case: Case) -> float:
    """Share of the total absolute Shapley value that gendered words carry; NaN if it is zero."""
    total = np.abs(case.values).sum()
    return float(np.abs(case.values[_gendered_mask(case)]).sum() / total) if total else np.nan


def pick_error(cases: Sequence[Case]) -> Case:
    """The wrong call with the largest `gendered_push`, which must be above zero."""
    scored = [(gendered_push(c), c) for c in cases if c.true != c.predicted and _eligible(c)]
    scored = [(score, case) for score, case in scored if score > 0]
    if not scored:
        raise ValueError("no error has gendered words pushing towards the predicted occupation")
    return max(scored, key=lambda pair: pair[0])[1]  # max keeps the first of equal scores


def pick_correct(cases: Sequence[Case]) -> Case:
    """The right call with the largest `gendered_share`."""
    scored = [(gendered_share(c), c) for c in cases if c.true == c.predicted and _eligible(c)]
    scored = [(share, case) for share, case in scored if not np.isnan(share)]
    if not scored:
        raise ValueError("no correct call has a gendered share")
    return max(scored, key=lambda pair: pair[0])[1]


def load_cases(subset: pd.DataFrame, predicted: Sequence[str]) -> tuple[list[Case], int]:
    """A case per bio of `subset` from its saved attribution, and how many bios were skipped."""
    cases, skipped = [], 0
    for bio, label in zip(subset.itertuples(), predicted, strict=True):
        saved = attribution.load(attribution.path(SCREENER, bio.id))
        column = data.OCCUPATIONS.index(label)
        if saved.logits[saved.index[0, -1]].argmax() != column:
            skipped += 1
            continue
        cases.append(Case(bio.id, saved.words, bio.occupation, label, saved.values_for(column)))
    return cases, skipped
