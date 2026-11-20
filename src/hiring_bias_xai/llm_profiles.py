"""Candidates for the LLM audit: which bios, and how each arm presents them."""

import zlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from hiring_bias_xai.config import SEED
from hiring_bias_xai.counterfactual import swap_gender


def draw_name(groups: dict, group: str, gender: str, bio_id: str) -> tuple[str, str]:
    """A first name of `gender` and a surname of `group`, the same for the same bio and arm."""
    key = zlib.crc32(f"{bio_id}|{group}|{gender}".encode())
    rng = np.random.default_rng([SEED, key])
    first_names = groups[group]["first_names"][gender]
    surnames = groups[group]["surnames"]
    return first_names[rng.integers(len(first_names))], surnames[rng.integers(len(surnames))]


def candidate(bio: str, bio_gender: str, bio_id: str, arm: str, groups: dict) -> tuple:
    """The header line (or None) and the text of the bio as the arm presents it.

    `arm` is `none`, `neutral`, `swap` or `name|<group>|<gender>`, whose header is the drawn name
    and whose pronouns are swapped when the bio's gender is not the gender of the name.
    """
    if arm == "none":
        return None, bio
    if arm == "neutral":
        return f"Candidate ID: {bio_id.split('-')[1]}", bio
    if arm == "swap":
        return None, swap_gender(bio)
    kind, group, gender = (arm.split("|") + [None, None])[:3]
    if kind == "name":
        first, surname = draw_name(groups, group, gender, bio_id)
        text = bio if gender == bio_gender else swap_gender(bio)
        return f"Name: {first} {surname}", text
    raise ValueError(f"unknown arm {arm!r}")


def hardest_confusions(true: Sequence, predicted: Sequence, occupations: Sequence) -> dict:
    """For each occupation, the other one in `occupations` that its bios are most often predicted
    as (the first in `occupations` on a tie). Predictions outside `occupations` are ignored."""
    counts = {occupation: Counter() for occupation in occupations}
    for actual, guess in zip(true, predicted, strict=True):
        if actual in counts and guess in counts and guess != actual:
            counts[actual][guess] += 1
    return {
        occupation: max(
            (o for o in occupations if o != occupation), key=lambda o: counts[occupation][o]
        )
        for occupation in occupations
    }
