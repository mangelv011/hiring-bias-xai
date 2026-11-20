"""Candidates for the LLM audit: which bios, and how each arm presents them."""

import zlib

import numpy as np

from hiring_bias_xai.config import SEED


def draw_name(groups: dict, group: str, gender: str, bio_id: str) -> tuple[str, str]:
    """A first name of `gender` and a surname of `group`, the same for the same bio and arm."""
    key = zlib.crc32(f"{bio_id}|{group}|{gender}".encode())
    rng = np.random.default_rng([SEED, key])
    first_names = groups[group]["first_names"][gender]
    surnames = groups[group]["surnames"]
    return first_names[rng.integers(len(first_names))], surnames[rng.integers(len(surnames))]


def candidate(bio: str, bio_gender: str, bio_id: str, arm: str, groups: dict) -> tuple:
    """The header line (or None) and the text of the bio as the arm presents it.

    `arm` is `none` or `neutral`.
    """
    if arm == "none":
        return None, bio
    if arm == "neutral":
        return f"Candidate ID: {bio_id.split('-')[1]}", bio
    raise ValueError(f"unknown arm {arm!r}")
