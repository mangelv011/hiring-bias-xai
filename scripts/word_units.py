"""How often a whitespace unit glues words joined by punctuation, on the showcase subset.

Shapley values are computed per whitespace-separated unit, so "shoot...she" is one unit. This
script counts those units and how many of the gendered units a finer split would change. A unit is
glued when it has letters on both sides of punctuation inside it (an apostrophe does not
count: "it's" is one word). Results go to `results/word_units.json`.
"""

import re
from collections.abc import Sequence
from pathlib import Path

from hiring_bias_xai import data, results
from hiring_bias_xai.config import RESULTS_DIR

RULE = (
    "Units are whitespace-separated words. A unit is glued when its core (punctuation at its ends "
    "removed) has letters on both sides of punctuation inside it, apostrophes excepted. A glued "
    "unit has a gendered part when a piece between that punctuation is a scrubbed word. Gendered "
    "units are the whole units whose core is a scrubbed word plus the glued units with a "
    "gendered part."
)
_GLUED = re.compile(r"[^\W\d_]+[^\w\s'’]+[^\W\d_]+")
_PIECE_BREAK = re.compile(r"[^\w'’]+")


def is_glued(word: str) -> bool:
    return _GLUED.search(word) is not None


def has_gendered_part(word: str) -> bool:
    """Whether a piece of the unit between punctuation is a gendered word."""
    return any(data.is_gendered_word(piece) for piece in _PIECE_BREAK.split(word))


def unit_statistics(texts: Sequence[str]) -> dict:
    units = [word for text in texts for word in text.split()]
    glued = [word for word in units if is_glued(word)]
    whole = sum(data.is_gendered_word(word) for word in units)
    glued_with_gendered = sum(has_gendered_part(word) for word in glued)
    return {
        "n_units": len(units),
        "n_glued": len(glued),
        "share_glued": len(glued) / len(units),
        "n_gendered_units": whole,
        "n_glued_with_gendered_part": glued_with_gendered,
        "share_of_gendered_units_glued": glued_with_gendered / (whole + glued_with_gendered),
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    results.write_json(
        results_dir / "word_units.json",
        {"rule": RULE, "n_bios": len(subset)} | unit_statistics(list(subset["text"])),
    )


if __name__ == "__main__":
    main()
