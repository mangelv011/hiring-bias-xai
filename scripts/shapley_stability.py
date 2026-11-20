"""Seed agreement of the Shapley values, for the largest words and for the rest.

The cost pilot reports one Spearman correlation per bio over all its words. Most words have values
near zero, and their order is noise, so that number understates how stable the words that matter
are. This script splits the correlation: the TOP words by size of value against the others, and the
overlap of the SHOWN largest words between two seeds. It reruns the cost pilot's bios and seeds on
the baseline model, which is fast on a CPU; the main model's two-seed values are not stored.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from hiring_bias_xai import attribution, data, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import BaselineModel

BIOS = 50
PERMUTATIONS = (5, 10, 20, 40)
SEEDS = (0, 1)
TOP = 10
SHOWN = 8
RULE = (
    "The cost pilot's 50 showcase-subset bios, its removal and its seeds 0 and 1, on the baseline "
    "model. For each bio and count: the Spearman correlation between the two seeds over all words, "
    "over the top 10 words (by the mean of |value| over the two seeds) and over the other words, "
    "and the share of the 8 words with the largest |value| in one seed that are also among the 8 "
    "largest in the other. Bios where a correlation is undefined are left out of its median."
)


def _load_baseline():
    return BaselineModel.load()


def _spearman(one: np.ndarray, other: np.ndarray) -> float:
    if len(one) < 2 or np.ptp(one) == 0 or np.ptp(other) == 0:
        return float("nan")
    return float(spearmanr(one, other).statistic)


def agreement(one: np.ndarray, other: np.ndarray, top: int, shown: int) -> dict:
    """Agreement of two seeds' values for one bio; NaN where a correlation is undefined."""
    largest = np.argsort(-(np.abs(one) + np.abs(other)), kind="stable")
    in_top = np.zeros(len(one), dtype=bool)
    in_top[largest[:top]] = True
    shown_one = set(np.argsort(-np.abs(one), kind="stable")[:shown])
    shown_other = set(np.argsort(-np.abs(other), kind="stable")[:shown])
    return {
        "spearman_all": _spearman(one, other),
        "spearman_top": _spearman(one[in_top], other[in_top]),
        "spearman_rest": _spearman(one[~in_top], other[~in_top]),
        "top_overlap": len(shown_one & shown_other) / min(shown, len(one)),
    }


def _median(values: list[float]):
    defined = [v for v in values if not np.isnan(v)]
    return float(np.median(defined)) if defined else None


def main(results_dir: Path = RESULTS_DIR) -> None:
    removal = json.loads((results_dir / "shapley_pilot.json").read_text())["chosen"]["removal"]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    sample = subset.sample(BIOS, random_state=SEED).reset_index(drop=True)
    model = _load_baseline()
    texts, targets = list(sample["text"]), list(sample["occupation"])
    by_count = {}
    for count in PERMUTATIONS:
        runs = [attribution.shapley(model, texts, targets, count, removal, seed) for seed in SEEDS]
        rows = [
            agreement(one.values, other.values, TOP, SHOWN)
            for one, other in zip(*runs, strict=True)
        ]
        by_count[count] = {
            "n_bios": len(rows),
            "median_spearman_all": _median([r["spearman_all"] for r in rows]),
            "median_spearman_top": _median([r["spearman_top"] for r in rows]),
            "median_spearman_rest": _median([r["spearman_rest"] for r in rows]),
            "mean_top_overlap": float(np.nanmean([r["top_overlap"] for r in rows])),
        }
    results.write_json(
        results_dir / "shapley_stability.json",
        {
            "rule": RULE,
            "model": "baseline",
            "removal": removal,
            "n_bios": len(texts),
            "seeds": list(SEEDS),
            "top": TOP,
            "shown": SHOWN,
            "by_permutations": by_count,
        },
    )


if __name__ == "__main__":
    main()
