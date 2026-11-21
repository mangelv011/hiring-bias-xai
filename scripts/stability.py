"""Stability of the main model's Shapley values, and their agreement with leave-one-out occlusion.

Three checks, each with an expectation written before the run and a threshold (`passed` is true
only when the median Spearman correlation and the median top-10 overlap both reach theirs):

- seed: the saved values (seed 42) against a second run with seed 7 (`scripts/stability_runs.py`),
  on the stratified sample of 100 bios;
- removal: the saved values (`delete`) against a run with `unk` and the same orders, same sample;
- method: the saved values against leave-one-out occlusion, on all showcase-subset bios.

A fourth comparison, `randomisation`, has no threshold and no verdict: the saved values against a
run of the same model with every weight redrawn (`deberta-random`), same sample. Its expectation
is written before the run; the numbers are reported as they are.

Per bio, the Spearman correlation is over all words and the overlap is the share of the 10 words
with the largest absolute value in one vector that are among the largest in the other. Values are
for the true occupation. Means and medians over bios come with 95% bootstrap intervals over bios;
a bio whose correlation is undefined is left out of the correlation statistics.

The disagreement example is chosen by `explanation_checks.disagreement_example` (the rule is in
the json). Writes `results/stability.json`; the leave-one-out values need the main model and GPU.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hiring_bias_xai import attribution, data, results
from hiring_bias_xai import explanation_checks as checks
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel

SCREENER = "deberta"
SAMPLE_FILE = "stability_sample_ids.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHOWN = 8
RANKING = "Words are ranked by absolute value; ties go to the word that comes first in the bio."
SAMPLE_RULE = (
    "100 showcase-subset bios, equal shares from each occupation and gender group (the last 4 go "
    "to the first groups in sorted order), drawn without replacement with seed 42."
)
EXAMPLE_RULE = (
    "Among bios with at least 15 word units, no run of six digits, and a top Shapley word (largest "
    "absolute value) that occurs at least twice (compared without case and without punctuation at "
    "its ends), the bio with the lowest Spearman correlation between Shapley and leave-one-out "
    "values; the first in the subset on a tie."
)
RULE = (
    "Passed when the median Spearman correlation and the median top-10 overlap over bios both "
    "reach their thresholds."
)
THRESHOLDS = {
    "seed": {"median_spearman": 0.9, "median_top10_overlap": 0.7},
    "removal": {"median_spearman": 0.8, "median_top10_overlap": 0.5},
    "method": {"median_spearman": 0.7, "median_top10_overlap": 0.5},
}
COMPARISON = {
    "seed": "Shapley values, delete removal, seed 42 against seed 7 (40 permutations each)",
    "removal": "Shapley values, seed 42, delete removal against unk removal (same orders)",
    "method": "Shapley values (delete, seed 42) against leave-one-out occlusion (delete)",
}
EXPECTED = {
    "seed": (
        "Median Spearman near 0.89, the cost pilot's value at 40 permutations on 50 other bios, so "
        "just under the 0.9 rule of ADR 0003 and likely a miss; median top-10 overlap about 0.7 to "
        "0.8, because the largest words are more stable than the many words near zero."
    ),
    "removal": (
        "Lower than the seed check: the main model reads [UNK] as a token, so the value of a word "
        "depends on what replaces its neighbours. Median Spearman about 0.7 to 0.85, so a miss "
        "of the 0.8 threshold is likely; median top-10 overlap about 0.5 to 0.7."
    ),
    "method": (
        "Median Spearman about 0.6 to 0.8 and median top-10 overlap about 0.5: both methods find "
        "the strongest words, but leave-one-out gives each of several redundant words a small "
        "value and Shapley shares the credit. Borderline on the 0.7 threshold."
    ),
}


RANDOM_COMPARISON = (
    "Shapley values, delete removal, seed 42: the trained main model against the same model with "
    "every weight redrawn by its initialiser (seed 42), same orders"
)
RANDOM_EXPECTED = (
    "Median Spearman correlation and median top-10 overlap near 0, against 0.89 and 0.80 for seed "
    "7 against seed 42. The scores of a random network do not depend on what the trained model "
    "learned, so its ranking of the words should not follow the trained one. An overlap of a few "
    "words in 10 can come from the bio's own structure (long words, repeated words), so the "
    "overlap may sit a little above 0."
)


def _load(batch_size: int):
    model = DebertaModel.load(batch_size)
    model.network.to(DEVICE)
    return model


def _statistics(values: np.ndarray) -> dict:
    """Mean and median over bios with 95% bootstrap intervals; NaN values are left out."""
    kept = values[~np.isnan(values)]
    out = {"n_undefined": int(len(values) - len(kept))}
    for name, function in (("mean", np.mean), ("median", np.median)):
        low, high = checks.bootstrap_ci(lambda rows, f=function: f(kept[rows]), len(kept))
        out[name] = float(function(kept))
        out[f"{name}_ci"] = [float(low), float(high)]
    return out


def _summary(pairs: list[tuple[float, float]]) -> dict:
    spearman = np.array([pair[0] for pair in pairs])
    overlap = np.array([pair[1] for pair in pairs])
    return {"spearman": _statistics(spearman), "top10_overlap": _statistics(overlap)}


def _randomisation(pairs: list[tuple[float, float]]) -> dict:
    return {
        "comparison": RANDOM_COMPARISON,
        "n_bios": len(pairs),
        "expected": RANDOM_EXPECTED,
        **_summary(pairs),
    }


def _check(name: str, pairs: list[tuple[float, float]]) -> dict:
    summary = _summary(pairs)
    thresholds = THRESHOLDS[name]
    return {
        "comparison": COMPARISON[name],
        "n_bios": len(pairs),
        "expected": EXPECTED[name],
        "rule": RULE,
        "thresholds": thresholds,
        **summary,
        "passed": bool(
            summary["spearman"]["median"] >= thresholds["median_spearman"]
            and summary["top10_overlap"]["median"] >= thresholds["median_top10_overlap"]
        ),
    }


def _values(variant: str, bio_id: str) -> tuple[list[str], int, np.ndarray]:
    saved = attribution.load(attribution.path(variant, bio_id))
    return saved.words, saved.target, saved.values


def _run_pairs(variant: str, ids: list[str]) -> list[tuple[float, float]]:
    """Agreement between the main run and `variant` for each bio of `ids`."""
    pairs = []
    for bio_id in ids:
        words, target, main = _values(SCREENER, bio_id)
        other_words, other_target, other = _values(variant, bio_id)
        if (words, target) != (other_words, other_target):
            raise ValueError(f"{bio_id}: {variant} was run on other words or another occupation")
        pairs.append(checks.rank_agreement(main, other))
    return pairs


def _explain(case: checks.Case, spearman: float, model) -> dict:
    """The disagreement example: what Shapley and leave-one-out say about its top word."""
    target = attribution.load(attribution.path(SCREENER, case.bio_id)).target
    cores = [data.word_core(word) or word for word in case.words]
    top = int(np.argmax(np.abs(case.shapley)))
    where = [i for i, core in enumerate(cores) if core == cores[top]]
    drop = checks.drop_without(model, case.words, [where], target)[0]
    largest = np.argsort(-np.abs(case.shapley), kind="stable")[:SHOWN]
    return {
        "rule": EXAMPLE_RULE,
        "bio_id": case.bio_id,
        "true_occupation": data.OCCUPATIONS[target],
        "n_words": len(case.words),
        "spearman": spearman,
        "word": case.words[top],
        "n_occurrences": len(where),
        "occurrences": [
            {"position": i, "shapley": float(case.shapley[i]), "loo": float(case.loo[i])}
            for i in where
        ],
        "drop_all_occurrences": float(drop),
        "top_words": [
            {"word": case.words[i], "shapley": float(case.shapley[i]), "loo": float(case.loo[i])}
            for i in largest
        ],
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    batch_size = json.loads((results_dir / "shapley_pilot.json").read_text())[
        "inference_batch_size"
    ]
    sample_ids = pd.read_csv(results_dir / SAMPLE_FILE)["id"].tolist()
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    model = _load(batch_size)
    cases = []
    for bio_id in subset["id"]:
        words, target, shapley = _values(SCREENER, bio_id)
        cases.append(checks.Case(bio_id, words, shapley, checks.loo_values(model, words, target)))
    method = [checks.rank_agreement(case.shapley, case.loo) for case in cases]
    example = checks.disagreement_example(cases)
    results.write_json(
        results_dir / "stability.json",
        {
            "settings": {
                "top": checks.TOP,
                "word_ranking": RANKING,
                "target": "true occupation",
                "n_boot": checks.N_BOOT,
                "bootstrap_seed": SEED,
            },
            "sample": {
                "n_bios": len(sample_ids),
                "ids_file": SAMPLE_FILE,
                "rule": SAMPLE_RULE,
                "seed": SEED,
            },
            "checks": {
                "seed": _check("seed", _run_pairs("deberta-seed7", sample_ids)),
                "removal": _check("removal", _run_pairs("deberta-unk", sample_ids)),
                "method": _check("method", method),
            },
            "randomisation": _randomisation(_run_pairs("deberta-random", sample_ids)),
            "disagreement_example": _explain(*example, model) if example else None,
        },
    )


if __name__ == "__main__":
    main()
