"""Counterfactual gender swaps on the showcase subset, with residual-cue flags and a removal control.

For each bio: the gender swap (`counterfactual.swap_gender`) and the removal control
(`data.scrub`) are scored by both screeners, and the flip rate, the change in the log-odds of
the true occupation and the flip direction are summarised per showcase occupation and pooled,
with and without the bios that carry a residual cue (`counterfactual.residual_cues`).
Writes `results/counterfactual_gender.csv` and `results/counterfactual_gender.json`; the json
also holds the summary of the committed hand check (`results/counterfactual_hand_check.csv`).
Needs the saved models and `scripts/fetch_first_names.py` run first; the main model uses the GPU.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hiring_bias_xai import counterfactual, counterfactual_effects, data, first_names, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import BaselineModel, DebertaModel, log_odds

SCREENERS = ("deberta", "baseline")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
N_BOOT = counterfactual_effects.N_BOOT
HAND_CHECK_FILE = "counterfactual_hand_check.csv"


def _load(name: str):
    if name == "baseline":
        return BaselineModel.load()
    model = DebertaModel.load(BATCH_SIZE)
    model.network.to(DEVICE)
    return model


def _score(model, texts: list[str], occupations: pd.Series) -> tuple[list[str], np.ndarray]:
    """The predicted occupation of each text and the log-odds of its true occupation."""
    logits = model.logits(texts)
    predicted = [model.labels[i] for i in logits.argmax(axis=1)]
    true_log_odds = np.empty(len(texts))
    for occupation in occupations.unique():
        rows = (occupations == occupation).to_numpy()
        true_log_odds[rows] = log_odds(logits[rows], model.labels.index(occupation))
    return predicted, true_log_odds


def _flags(subset: pd.DataFrame, names: frozenset[str]) -> tuple[pd.Series, dict]:
    """Per bio whether it has a residual cue, and the counts that go into the json."""
    nouns = [counterfactual.residual_cues(text) for text in subset["text"]]
    found = [counterfactual.first_names_in(text, names) for text in subset["text"]]
    flagged = pd.Series([bool(a or b) for a, b in zip(nouns, found, strict=True)], dtype=bool)
    cues = Counter(cue for bio in (*nouns, *found) for cue in bio)
    by_occupation = flagged.groupby(subset["occupation"].to_numpy()).sum()
    return flagged, {
        "n_bios": len(subset),
        "n_flagged": int(flagged.sum()),
        "n_noun": sum(bool(bio) for bio in nouns),
        "n_name": sum(bool(bio) for bio in found),
        "by_occupation": {name: int(count) for name, count in by_occupation.items()},
        "cues": dict(sorted(cues.items(), key=lambda item: (-item[1], item[0]))),
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    share_women = data.share_women(data.load_split("train"))
    flagged, flags = _flags(subset, first_names.load())
    swapped = [counterfactual.swap_gender(text) for text in subset["text"]]
    removed = [data.scrub(text) for text in subset["text"]]
    tables = []
    for screener in SCREENERS:
        model = _load(screener)
        original, lo_original = _score(model, list(subset["text"]), subset["occupation"])
        after_swap, lo_swapped = _score(model, swapped, subset["occupation"])
        after_removal, lo_removed = _score(model, removed, subset["occupation"])
        scored = pd.DataFrame(
            {
                "occupation": subset["occupation"].to_numpy(),
                "gender": subset["gender"].to_numpy(),
                "flagged": flagged.to_numpy(),
                "original": original,
                "swapped": after_swap,
                "removed": after_removal,
                "lo_original": lo_original,
                "lo_swapped": lo_swapped,
                "lo_removed": lo_removed,
            }
        )
        tables.append(counterfactual_effects.table(scored, share_women, screener, N_BOOT))
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    pd.concat(tables).to_csv(results_dir / "counterfactual_gender.csv", index=False)
    record = {
        "settings": {"n_boot": N_BOOT, "seed": SEED, "min_name_count": first_names.MIN_COUNT},
        "direction_basis": counterfactual_effects.DIRECTION_BASIS,
        "flags": flags,
        "edits": {
            "swap_changed": sum(a != b for a, b in zip(subset["text"], swapped, strict=True)),
            "removal_changed": sum(a != b for a, b in zip(subset["text"], removed, strict=True)),
        },
    }
    checks = results_dir / HAND_CHECK_FILE
    if checks.is_file():
        record["hand_check"] = counterfactual_effects.hand_check_summary(pd.read_csv(checks))
    results.write_json(results_dir / "counterfactual_gender.json", record)


if __name__ == "__main__":
    main()
