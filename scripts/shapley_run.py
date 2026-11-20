"""Word-level Shapley values of both screeners on the showcase subset, target = true occupation.

The removal and the number of permutations come from `results/shapley_pilot.json`, so run
`scripts/shapley_pilot.py` first. One file per bio and screener goes to
`artifacts_dir()/attributions/<screener>/`, and the run is resumable: a bio whose file exists, is
readable and was made with the same settings is skipped. Summary statistics are read back from
those files and written to `results/shapley_summary.json`.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from hiring_bias_xai import attribution, data, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import BaselineModel, DebertaModel

SCREENERS = ("deberta", "baseline")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROGRESS_EVERY = 25


def _load(name: str, batch_size: int):
    if name == "baseline":
        return BaselineModel.load()
    model = DebertaModel.load(batch_size)
    model.network.to(DEVICE)
    return model


def _run_screener(name: str, model, subset, removal: str, n_permutations: int) -> None:
    started, done = time.perf_counter(), 0
    for position, bio in enumerate(subset.itertuples()):
        file = attribution.path(name, bio.id)
        target = model.labels.index(bio.occupation)
        if not attribution.is_current(file, bio.text.split(), target, removal, n_permutations):
            (result,) = attribution.shapley(
                model, [bio.text], [bio.occupation], n_permutations, removal, seed=(SEED, position)
            )
            attribution.save(file, result)
            done += 1
        if (position + 1) % PROGRESS_EVERY == 0 or position + 1 == len(subset):
            print(
                f"{name}: {position + 1}/{len(subset)} bios, {done} scored in "
                f"{time.perf_counter() - started:.0f} s",
                flush=True,
            )


def _summarise(name: str, subset) -> dict:
    saved = [attribution.load(attribution.path(name, bio_id)) for bio_id in subset["id"]]
    return {
        "n_bios": len(saved),
        "mean_words": float(np.mean([len(item.words) for item in saved])),
        "mean_evaluations": float(np.mean([len(item.logits) for item in saved])),
        "total_seconds": round(sum(item.seconds for item in saved), 1),
        "efficiency_max_error": max(item.efficiency_error() for item in saved),
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    pilot_file = results_dir / "shapley_pilot.json"
    if not pilot_file.is_file():
        raise SystemExit(f"{pilot_file} is missing: run scripts/shapley_pilot.py first")
    pilot = json.loads(pilot_file.read_text())
    chosen = pilot["chosen"]
    if chosen["method"] != "shapley":
        raise SystemExit(
            f"the pilot chose {chosen['method']}: the projected Shapley run is too long, so "
            "fall back to leave-one-out occlusion (see ADR 0003)"
        )
    removal, n_permutations = chosen["removal"], chosen["n_permutations"]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    summary = {}
    for name in SCREENERS:
        model = _load(name, pilot["inference_batch_size"])
        _run_screener(name, model, subset, removal, n_permutations)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        summary[name] = _summarise(name, subset)
    results.write_json(
        results_dir / "shapley_summary.json",
        {
            "settings": {
                "removal": removal,
                "n_permutations": n_permutations,
                "target": "true occupation",
                "seed": SEED,
            },
            "screeners": summary,
        },
    )


if __name__ == "__main__":
    main()
