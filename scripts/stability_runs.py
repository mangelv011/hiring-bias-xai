"""The two extra Shapley runs of the stability check, on a stratified sample of the subset.

The sample is 100 showcase-subset bios, equal shares from each occupation and gender group (the
last 4 of the 100 go to the first groups in sorted order), drawn with seed 42 and saved to
`results/stability_sample_ids.csv`. The main model is scored with the pilot's permutation count
and with:

- `deberta-seed7`: removal `delete` as in the main run, orders from seed 7;
- `deberta-unk`: removal `unk`, the orders of the main run (seed 42), so only the removal differs.

Each bio's generator is seeded with (seed, position of the bio in the whole subset), as in
`scripts/shapley_run.py`. One file per bio goes to `artifacts_dir()/attributions/<variant>/`; the
main run's files are never touched. The run is resumable: a bio whose file is current is skipped.
Pass variant names to run only those, one at a time if a call must stay under ten minutes.
"""

import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from hiring_bias_xai import attribution, data
from hiring_bias_xai import explanation_checks as checks
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel

SAMPLE_SIZE = 100
SAMPLE_FILE = "stability_sample_ids.csv"
# variant -> (removal, or None for the main run's, seed of the orders)
VARIANTS = {"deberta-seed7": (None, 7), "deberta-unk": ("unk", SEED)}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(batch_size: int):
    model = DebertaModel.load(batch_size)
    model.network.to(DEVICE)
    return model


def _randomised(model, seed: int):
    """A copy of `model` with every weight redrawn by the network's own initialiser.

    The initialiser is applied module by module, as `init_weights` would on a new network, but
    that method skips modules flagged as already initialised. The trained network is not changed.
    """
    network = copy.deepcopy(model.network).cpu()
    torch.manual_seed(seed)
    network.apply(network._init_weights)
    return DebertaModel(network.to(model.network.device), model.tokenizer, model.batch_size)


def _run_variant(name: str, model, subset, positions, removal: str, n_permutations: int, seed):
    started, done = time.perf_counter(), 0
    for count, position in enumerate(positions, start=1):
        bio = subset.iloc[position]
        file = attribution.path(name, bio["id"])
        target = model.labels.index(bio["occupation"])
        if not attribution.is_current(file, bio["text"].split(), target, removal, n_permutations):
            (result,) = attribution.shapley(
                model,
                [bio["text"]],
                [bio["occupation"]],
                n_permutations,
                removal,
                seed=(seed, int(position)),
            )
            attribution.save(file, result)
            done += 1
        if count % 10 == 0 or count == len(positions):
            print(
                f"{name}: {count}/{len(positions)} bios, {done} scored in "
                f"{time.perf_counter() - started:.0f} s",
                flush=True,
            )


def main(results_dir: Path = RESULTS_DIR, variants=tuple(VARIANTS)) -> None:
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant {unknown[0]!r}; choose from {', '.join(VARIANTS)}")
    pilot = json.loads((results_dir / "shapley_pilot.json").read_text())
    main_removal, n_permutations = pilot["chosen"]["removal"], pilot["chosen"]["n_permutations"]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    positions = checks.stratified_sample(subset, SAMPLE_SIZE, SEED)
    pd.DataFrame({"id": subset["id"].iloc[positions]}).to_csv(
        results_dir / SAMPLE_FILE, index=False
    )
    model = _load(pilot["inference_batch_size"])
    for name in variants:
        removal, seed = VARIANTS[name]
        _run_variant(name, model, subset, positions, removal or main_removal, n_permutations, seed)


if __name__ == "__main__":
    main(variants=tuple(sys.argv[1:]) or tuple(VARIANTS))
