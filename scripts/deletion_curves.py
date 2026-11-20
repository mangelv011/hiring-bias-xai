"""Deletion curves of both screeners on the showcase subset, from the saved Shapley values.

For each bio, words are removed with `unk` (the attributions used `delete`, so the check is not
circular for the main model) in three orders: MoRF (largest signed Shapley value toward the true
occupation first), LeRF (smallest first) and 10 random orders. At each of 0, 5, ..., 50 percent of
the bio's words (a share of a bio's words is rounded half up to a whole word) the script records
whether the screener's call is still its call on the full bio and how far the log-odds of the true
occupation have fallen. The curve of an order is the mean over bios; the random curve is the mean
over bios of each bio's mean over its 10 orders. Intervals are 95% bootstrap intervals over bios.
Writes `results/deletion_curves.csv`.

The baseline model's TF-IDF reads `[UNK]` as the word "unk", which it does not know, so for it
`unk` is close to `delete` and its curves are partly circular.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hiring_bias_xai import attribution, data
from hiring_bias_xai import explanation_checks as checks
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import BaselineModel, DebertaModel

SCREENERS = ("deberta", "baseline")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCENTS = tuple(range(0, 55, 5))
N_RANDOM = 10
ORDERS = ("morf", "lerf", "random")
PROGRESS_EVERY = 250


def _load(name: str, batch_size: int):
    if name == "baseline":
        return BaselineModel.load()
    model = DebertaModel.load(batch_size)
    model.network.to(DEVICE)
    return model


def _per_bio_curves(name: str, model, subset) -> tuple[np.ndarray, np.ndarray]:
    """Agreement and drop of every bio as arrays of shape (bios, 3 orders, steps)."""
    agreement, drop = [], []
    for position, bio in enumerate(subset.itertuples()):
        saved = attribution.load(attribution.path(name, bio.id))
        if saved.words != bio.text.split():
            raise ValueError(f"{bio.id}: the saved attribution is of other words than the bio")
        rng = np.random.default_rng((SEED, position))
        curve = checks.deletion_curve(
            model, saved.words, saved.values, saved.target, PERCENTS, N_RANDOM, rng
        )
        agreement.append(_by_order(curve.agreement))
        drop.append(_by_order(curve.drop))
        if (position + 1) % PROGRESS_EVERY == 0 or position + 1 == len(subset):
            print(f"{name}: {position + 1}/{len(subset)} bios", flush=True)
    return np.stack(agreement), np.stack(drop)


def _by_order(rows: np.ndarray) -> np.ndarray:
    """MoRF, LeRF and the mean of the random orders, from rows MoRF, LeRF, random..."""
    return np.stack([rows[0], rows[1], rows[2:].mean(axis=0)])


def _mean_over_bios(per_bio: np.ndarray):
    """The statistic `bootstrap_ci` resamples: the mean of the chosen bios' curves."""
    return lambda rows: per_bio[rows].mean(axis=0)


def _curve_table(name: str, agreement: np.ndarray, drop: np.ndarray) -> pd.DataFrame:
    n = len(agreement)
    columns = {}
    for label, per_bio in (("agreement", agreement), ("drop", drop)):
        mean = _mean_over_bios(per_bio)
        low, high = checks.bootstrap_ci(mean, n)
        columns[label] = (mean(np.arange(n)), low, high)
    frames = []
    for i, order in enumerate(ORDERS):
        frames.append(
            pd.DataFrame(
                {
                    "screener": name,
                    "order": order,
                    "fraction": [percent / 100 for percent in PERCENTS],
                    "mean_agreement": columns["agreement"][0][i],
                    "agreement_ci_low": columns["agreement"][1][i],
                    "agreement_ci_high": columns["agreement"][2][i],
                    "mean_logodds_drop": columns["drop"][0][i],
                    "drop_ci_low": columns["drop"][1][i],
                    "drop_ci_high": columns["drop"][2][i],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main(results_dir: Path = RESULTS_DIR) -> None:
    batch_size = json.loads((results_dir / "shapley_pilot.json").read_text())[
        "inference_batch_size"
    ]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    tables = []
    for name in SCREENERS:
        model = _load(name, batch_size)
        agreement, drop = _per_bio_curves(name, model, subset)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        tables.append(_curve_table(name, agreement, drop))
    pd.concat(tables, ignore_index=True).to_csv(results_dir / "deletion_curves.csv", index=False)


if __name__ == "__main__":
    main()
