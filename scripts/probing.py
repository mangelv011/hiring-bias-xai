"""Can gender be read from scrubbed bios, and from which layer? Linear probes on mean-pooled states.

Run `scripts/probing_features.py` for both models first. Then run this script once per part
(`pretrained`, `deberta`, `references`; each takes under 9 minutes on the laptop CPU). The last
part to finish writes `results/probing.csv` and `results/probing.json` from all three.

A probe is logistic regression on standardised features, C chosen by 5-fold cross-validation on
the 10,000 training bios, scored on the scrubbed showcase subset (accuracy, 95% bootstrap
interval over bios, permutation p-value from `probing.probe`). The references, on the same split:
the majority gender, the majority gender of the true occupation in the training bios, and
TF-IDF with logistic regression on the same scrubbed text. Every number is given for all bios and
for the bios without a first name found by `counterfactual.first_names_in`.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hiring_bias_xai import counterfactual, data, first_names, probing, results
from hiring_bias_xai.config import RESULTS_DIR, SEED, artifacts_dir

MODELS = ("pretrained", "deberta")
PARTS = (*MODELS, "references")
N_PERMUTATIONS = 100
N_BOOT = probing.N_BOOT
N_JOBS = 1
COLUMNS = [
    "model", "layer", "subset", "accuracy", "ci_low", "ci_high", "p_value", "n", "c",
    "n_permutations", "null_mean",
]  # fmt: skip


def _folder() -> Path:
    return artifacts_dir() / "probing"


def _bios(results_dir: Path):
    """Training bios in the order of the saved ids, the scrubbed showcase subset and, per
    showcase bio, whether a first name was found in its original text."""
    ids = pd.read_csv(results_dir / "probing_train_ids.csv")["id"]
    train = data.load_split("train", "scrubbed").set_index("id").loc[ids].reset_index()
    showcase = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    names = first_names.load()
    named = np.array([bool(counterfactual.first_names_in(t, names)) for t in showcase["text"]])
    showcase = showcase.assign(text=showcase["text"].map(data.scrub))
    return train, showcase, named


def _probe_rows(model: str, train, showcase, unflagged: np.ndarray) -> list[dict]:
    features = {
        part: np.load(_folder() / f"{model}_{part}.npy").astype(np.float32)
        for part in ("train", "showcase")
    }
    rows = []
    for layer in range(features["train"].shape[1]):
        result = probing.probe(
            features["train"][:, layer],
            train["gender"].to_numpy(),
            train["occupation"].to_numpy(),
            features["showcase"][:, layer],
            showcase["gender"].to_numpy(),
            unflagged,
            N_PERMUTATIONS,
            N_BOOT,
            SEED,
            N_JOBS,
        )
        for subset in ("all", "unflagged"):
            row = result[subset] | {"model": model, "layer": layer, "subset": subset}
            rows.append(row | {"c": result["C"], "n_permutations": result["n_permutations"]})
    return rows


def _reference_rows(train, showcase, unflagged: np.ndarray) -> list[dict]:
    truth = showcase["gender"].to_numpy()
    predictions = {
        "majority": probing.majority_class(train["gender"], len(showcase)),
        "occupation_only": probing.occupation_only(
            train["occupation"], train["gender"], showcase["occupation"]
        ),
        "tfidf": probing.tfidf_predictions(train["text"], train["gender"], showcase["text"]),
    }
    rows = []
    for name, predicted in predictions.items():
        hits = predicted == truth
        for subset, chosen in (("all", np.ones(len(hits), dtype=bool)), ("unflagged", unflagged)):
            rows.append(
                {"model": name, "layer": None, "subset": subset}
                | probing.accuracy_ci(hits[chosen], N_BOOT, SEED)
            )
    return rows


def _assemble(results_dir: Path, parts: dict) -> None:
    table = pd.DataFrame([row for name in PARTS for row in parts[name]["rows"]])
    table["layer"] = table["layer"].astype("Int64")
    table.reindex(columns=COLUMNS).to_csv(results_dir / "probing.csv", index=False)
    results.write_json(
        results_dir / "probing.json",
        {
            "settings": {
                "n_train": len(pd.read_csv(results_dir / "probing_train_ids.csv")),
                "n_test": int(table[table["subset"] == "all"]["n"].max()),
                "n_boot": N_BOOT,
                "n_permutations": N_PERMUTATIONS,
                "seed": SEED,
                "c_grid": list(probing.C_GRID),
                "tfidf_c_grid": list(probing.TFIDF_C_GRID),
                "folds": probing.FOLDS,
            },
            "n_flagged": parts["references"]["n_flagged"],
            "seconds": {name: parts[name]["seconds"] for name in PARTS},
        },
    )


def main(part: str, results_dir: Path = RESULTS_DIR) -> None:
    if part not in PARTS:
        raise ValueError(f"part must be one of {', '.join(PARTS)}; got {part!r}")
    started = time.perf_counter()
    train, showcase, named = _bios(results_dir)
    if part == "references":
        rows = _reference_rows(train, showcase, ~named)
    else:
        rows = _probe_rows(part, train, showcase, ~named)
    seconds = round(time.perf_counter() - started, 1)
    _folder().mkdir(parents=True, exist_ok=True)
    results.write_json(
        _folder() / f"part_{part}.json",
        {"rows": rows, "seconds": seconds, "n_flagged": int(named.sum())},
    )
    print(f"{part}: {len(rows)} rows in {seconds} s", flush=True)
    files = {name: _folder() / f"part_{name}.json" for name in PARTS}
    if all(file.is_file() for file in files.values()):
        _assemble(results_dir, {name: json.loads(file.read_text()) for name, file in files.items()})


if __name__ == "__main__":
    main(sys.argv[1])
