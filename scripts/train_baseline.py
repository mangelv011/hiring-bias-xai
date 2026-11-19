"""Train the baseline model, pick C and min_df on dev, and evaluate once on test."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from hiring_bias_xai import data, fairness
from hiring_bias_xai.config import RESULTS_DIR, artifacts_dir
from hiring_bias_xai.models import BaselineModel

# Each fit takes 30 to 70 s (lbfgs stops at 100 iterations), so the grid stays small.
# C below 4 underfits TF-IDF features: dev macro-F1 0.66 at C=1.
C_GRID = (4, 16, 64)
MIN_DF_GRID = (5, 20)


def _write_json(path: Path, content: dict) -> None:
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def _predict(model: BaselineModel, texts) -> np.ndarray:
    return np.array(model.labels)[model.predict_proba(texts).argmax(axis=1)]


def _macro_f1(model: BaselineModel, frame) -> float:
    occupations = frame["occupation"]
    predicted = _predict(model, frame["text"])
    return float(f1_score(occupations, predicted, average="macro"))


def _share_women(frame) -> dict:
    return (frame["gender"] == "female").groupby(frame["occupation"]).mean().to_dict()


def main(results_dir: Path = RESULTS_DIR) -> None:
    started = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    train, dev, test = (data.load_split(split) for split in ("train", "dev", "test"))
    dev_grid, best, best_key = [], None, None
    for C, min_df in itertools.product(C_GRID, MIN_DF_GRID):
        model = BaselineModel(C=C, min_df=min_df).fit(train["text"], train["occupation"])
        dev_f1 = _macro_f1(model, dev)
        dev_grid.append({"C": C, "min_df": min_df, "dev_macro_f1": dev_f1})
        key = (dev_f1, -C, min_df)  # ties go to the smaller C, then the larger min_df
        if best_key is None or key > best_key:
            best, best_key = model, key
    predicted = _predict(best, test["text"])
    best.save()
    predictions = artifacts_dir() / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "predicted_occupation": predicted}).to_csv(
        predictions / "baseline_test.csv", index=False
    )
    report = fairness.evaluate(
        test["occupation"],
        predicted,
        test["gender"],
        _share_women(train),
    )
    train_f1 = _macro_f1(best, train)
    _write_json(
        results_dir / "metrics_baseline.json",
        {
            "hyperparameters": {"C": best.C, "min_df": best.min_df},
            "dev_grid": dev_grid,
            "test": report,
            "overfitting_check": {
                "train_macro_f1": train_f1,
                "test_macro_f1": report["macro_f1"]["value"],
                "gap": train_f1 - report["macro_f1"]["value"],
            },
            "runtime_seconds": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
