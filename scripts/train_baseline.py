"""Train the baseline model, pick C and min_df on dev, and evaluate once on test."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from hiring_bias_xai import data, fairness
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.models import BaselineModel

C_GRID = (0.5, 1, 2, 4)
MIN_DF_GRID = (2, 5)


def _write_json(path: Path, content: dict) -> None:
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def _predict(model: BaselineModel, texts) -> np.ndarray:
    return np.array(model.labels)[model.predict_proba(texts).argmax(axis=1)]


def _macro_f1(model: BaselineModel, frame) -> float:
    occupations = frame["occupation"]
    predicted = _predict(model, frame["text"])
    return float(f1_score(occupations, predicted, labels=occupations.unique(), average="macro"))


def _share_women(frame) -> dict:
    return (frame["gender"] == "female").groupby(frame["occupation"]).mean().to_dict()


def main(results_dir: Path = RESULTS_DIR) -> None:
    started = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    train, dev, test = (data.load_split(split) for split in ("train", "dev", "test"))
    fitted = []
    for C, min_df in itertools.product(C_GRID, MIN_DF_GRID):
        model = BaselineModel(C=C, min_df=min_df).fit(train["text"], train["occupation"])
        fitted.append((_macro_f1(model, dev), model))
    _, best = max(fitted, key=lambda scored: scored[0])
    report = fairness.evaluate(
        test["occupation"],
        _predict(best, test["text"]),
        test["gender"],
        _share_women(train),
    )
    train_f1 = _macro_f1(best, train)
    _write_json(
        results_dir / "metrics_baseline.json",
        {
            "hyperparameters": {"C": best.C, "min_df": best.min_df},
            "dev_grid": [
                {"C": model.C, "min_df": model.min_df, "dev_macro_f1": f1} for f1, model in fitted
            ],
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
