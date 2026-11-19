"""Train the baseline model, pick C and min_df on dev, and evaluate once on test."""

import itertools
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from hiring_bias_xai import data, fairness, results
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.models import BaselineModel

# The run must fit in 10 minutes, so the grid stays small. Dev macro-F1 rises with C and
# peaks at C=256 for min_df=5 (0.720, 0.726, 0.729 at C=16, 64, 256). Off-grid checks: C=4 is
# clearly worse, and C=1024 and C=4096 fall to 0.725 and 0.710, so the best C is inside the grid.
C_GRID = (16, 64, 256)
MIN_DF_GRID = (5, 20)
# One process per grid point. Fits are single-threaded and memory-bound: 6 at once run at
# 40% of the speed of one (2.4x in total).
N_JOBS = 6


def _predict(model: BaselineModel, texts) -> np.ndarray:
    return np.array(model.labels)[model.predict_proba(texts).argmax(axis=1)]


def _macro_f1(model: BaselineModel, frame) -> float:
    occupations = frame["occupation"]
    predicted = _predict(model, frame["text"])
    return float(f1_score(occupations, predicted, average="macro"))


def _fit_and_score(C: float, min_df: int, train, dev) -> tuple[BaselineModel, float]:
    model = BaselineModel(C=C, min_df=min_df).fit(train["text"], train["occupation"])
    return model, _macro_f1(model, dev)


def main(results_dir: Path = RESULTS_DIR) -> None:
    started = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    train, dev, test = (data.load_split(split) for split in ("train", "dev", "test"))
    fits = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_and_score)(C, min_df, train, dev)
        for C, min_df in itertools.product(C_GRID, MIN_DF_GRID)
    )
    dev_grid, best, best_key = [], None, None
    for model, dev_f1 in fits:
        dev_grid.append({"C": model.C, "min_df": model.min_df, "dev_macro_f1": dev_f1})
        key = (dev_f1, -model.C, model.min_df)  # ties: the smaller C, then the larger min_df
        if best_key is None or key > best_key:
            best, best_key = model, key
    predicted = _predict(best, test["text"])
    best.save()
    results.save_predictions("baseline", test, predicted)
    report = fairness.evaluate(
        test["occupation"],
        predicted,
        test["gender"],
        data.share_women(train),
    )
    train_f1 = _macro_f1(best, train)
    results.write_json(
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
