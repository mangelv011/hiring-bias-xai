"""Gap statistics of the baseline model and the main model, from their saved test predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hiring_bias_xai import data, gap_statistics
from hiring_bias_xai.config import RESULTS_DIR, artifacts_dir

N_BOOT = 1000
N_PERM = 1000
MODELS = {"baseline_model": "baseline", "main_model": "deberta"}
TOLERANCE = 1e-9


def _share_women(frame) -> dict:
    return (frame["gender"] == "female").groupby(frame["occupation"]).mean().to_dict()


def _check_against_metrics(name: str, statistics: dict, test_metrics: dict) -> None:
    """Fail if a point value here differs from the one `fairness.evaluate` stored."""
    expected = {
        "gap_share_pearson": (statistics["pearson"]["value"], test_metrics["gap_share_pearson"]),
        "rms_tpr_gap": (statistics["rms_tpr_gap"]["observed"], test_metrics["rms_tpr_gap"]),
    }
    for occupation, row in statistics["occupations"].items():
        expected[f"tpr_gap of {occupation}"] = (
            row["tpr_gap"],
            test_metrics["per_occupation"][occupation]["tpr_gap"],
        )
    for metric, (here, stored) in expected.items():
        if not np.isclose(here, stored["value"], rtol=0, atol=TOLERANCE):
            raise ValueError(
                f"{name}: {metric} is {here} here and {stored['value']} in the metrics"
            )


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    share_women = _share_women(data.load_split("train"))
    test = data.load_split("test")
    output = {}
    for name, file in MODELS.items():
        predictions = pd.read_csv(artifacts_dir() / "predictions" / f"{file}_test.csv")
        predicted = test[["id"]].merge(predictions, on="id", how="left", validate="one_to_one")
        if predicted["predicted_occupation"].isna().any():
            raise ValueError(f"{name}: some test bios have no prediction")
        statistics = gap_statistics.compute(
            test["occupation"],
            predicted["predicted_occupation"],
            test["gender"],
            share_women,
            n_boot=N_BOOT,
            n_perm=N_PERM,
        )
        metrics = json.loads((results_dir / f"metrics_{file}.json").read_text())["test"]
        _check_against_metrics(name, statistics, metrics)
        output[name] = statistics
    (results_dir / "gap_statistics.json").write_bytes(
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode()
    )


if __name__ == "__main__":
    main()
