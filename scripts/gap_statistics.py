"""Gap statistics of the baseline model and the main model, from their saved test predictions."""

import json
from pathlib import Path

import numpy as np

from hiring_bias_xai import data, gap_statistics, results
from hiring_bias_xai.config import RESULTS_DIR

N_BOOT = 1000
N_PERM = 1000
MODELS = {"baseline_model": "baseline", "main_model": "deberta"}
TOLERANCE = 1e-9


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
    share_women = data.share_women(data.load_split("train"))
    test = data.load_split("test")
    output = {}
    for name, file in MODELS.items():
        statistics = gap_statistics.compute(
            test["occupation"],
            results.load_predictions(file, test),
            test["gender"],
            share_women,
            n_boot=N_BOOT,
            n_perm=N_PERM,
        )
        metrics = json.loads((results_dir / f"metrics_{file}.json").read_text())["test"]
        _check_against_metrics(name, statistics, metrics)
        output[name] = statistics
    results.write_json(results_dir / "gap_statistics.json", output)


if __name__ == "__main__":
    main()
