import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"

# Two easy occupations: every nurse bio says "ward patient" and every surgeon bio "scalpel
# theatre", so any hyperparameters give a perfect dev score.
WORDS = {"nurse": "ward patient", "surgeon": "scalpel theatre"}


def _split(name, per_cell):
    cells = [(o, g) for o in WORDS for g in ("female", "male") for _ in range(per_cell)]
    rows = [(f"{name}-{i}", f"{WORDS[o]} bio", o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture
def train_baseline(monkeypatch, tmp_path):
    """Return run(): the script's main() on synthetic splits, giving the results directory."""
    splits = {"train": _split("train", 6), "dev": _split("dev", 3), "test": _split("test", 3)}
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": splits[split])
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("train_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run():
        module.main(tmp_path / "results")
        return tmp_path / "results"

    return run


def test_train_baseline_writes_the_metrics_file(train_baseline):
    metrics = json.loads((train_baseline() / "metrics_baseline.json").read_text())
    assert sorted(metrics) == [
        "dev_grid",
        "hyperparameters",
        "overfitting_check",
        "runtime_seconds",
        "test",
    ]


def test_ties_on_dev_go_to_the_smaller_C_and_the_larger_min_df(train_baseline):
    metrics = json.loads((train_baseline() / "metrics_baseline.json").read_text())
    assert {row["dev_macro_f1"] for row in metrics["dev_grid"]} == {1.0}
    assert metrics["hyperparameters"] == {"C": 0.5, "min_df": 5}
