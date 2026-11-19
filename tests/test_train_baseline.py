import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data
from hiring_bias_xai.models import BaselineModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"

# Two easy occupations: every nurse bio says "ward patient" and every surgeon bio "scalpel
# theatre", so any hyperparameters give a perfect dev score.
WORDS = {"nurse": "ward patient", "surgeon": "scalpel theatre"}


def _split(name, per_cell):
    cells = [(o, g) for o in WORDS for g in ("female", "male") for _ in range(per_cell)]
    rows = [(f"{name}-{i}", f"{WORDS[o]} bio", o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture
def splits():
    return {"train": _split("train", 12), "dev": _split("dev", 3), "test": _split("test", 3)}


@pytest.fixture
def train_baseline(monkeypatch, tmp_path, splits):
    """Return run(): the script's main() on synthetic splits, giving the results directory."""
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


def test_train_baseline_saves_the_model_and_the_test_predictions_outside_the_repo(train_baseline):
    artifacts = train_baseline().parent / "artifacts"
    assert BaselineModel.load().min_df == 20
    predictions = pd.read_csv(artifacts / "predictions" / "baseline_test.csv")
    assert list(predictions.columns) == ["id", "predicted_occupation"]
    # The synthetic test split is 6 nurse bios then 6 surgeon bios, all easy.
    assert predictions["id"].tolist() == [f"test-{i}" for i in range(12)]
    assert predictions["predicted_occupation"].tolist() == ["nurse"] * 6 + ["surgeon"] * 6


def test_ties_on_dev_go_to_the_smaller_C_and_the_larger_min_df(train_baseline):
    metrics = json.loads((train_baseline() / "metrics_baseline.json").read_text())
    assert {row["dev_macro_f1"] for row in metrics["dev_grid"]} == {1.0}
    assert metrics["hyperparameters"] == {"C": 4, "min_df": 20}


def test_overfitting_check_is_train_macro_f1_minus_test_macro_f1(train_baseline, splits):
    # Swap the text of one nurse and one surgeon in test: per occupation TP 5, FP 1, FN 1, so
    # F1 = 10 / 12 and the test macro-F1 is 5/6. The model fits train perfectly: gap 1/6.
    # Dev has two swaps per occupation, so reading train off dev would give 2/3 instead of 1.
    test = splits["test"]
    test.loc[[0, 6], "text"] = ["scalpel theatre bio", "ward patient bio"]
    splits["dev"].loc[[0, 1, 6, 7], "text"] = ["scalpel theatre bio"] * 2 + ["ward patient bio"] * 2
    check = json.loads((train_baseline() / "metrics_baseline.json").read_text())[
        "overfitting_check"
    ]
    assert check == pytest.approx({"train_macro_f1": 1.0, "test_macro_f1": 5 / 6, "gap": 1 / 6})
