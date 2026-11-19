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
    default_n_jobs = getattr(module, "N_JOBS", None)
    module.N_JOBS = 1  # worker processes would not see the monkeypatched load_split

    def run():
        module.main(tmp_path / "results")
        return tmp_path / "results"

    run.module = module
    run.default_n_jobs = default_n_jobs
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
    assert 0 < metrics["runtime_seconds"] < 600


def test_train_baseline_saves_the_model_and_the_test_predictions_outside_the_repo(train_baseline):
    artifacts = train_baseline().parent / "artifacts"
    saved = BaselineModel.load()
    assert (saved.C, saved.min_df) == (4, 20)  # the selected model, not the last one fitted
    predictions = pd.read_csv(artifacts / "predictions" / "baseline_test.csv")
    assert list(predictions.columns) == ["id", "predicted_occupation"]
    # The synthetic test split is 6 nurse bios then 6 surgeon bios, all easy.
    assert predictions["id"].tolist() == [f"test-{i}" for i in range(12)]
    assert predictions["predicted_occupation"].tolist() == ["nurse"] * 6 + ["surgeon"] * 6


def test_ties_on_dev_go_to_the_smaller_C_and_the_larger_min_df(train_baseline):
    metrics = json.loads((train_baseline() / "metrics_baseline.json").read_text())
    assert {row["dev_macro_f1"] for row in metrics["dev_grid"]} == {1.0}
    assert [(row["C"], row["min_df"]) for row in metrics["dev_grid"]] == [
        (4, 5),
        (4, 20),
        (16, 5),
        (16, 20),
        (64, 5),
        (64, 20),
    ]
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


def test_hyperparameters_are_chosen_by_macro_f1_on_dev(train_baseline, splits):
    # Six surgeon bios of train say "scalpel theatre rare" (6 bios: kept by min_df=5, dropped by
    # min_df=20) and 20 more nurse bios make nurse the majority. Dev has 8 nurse bios and 6 surgeon
    # bios, three of them saying only "rare". With min_df=5 they are found: macro-F1 1. With
    # min_df=20 the text is empty and the model answers with the majority, nurse: nurse TP 8,
    # FP 3 -> F1 16/19; surgeon TP 3, FN 3 -> F1 6/9; macro-F1 (16/19 + 2/3) / 2 = 43/57
    # (weighted by size it would be 0.767).
    train, dev = splits["train"], splits["dev"]
    train.loc[train.index[train["occupation"] == "surgeon"][:6], "text"] = "scalpel theatre rare"
    nurses = _split("extra", 5).query("occupation == 'nurse'")
    splits["train"] = pd.concat([train, nurses.assign(id="extra")], ignore_index=True)
    dev.loc[dev.index[dev["occupation"] == "surgeon"][:3], "text"] = "rare"
    dev_nurses = _split("dev-extra", 1).query("occupation == 'nurse'")
    splits["dev"] = pd.concat([dev, dev_nurses.assign(id="dev-extra")], ignore_index=True)
    metrics = json.loads((train_baseline() / "metrics_baseline.json").read_text())
    assert metrics["hyperparameters"] == {"C": 4, "min_df": 5}
    by_min_df = {row["min_df"]: row["dev_macro_f1"] for row in metrics["dev_grid"]}
    assert by_min_df == pytest.approx({5: 1.0, 20: 43 / 57})


def test_test_metrics_come_from_the_test_split_with_the_training_shares_of_women(
    train_baseline, splits
):
    # Test bio 3 (a male nurse) and bio 6 (a female surgeon) get the other occupation's words, so
    # nurse TPR is 3/3 women and 2/3 men (gap +1/3) and surgeon 2/3 and 3/3 (gap -1/3): accuracy
    # 10/12, RMS gap 1/3. Train has 12 female and 4 male nurses (share 3/4) and 12 and 12 surgeons
    # (1/2), so the gap rises with the share: r = 1. Test shares (1/2, 1/2) would give NaN.
    train = splits["train"]
    splits["train"] = train.drop(
        index=train.index[(train["occupation"] == "nurse") & (train["gender"] == "male")][:8]
    )
    splits["dev"] = _split("dev", 2)  # a layout unlike test, so dev cannot stand in for it
    splits["test"].loc[[3, 6], "text"] = ["scalpel theatre bio", "ward patient bio"]
    test = json.loads((train_baseline() / "metrics_baseline.json").read_text())["test"]
    assert test["accuracy"]["value"] == pytest.approx(10 / 12)
    assert test["rms_tpr_gap"]["value"] == pytest.approx(1 / 3)
    assert test["gap_share_pearson"]["value"] == pytest.approx(1.0)


def test_the_grid_runs_in_parallel_and_the_results_do_not_depend_on_it(train_baseline):
    assert train_baseline.default_n_jobs > 1

    def metrics(n_jobs):
        train_baseline.module.N_JOBS = n_jobs
        result = json.loads((train_baseline() / "metrics_baseline.json").read_text())
        result.pop("runtime_seconds")
        return result

    assert metrics(2) == metrics(1)
