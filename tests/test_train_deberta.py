import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_deberta.py"

# Two easy occupations: every nurse bio says "alpha" and every surgeon bio "gamma".
WORDS = {"nurse": "alpha alpha alpha", "surgeon": "gamma gamma gamma"}


def _split(name, per_cell):
    cells = [(o, g) for o in WORDS for g in ("female", "male") for _ in range(per_cell)]
    rows = [(f"{name}-{i}", WORDS[o], o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture(scope="module")
def execute(tmp_path_factory, tiny_network, tiny_tokenizer):
    """Return execute(**constants): the script's main() on synthetic splits and a tiny network.

    The result is the directory that holds `results/` and `artifacts/`.
    """

    def execute(**constants):
        root = tmp_path_factory.mktemp("deberta")
        splits = {"train": _split("train", 8), "dev": _split("dev", 2), "test": _split("test", 2)}
        splits["test"].loc[2, "occupation"] = "surgeon"  # a male nurse bio the model gets wrong

        def pretrained(cls, max_length, batch_size=32):
            return cls(tiny_network(), tiny_tokenizer(max_length), batch_size)

        defaults = {
            "LEARNING_RATES": (3e-3, 1e-5),
            "EPOCHS": 30,
            "BATCH_SIZE": 4,
            "ACCUMULATION": 1,
            "EVAL_EVERY": 40,
            "PATIENCE": 100,
            "MAX_LENGTH": 64,
        }
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(data, "load_split", lambda split, variant="original": splits[split])
            patch.setattr(DebertaModel, "pretrained", classmethod(pretrained))
            patch.setenv("HBX_ARTIFACTS", str(root / "artifacts"))
            spec = importlib.util.spec_from_file_location("train_deberta", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, value in (defaults | constants).items():
                patch.setattr(module, name, value)
            module.main(root / "results")
        return root

    return execute


@pytest.fixture(scope="module")
def run(execute):
    return execute()


def _metrics(run):
    return json.loads((run / "results" / "metrics_deberta.json").read_text())


def test_the_learning_rate_with_the_best_dev_macro_f1_is_chosen(run):
    # 1e-5 barely moves the toy network in 240 steps; 3e-3 learns the task. The better rate
    # comes first, so a script that kept the last model would fail the next test.
    assert _metrics(run)["hyperparameters"]["learning_rate"] == 3e-3


def test_the_saved_checkpoint_is_the_one_of_the_chosen_learning_rate(run, monkeypatch):
    monkeypatch.setenv("HBX_ARTIFACTS", str(run / "artifacts"))
    probabilities = DebertaModel.load().predict_proba(list(WORDS.values()))
    assert [data.OCCUPATIONS[i] for i in probabilities.argmax(axis=1)] == list(WORDS)


def test_the_training_log_has_one_row_per_evaluation_of_every_run(run):
    # 32 train bios in batches of 4 make 8 steps per epoch, 240 in 30 epochs; an evaluation
    # every 40 steps gives 6 rows per run, and the runs follow LEARNING_RATES.
    log = pd.read_csv(run / "results" / "training_log.csv")
    assert list(log.columns) == [
        "base_learning_rate",
        "step",
        "epoch",
        "train_loss",
        "dev_loss",
        "dev_macro_f1",
        "learning_rate",
    ]
    assert log["base_learning_rate"].tolist() == [3e-3] * 6 + [1e-5] * 6
    assert log["step"].tolist() == [40, 80, 120, 160, 200, 240] * 2


def test_the_test_report_scores_the_chosen_model_once_with_the_fairness_module(run):
    # The toy test split is 4 nurse bios then 4 surgeon bios, all easy, but bio 2 is labelled
    # surgeon: 7 of 8 right. Nurse: 3 hits, 4 predicted, 3 true, F1 = 6/7. Surgeon: 4 hits,
    # 4 predicted, 5 true, F1 = 8/9.
    test = _metrics(run)["test"]
    assert test["macro_f1"]["value"] == pytest.approx((6 / 7 + 8 / 9) / 2)
    assert test["accuracy"]["value"] == 7 / 8
    assert sorted(test["per_occupation"]) == ["nurse", "surgeon"]
    assert test["per_occupation"]["nurse"]["n_female"] == 2


def test_the_test_predictions_are_saved_outside_the_repo(run):
    predictions = pd.read_csv(run / "artifacts" / "predictions" / "deberta_test.csv")
    assert list(predictions.columns) == ["id", "predicted_occupation"]
    assert predictions["id"].tolist() == [f"test-{i}" for i in range(8)]
    assert predictions["predicted_occupation"].tolist() == ["nurse"] * 4 + ["surgeon"] * 4


def test_the_overfitting_check_compares_train_and_test_macro_f1(run):
    # The toy network gets every train bio right (F1 1) and the test macro-F1 is (6/7 + 8/9) / 2.
    check = _metrics(run)["overfitting_check"]
    test_f1 = (6 / 7 + 8 / 9) / 2
    assert check["train_macro_f1"] == 1.0
    assert check["test_macro_f1"] == pytest.approx(test_f1)
    assert check["gap"] == pytest.approx(1 - test_f1)


def test_max_length_report_gives_the_untruncated_token_lengths_of_the_training_bios(run):
    # Every toy bio is three words plus [CLS] and [SEP]: 5 tokens, far below 64.
    report = _metrics(run)["max_length"]
    assert report["value"] == 64
    assert report["token_length_percentiles"] == {
        "p50": 5,
        "p90": 5,
        "p95": 5,
        "p99": 5,
        "p99.9": 5,
        "max": 5,
    }
    assert report["share_above"] == {"128": 0.0, "256": 0.0, "512": 0.0}
    assert report["share_truncated"] == 0.0


def test_max_length_report_counts_the_bios_longer_than_max_length_as_truncated(execute):
    # With max_length 4 every 5-token bio is cut, yet the lengths are measured before cutting.
    report = json.loads(
        (
            execute(LEARNING_RATES=(1e-3,), EPOCHS=1, MAX_LENGTH=4)
            / "results"
            / "metrics_deberta.json"
        ).read_text()
    )["max_length"]
    assert report["share_truncated"] == 1.0
    assert report["token_length_percentiles"]["max"] == 5


def test_every_run_records_its_best_dev_evaluation(run):
    runs = _metrics(run)["runs"]
    log = pd.read_csv(run / "results" / "training_log.csv")
    assert [r["learning_rate"] for r in runs] == [3e-3, 1e-5]
    for entry in runs:
        rows = log[log["base_learning_rate"] == entry["learning_rate"]]
        best = rows.loc[rows["dev_macro_f1"].idxmax()]  # the first row of the highest F1
        assert entry["best_dev_macro_f1"] == best["dev_macro_f1"]
        assert (entry["best_step"], entry["best_epoch"]) == (best["step"], best["epoch"])
        assert entry["evaluations"] == 6
    assert runs[0]["best_dev_macro_f1"] == 1.0
    assert runs[1]["best_dev_macro_f1"] < 1.0
