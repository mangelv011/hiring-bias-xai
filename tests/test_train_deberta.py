import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from hiring_bias_xai import data
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_deberta.py"

# Two easy occupations: every nurse bio says "alpha" and every surgeon bio "gamma".
WORDS = {"nurse": "alpha alpha alpha", "surgeon": "gamma gamma gamma"}


def _load_script():
    spec = importlib.util.spec_from_file_location("train_deberta", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(name, per_cell):
    cells = [(o, g) for o in WORDS for g in ("female", "male") for _ in range(per_cell)]
    rows = [(f"{name}-{i}", WORDS[o], o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture(scope="module")
def execute(tmp_path_factory, tiny_network, tiny_tokenizer):
    """Return execute(**constants): the script's main() on synthetic splits and a tiny network.

    The result is the directory that holds `results/` and `artifacts/`.
    """

    draws = []

    def execute(**constants):
        root = tmp_path_factory.mktemp("deberta")
        splits = {"train": _split("train", 8), "dev": _split("dev", 2), "test": _split("test", 2)}
        # Train shares of women: nurse 4 of 16 (0.25), surgeon 12 of 16 (0.75). In the test
        # split nurse has the higher share, so a script that took the shares from test would
        # get the opposite correlation.
        splits["train"].loc[0:3, "gender"] = "male"
        splits["train"].loc[24:27, "gender"] = "female"
        splits["test"].loc[2, "occupation"] = "surgeon"  # a male nurse bio the model gets wrong
        # The dev split has a different F1 and longer bios than the train split, so a script
        # that mixed the two up would report other numbers.
        splits["dev"].loc[0, "occupation"] = "surgeon"
        splits["dev"]["text"] = splits["dev"]["text"] + " " + splits["dev"]["text"].str[:5]

        def pretrained(cls, max_length, batch_size=32):
            draws.append(torch.rand(1).item())
            return cls(tiny_network(), tiny_tokenizer(max_length), batch_size)

        defaults = {
            "LEARNING_RATES": (3e-3, 1e-5),
            "EPOCHS": 30,
            "BATCH_SIZE": 4,
            "ACCUMULATION": 1,
            "EVAL_EVERY": 40,
            "MAX_LENGTH": 64,
            "DEVICE": "cpu",
        }
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(data, "load_split", lambda split, variant="original": splits[split])
            patch.setattr(DebertaModel, "pretrained", classmethod(pretrained))
            patch.setenv("HBX_ARTIFACTS", str(root / "artifacts"))
            module = _load_script()
            for name, value in (defaults | constants).items():
                patch.setattr(module, name, value)
            module.main(root / "results")
        return root

    execute.draws = draws
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
    # The learned toy model gets 7 of 8 dev bios right (dev bio 0 is labelled surgeon): nurse
    # F1 6/7, surgeon F1 8/9.
    assert runs[0]["best_dev_macro_f1"] == pytest.approx((6 / 7 + 8 / 9) / 2)
    assert runs[1]["best_dev_macro_f1"] < runs[0]["best_dev_macro_f1"]


def test_the_hyperparameters_name_the_chosen_run_and_every_fixed_setting(run):
    chosen = _metrics(run)["runs"][0]
    assert _metrics(run)["hyperparameters"] == {
        "learning_rate": 3e-3,
        "best_step": chosen["best_step"],
        "max_epochs": 30,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "effective_batch_size": 4,
        "warmup_ratio": 0.06,
        "evaluation_every_steps": 40,
        "early_stopping": False,
        "max_length": 64,
        "seed": 42,
        "precision": "bfloat16 autocast on CUDA, float32 otherwise",
    }


def test_training_seconds_add_up_the_wall_clock_time_of_the_runs(run):
    metrics = _metrics(run)
    seconds = [entry["training_seconds"] for entry in metrics["runs"]]
    assert all(0 < value < 600 for value in seconds)
    assert metrics["training_seconds"] == pytest.approx(sum(seconds), abs=0.1)


def test_peak_gpu_memory_is_null_when_training_on_the_cpu(run):
    assert _metrics(run)["peak_gpu_memory_mb"] is None


@pytest.mark.gpu
def test_peak_gpu_memory_is_the_most_memory_torch_allocated_during_training(execute):
    root = execute(DEVICE="cuda", LEARNING_RATES=(3e-3,), EPOCHS=1)
    peak = json.loads((root / "results" / "metrics_deberta.json").read_text())["peak_gpu_memory_mb"]
    # The tiny network holds about 0.2 MB of weights, plus gradients and Adam state.
    assert 0.5 < peak < 100


def test_the_comparison_with_the_baseline_says_whether_the_confidence_intervals_overlap(
    execute, tmp_path
):
    # Baseline macro-F1 0.005 [0.0, 0.01] is far below the toy model's (6/7 + 8/9) / 2, whose
    # interval stays above 0.01: no overlap. Baseline RMS TPR gap 0.5 [0.0, 1.0] covers any
    # value: overlap.
    macro_f1 = {"value": 0.005, "ci_low": 0.0, "ci_high": 0.01}
    rms_tpr_gap = {"value": 0.5, "ci_low": 0.0, "ci_high": 1.0}
    path = tmp_path / "metrics_baseline.json"
    path.write_text(json.dumps({"test": {"macro_f1": macro_f1, "rms_tpr_gap": rms_tpr_gap}}))
    root = execute(BASELINE_METRICS=path, LEARNING_RATES=(3e-3,))
    metrics = json.loads((root / "results" / "metrics_deberta.json").read_text())
    comparison = metrics["comparison_with_baseline"]
    assert comparison["macro_f1"]["baseline"] == macro_f1
    assert comparison["macro_f1"]["deberta"] == metrics["test"]["macro_f1"]
    assert comparison["macro_f1"]["cis_overlap"] is False
    assert comparison["rms_tpr_gap"]["baseline"] == rms_tpr_gap
    assert comparison["rms_tpr_gap"]["deberta"] == metrics["test"]["rms_tpr_gap"]
    assert comparison["rms_tpr_gap"]["cis_overlap"] is True


def test_the_script_prints_one_progress_line_per_evaluation(execute, capsys):
    # One epoch is 8 steps and the evaluation interval is 40, so only the last step is evaluated.
    execute(LEARNING_RATES=(3e-3,), EPOCHS=1)
    lines = [line for line in capsys.readouterr().out.splitlines() if "dev_macro_f1" in line]
    assert len(lines) == 1
    assert "lr=0.003" in lines[0] and "step=8" in lines[0] and "epoch=1" in lines[0]


def test_the_gap_share_correlation_uses_the_share_of_women_in_the_training_subsample(run):
    # TPR gaps (women minus men): nurse 1 - 1 = 0; surgeon 1 - 2/3 = 1/3, because the male
    # surgeon bio 2 is a miss. With training shares 0.25 and 0.75 the gap rises with the share:
    # correlation +1 (two points).
    test = _metrics(run)["test"]
    assert test["per_occupation"]["nurse"]["tpr_gap"]["value"] == 0.0
    assert test["per_occupation"]["surgeon"]["tpr_gap"]["value"] == pytest.approx(1 / 3)
    assert test["gap_share_pearson"]["value"] == pytest.approx(1.0)


def test_length_report_percentiles_follow_numpy_linear_interpolation(tiny_tokenizer, monkeypatch):
    # Bios of 1 to 10 words have 3 to 12 tokens with [CLS] and [SEP]. Sorted, index 4.5 is
    # (7 + 8) / 2; index 8.1 is 11.1; 8.55 is 11.55; 8.91 is 11.91; 8.991 is 11.991.
    module = _load_script()
    monkeypatch.setattr(module, "MAX_LENGTH", 8)
    monkeypatch.setattr(
        DebertaModel, "pretrained", classmethod(lambda cls, n: cls(None, tiny_tokenizer(n)))
    )
    report = module._length_report(["alpha " * n for n in range(1, 11)])
    assert report["token_length_percentiles"] == pytest.approx(
        {"p50": 7.5, "p90": 11.1, "p95": 11.55, "p99": 11.91, "p99.9": 11.991, "max": 12}
    )
    # 9, 10, 11 and 12 tokens are longer than 8; a bio of exactly 8 tokens is not cut.
    assert report["share_truncated"] == 0.4


def test_confidence_intervals_that_touch_overlap():
    module = _load_script()

    def interval(low, high):
        return {"value": (low + high) / 2, "ci_low": low, "ci_high": high}

    def overlap(baseline, deberta):
        both = {"macro_f1": baseline, "rms_tpr_gap": baseline}
        return module._comparison({"macro_f1": deberta, "rms_tpr_gap": deberta}, both)["macro_f1"][
            "cis_overlap"
        ]

    assert overlap(interval(0.1, 0.5), interval(0.5, 0.9)) is True
    assert overlap(interval(0.5, 0.9), interval(0.1, 0.5)) is True
    assert overlap(interval(0.1, 0.4), interval(0.5, 0.9)) is False
    assert overlap(interval(0.5, 0.9), interval(0.1, 0.4)) is False


def test_the_effective_batch_is_batch_size_times_accumulation_and_both_reach_training(execute):
    # Batches of 2 bios, 2 per step: 4 bios per step, 8 steps per epoch of 32 bios. Step 40 is
    # then the end of epoch 5; without accumulation it would be epoch 3 (16 steps per epoch).
    root = execute(BATCH_SIZE=2, ACCUMULATION=2, LEARNING_RATES=(3e-3,))
    metrics = json.loads((root / "results" / "metrics_deberta.json").read_text())
    log = pd.read_csv(root / "results" / "training_log.csv")
    assert metrics["hyperparameters"]["effective_batch_size"] == 4
    assert log["epoch"].tolist() == [5, 10, 15, 20, 25, 30]


def test_every_epoch_runs_even_when_the_dev_macro_f1_stops_improving(execute):
    # No early stopping: a noisy dev macro-F1 stopped an earlier run after 0.6 of an epoch. The
    # 1e-9 toy run cannot learn, so its dev macro-F1 never improves, and it still logs all 6
    # evaluations.
    root = execute(LEARNING_RATES=(1e-9,))
    assert len(pd.read_csv(root / "results" / "training_log.csv")) == 6


def test_the_head_is_initialised_after_seeding_the_global_generators(execute):
    execute.draws.clear()
    execute(LEARNING_RATES=(3e-3, 1e-5), EPOCHS=1)
    expected = torch.rand(1, generator=torch.Generator().manual_seed(42)).item()
    # One draw for the length report, then one per run, each taken right after seeding.
    assert execute.draws[1:] == [expected, expected]


def test_share_above_counts_bios_with_more_than_n_tokens(tiny_tokenizer, monkeypatch):
    # 126 words are 128 tokens with [CLS] and [SEP]: not above 128. 127 words are 129: above.
    module = _load_script()
    monkeypatch.setattr(
        DebertaModel, "pretrained", classmethod(lambda cls, n: cls(None, tiny_tokenizer(n)))
    )
    report = module._length_report(["alpha " * 126, "alpha " * 127])
    assert report["share_above"] == {"128": 0.5, "256": 0.0, "512": 0.0}
