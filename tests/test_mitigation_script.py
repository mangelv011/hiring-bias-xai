import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import config, data, fairness, gap_statistics, results
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mitigation.py"
MALE_DOMINATED = ["surgeon", "composer", "software_engineer"]
FEMALE_DOMINATED = ["dietitian", "nurse", "model"]
SHOWCASE = MALE_DOMINATED + FEMALE_DOMINATED
MARKERS = ["she", "he", "born", "plain"]
PROXY = ["born"]
WRONG = "poet"


def _split(name):
    """10 bios in each of 6 occupations x 2 genders; the text is the occupation and a marker."""
    cells = [(o, g) for o in SHOWCASE for g in ("female", "male") for _ in range(10)]
    rows = [(f"{name}-{i}", f"{o} {MARKERS[i % 4]}", o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


class FakeDeberta:
    """A model that reads the first word as the occupation and errs on some markers: the original
    model on `she` and `born`, the scrubbed model on `born` only."""

    loads = []
    devices = []
    texts = []

    def __init__(self, name):
        self.name = name
        self.network = self

    def to(self, device):
        FakeDeberta.devices.append(device)
        return self

    def predict(self, texts):
        FakeDeberta.texts = list(texts)
        wrong = {"she", "born"} if self.name == "deberta" else {"born"}
        return [WRONG if wrong & set(text.split()) else text.split()[0] for text in texts]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """env(): the script module on synthetic splits and a fake model, with the results directory
    that holds the showcase files and the proxy-word list. `env.module.main(...)` runs it."""
    FakeDeberta.loads, FakeDeberta.devices, FakeDeberta.texts = [], [], []
    test = _split("test")
    scrubbed = test.assign(text=test["text"].str.replace(r" (she|he)$", "", regex=True))

    def load_split(split, variant="original"):
        if split == "train":
            return _split("train")
        return scrubbed if variant == "scrubbed" else test

    def load(cls, batch_size=32, name="deberta"):
        FakeDeberta.loads.append((name, batch_size))
        return FakeDeberta(name)

    monkeypatch.setattr(data, "load_split", load_split)
    monkeypatch.setattr(DebertaModel, "load", classmethod(load))
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_text(
        "id" + "".join(f"{chr(10)}test-{i}" for i in range(0, 120, 10)) + chr(10)
    )
    names = lambda names: [{"occupation": n, "share_women": 0.5} for n in names]  # noqa: E731
    results.write_json(
        results_dir / "showcase_occupations.json",
        {"male_dominated": names(MALE_DOMINATED), "female_dominated": names(FEMALE_DOMINATED)},
    )
    results.write_json(results_dir / "proxy_words.json", {"words": PROXY})
    spec = importlib.util.spec_from_file_location("mitigation_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEVICE = "cpu"
    module.N_BOOT = module.N_PERM = 40

    def go(*argv):
        module.main(results_dir, list(argv))
        return results_dir

    go.module, go.test, go.results_dir = module, test, results_dir
    return go


def _saved(env, condition):
    frame = pd.read_csv(
        Path(env.results_dir).parent
        / "artifacts"
        / "predictions"
        / f"mitigation_{condition}_test.csv"
    )
    assert frame["id"].tolist() == env.test["id"].tolist()
    return frame["predicted_occupation"].tolist()


def _expected(env, wrong_markers, text_without=()):
    return [
        WRONG
        if any(m in row.text.split() and m not in text_without for m in wrong_markers)
        else row.occupation
        for row in env.test.itertuples()
    ]


def test_the_original_model_scores_the_raw_test_bios_for_condition_a(env):
    env("predict", "a")
    assert _saved(env, "a") == _expected(env, {"she", "born"})
    assert FakeDeberta.loads == [("deberta", 64)]


def test_condition_b_scores_the_scrubbed_test_bios_with_the_original_model(env):
    env("predict", "b")
    assert _saved(env, "b") == _expected(env, {"born"})


def test_condition_c_scores_the_raw_test_bios_with_the_scrubbed_model(env):
    env("predict", "c")
    assert _saved(env, "c") == _expected(env, {"born"})
    assert FakeDeberta.loads == [("deberta-scrubbed", 64)]


def test_condition_d_scores_the_scrubbed_test_bios_with_the_scrubbed_model(env):
    env("predict", "d")
    assert _saved(env, "d") == _expected(env, {"born"})
    assert FakeDeberta.loads == [("deberta-scrubbed", 64)]


def test_condition_e_deletes_the_proxy_words_from_the_raw_bios_for_the_original_model(env):
    env("predict", "e")
    # "born" is gone, so only the bios with "she" are wrong.
    assert _saved(env, "e") == _expected(env, {"she"})
    assert FakeDeberta.loads == [("deberta", 64)]


def test_the_scoring_batch_is_64_bios(env):
    assert env.module.EVAL_BATCH_SIZE == 64


def test_a_condition_whose_predictions_are_complete_is_not_scored_again(env):
    env("predict", "a")
    env("predict", "a")
    assert len(FakeDeberta.loads) == 1


def test_a_condition_with_predictions_for_only_some_bios_is_scored_again(env):
    env("predict", "a")
    path = Path(env.results_dir).parent / "artifacts" / "predictions" / "mitigation_a_test.csv"
    frame = pd.read_csv(path)
    frame.head(50).to_csv(path, index=False)
    env("predict", "a")
    assert len(FakeDeberta.loads) == 2
    assert len(pd.read_csv(path)) == len(env.test)


def test_the_script_refuses_an_unknown_command_or_condition(env):
    with pytest.raises(ValueError, match="condition"):
        env("predict", "z")
    with pytest.raises(ValueError, match="predict"):
        env("score", "a")


def _predict_everything(env):
    """Run every condition and save the predictions of the main run, which differ from condition
    a on one bio (scored in batches of 4 rather than 64)."""
    for condition in "abcde":
        env("predict", condition)
    folder = Path(env.results_dir).parent / "artifacts" / "predictions"
    saved = pd.read_csv(folder / "mitigation_a_test.csv")
    saved.loc[3, "predicted_occupation"] = (
        WRONG if saved.loc[3, "predicted_occupation"] != WRONG else "surgeon"
    )
    saved.to_csv(folder / "deberta_test.csv", index=False)


@pytest.fixture
def report(env):
    _predict_everything(env)
    env("report")
    return json.loads((env.results_dir / "mitigation.json").read_text())


def test_every_condition_is_scored_on_the_test_bios_outside_the_showcase_subset(report):
    assert report["evaluation_set"] == {
        "test_bios": 120,
        "showcase_bios": 12,
        "bios_scored": 108,
        "occupations_left_out_of_gap": [],
    }
    for condition in "abcde":
        assert report["conditions"][condition]["outside_showcase"]["n_bios"] == 108


def test_conditions_a_to_d_are_also_reported_on_the_full_test_split_and_e_is_not(report):
    for condition in "abcd":
        assert report["conditions"][condition]["full_test"]["n_bios"] == 120
    assert "full_test" not in report["conditions"]["e"]


def test_accuracy_is_the_share_of_right_predictions_on_the_bios_outside_the_showcase_subset(report):
    # Markers are index % 4: she (0), he (1), born (2), plain (3). The showcase subset is every
    # tenth bio: 6 with she and 6 with born. The 108 bios left have 24 she, 30 he, 24 born and 30
    # plain. The original model errs on she and born (48 bios); the others on one of the two only
    # (24 bios).
    outside = lambda c: report["conditions"][c]["outside_showcase"]["accuracy"]["value"]  # noqa: E731
    assert outside("a") == pytest.approx(60 / 108)
    assert outside("b") == pytest.approx(84 / 108)
    assert outside("c") == pytest.approx(84 / 108)
    assert outside("d") == pytest.approx(84 / 108)
    assert outside("e") == pytest.approx(84 / 108)


def test_every_metric_has_a_bootstrap_interval_and_the_rms_gap_has_its_permutation_null(report):
    block = report["conditions"]["a"]["outside_showcase"]
    for name in ("accuracy", "macro_f1"):
        assert block[name]["ci_low"] <= block[name]["value"] <= block[name]["ci_high"]
    # The RMS gap is never negative, so its resamples sit above a point estimate of 0.
    assert block["rms_tpr_gap"]["ci_low"] <= block["rms_tpr_gap"]["ci_high"]
    null = block["rms_tpr_gap"]["null"]
    assert set(null) == {"mean", "p95", "p_value", "n_permutations"}
    assert null["n_permutations"] == 40


def test_the_tpr_gap_of_each_showcase_occupation_has_an_interval(report):
    gaps = report["conditions"]["a"]["outside_showcase"]["tpr_gap"]
    assert sorted(gaps) == sorted(SHOWCASE)
    assert set(gaps["surgeon"]) == {"value", "ci_low", "ci_high"}


def test_each_condition_but_a_gives_its_change_from_a_on_the_same_bios_with_an_interval(report):
    assert "change_from_a" not in report["conditions"]["a"]["outside_showcase"]
    change = report["conditions"]["b"]["outside_showcase"]["change_from_a"]
    assert change["accuracy"]["value"] == pytest.approx(24 / 108)
    assert (
        change["accuracy"]["ci_low"] <= change["accuracy"]["value"] <= change["accuracy"]["ci_high"]
    )
    assert set(change) == {"accuracy", "macro_f1", "rms_tpr_gap"}


def test_the_scoring_check_compares_condition_a_with_the_predictions_of_the_main_run(report):
    assert report["scoring_check"]["agreement_with_main_run"] == pytest.approx(119 / 120)


def test_the_file_records_the_proxy_words_the_share_of_bios_they_touch_and_the_limitation(report):
    assert report["proxy_words"] == PROXY
    # Of the 108 bios scored, 54 have she or he and 24 have born.
    assert report["bios_changed"]["gendered_word_share"] == pytest.approx(54 / 108)
    assert report["bios_changed"]["proxy_word_share"] == pytest.approx(24 / 108)
    assert "one training seed" in report["limitation"].lower()
    assert report["settings"]["n_bootstrap"] == 40
    assert report["settings"]["scoring_batch_size"] == 64


def test_the_report_names_the_conditions_that_have_no_predictions_yet(env):
    env("predict", "a")
    with pytest.raises(FileNotFoundError, match=r"conditions: b, c, d, e$"):
        env("report")


def test_an_occupation_with_no_man_or_no_woman_outside_the_showcase_subset_is_left_out_of_the_gap(
    env,
):
    # Like dietitian, whose 71 men all sit in the showcase subset. Here the showcase subset holds
    # every male surgeon bio (test-10 to test-19) and test-0. The 109 bios left still count for
    # accuracy and macro-F1; the RMS gap and the gaps per occupation use the occupations with
    # both genders.
    _predict_everything(env)
    ids = [0, *range(10, 20)]
    (env.results_dir / "showcase_subset_ids.csv").write_text(
        "id" + "".join(f"{chr(10)}test-{i}" for i in ids) + chr(10)
    )
    env("report")
    out = json.loads((env.results_dir / "mitigation.json").read_text())
    assert out["evaluation_set"]["occupations_left_out_of_gap"] == ["surgeon"]
    block = out["conditions"]["a"]["outside_showcase"]
    assert block["n_bios"] == 109
    wrong = sum(
        any(m in row.text.split() for m in ("she", "born"))
        for row in env.test.itertuples()
        if row.id not in {f"test-{i}" for i in ids}
    )
    assert block["accuracy"]["value"] == pytest.approx(1 - wrong / 109)
    assert block["tpr_gap"]["surgeon"] is None
    assert block["tpr_gap"]["nurse"]["value"] is not None
    change = out["conditions"]["b"]["outside_showcase"]["change_from_a"]["rms_tpr_gap"]
    b = out["conditions"]["b"]["outside_showcase"]["rms_tpr_gap"]["value"]
    assert b != block["rms_tpr_gap"]["value"]
    assert change["value"] == pytest.approx(b - block["rms_tpr_gap"]["value"])
    for key in ("value", "ci_low", "ci_high"):
        assert math.isfinite(block["rms_tpr_gap"][key])
        assert math.isfinite(change[key])


def test_each_condition_loads_the_model_and_texts_of_its_definition(env):
    # `she` and `he` are in the raw bios only; `born` is in the raw and the scrubbed bios.
    seen = {}
    for condition in "abcde":
        env("predict", condition)
        seen[condition] = (
            FakeDeberta.loads[-1][0],
            any("she" in t.split() for t in FakeDeberta.texts),
        )
    assert seen == {
        "a": ("deberta", True),
        "b": ("deberta", False),
        "c": ("deberta-scrubbed", True),
        "d": ("deberta-scrubbed", False),
        "e": ("deberta", True),
    }


def test_the_network_is_moved_to_the_device_of_the_script(env):
    env.module.DEVICE = "cuda:0"
    env("predict", "a")
    assert FakeDeberta.devices == ["cuda:0"]


def test_the_defaults_are_a_thousand_resamples_and_a_thousand_permutations(env):
    spec = importlib.util.spec_from_file_location("mitigation_defaults", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (module.N_BOOT, module.N_PERM) == (1000, 1000)


def test_the_settings_and_the_conditions_are_recorded_as_run(env):
    _predict_everything(env)
    env.module.N_PERM = 25
    env("report")
    out = json.loads((env.results_dir / "mitigation.json").read_text())
    assert out["settings"] == {
        "n_bootstrap": 40,
        "n_permutations": 25,
        "seed": config.SEED,
        "scoring_batch_size": 64,
    }
    assert out["conditions"]["a"]["outside_showcase"]["rms_tpr_gap"]["null"]["n_permutations"] == 25
    described = {c: (v["model"], v["text"]) for c, v in out["conditions"].items()}
    assert described == {
        "a": ("deberta", "original"),
        "b": ("deberta", "scrubbed"),
        "c": ("deberta-scrubbed", "original"),
        "d": ("deberta-scrubbed", "scrubbed"),
        "e": ("deberta", "proxy words deleted"),
    }


def _close(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-12, nan_ok=True)


def _same_summary(actual, expected):
    for key in ("value", "ci_low", "ci_high"):
        _close(actual[key], expected[key])


@pytest.mark.parametrize("showcase", [range(0, 120, 10), [0, *range(10, 20)]])
def test_every_block_is_what_the_metrics_give_on_the_bios_outside_the_showcase_subset(
    env, showcase
):
    # With the second subset surgeon has no man left outside it, so the gaps leave it out.
    _predict_everything(env)
    (env.results_dir / "showcase_subset_ids.csv").write_text(
        "id" + "".join(f"{chr(10)}test-{i}" for i in showcase) + chr(10)
    )
    env("report")
    out = json.loads((env.results_dir / "mitigation.json").read_text())["conditions"]
    outside = ~env.test["id"].isin({f"test-{i}" for i in showcase})
    rest = env.test[outside]
    share_women = data.share_women(_split("train"))
    n = 40

    def predicted(condition):
        return results.load_predictions(f"mitigation_{condition}", env.test)[outside]

    def gender_gaps(frame, pred):
        both = frame.groupby("occupation")["gender"].transform("nunique") == 2
        return frame[both], pred[both]

    for condition in "abcde":
        block = out[condition]["outside_showcase"]
        pred = predicted(condition)
        every = fairness.evaluate(rest["occupation"], pred, rest["gender"], share_women, n_boot=n)
        gap_rest, gap_pred = gender_gaps(rest, pred)
        gaps = fairness.evaluate(
            gap_rest["occupation"], gap_pred, gap_rest["gender"], share_women, n_boot=n
        )
        _same_summary(block["macro_f1"], every["macro_f1"])
        _same_summary(block["accuracy"], every["accuracy"])
        _same_summary(block["rms_tpr_gap"], gaps["rms_tpr_gap"])
        for occupation, entry in block["tpr_gap"].items():
            if entry is not None:
                _same_summary(entry, gaps["per_occupation"][occupation]["tpr_gap"])
        null = gap_statistics.rms_permutation_test(
            gap_rest["occupation"], gap_pred, gap_rest["gender"], n_perm=n, seed=config.SEED
        )
        assert block["rms_tpr_gap"]["null"] == {
            "mean": null["null_mean"],
            "p95": null["null_p95"],
            "p_value": null["p_value"],
            "n_permutations": n,
        }
        if condition == "a":
            continue
        change = fairness.paired_difference(
            rest["occupation"], predicted("a"), pred, rest["gender"], share_women, n_boot=n
        )
        gap_change = fairness.paired_difference(
            gap_rest["occupation"],
            predicted("a")[gap_rest.index],
            gap_pred,
            gap_rest["gender"],
            share_women,
            n_boot=n,
        )
        _same_summary(block["change_from_a"]["accuracy"], change["accuracy"])
        _same_summary(block["change_from_a"]["macro_f1"], change["macro_f1"])
        _same_summary(block["change_from_a"]["rms_tpr_gap"], gap_change["rms_tpr_gap"])


def test_the_bios_changed_count_the_words_whatever_their_case_and_punctuation(env, monkeypatch):
    shouted = env.test.assign(text=env.test["text"].str.replace("born", "Born,"))
    split = data.load_split
    monkeypatch.setattr(
        data,
        "load_split",
        lambda name, variant="original": (
            shouted if (name, variant) == ("test", "original") else split(name, variant)
        ),
    )
    _predict_everything(env)
    env("report")
    out = json.loads((env.results_dir / "mitigation.json").read_text())
    assert out["bios_changed"]["proxy_word_share"] == pytest.approx(24 / 108)


def test_the_script_reads_its_command_from_the_command_line_by_default(env, monkeypatch):
    monkeypatch.setattr("sys.argv", ["mitigation.py", "predict", "a"])
    env.module.main(env.results_dir)
    assert len(FakeDeberta.loads) == 1


@pytest.mark.parametrize("argv", [("predict", "a", "b"), ("report", "now"), ("predict",), ()])
def test_the_script_refuses_extra_or_missing_arguments(env, argv):
    with pytest.raises(ValueError, match="usage"):
        env(*argv)
