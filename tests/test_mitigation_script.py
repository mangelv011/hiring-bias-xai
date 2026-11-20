import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data, results
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

    def __init__(self, name):
        self.name = name
        self.network = self

    def to(self, device):
        return self

    def predict(self, texts):
        wrong = {"she", "born"} if self.name == "deberta" else {"born"}
        return [WRONG if wrong & set(text.split()) else text.split()[0] for text in texts]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """env(): the script module on synthetic splits and a fake model, with the results directory
    that holds the showcase files and the proxy-word list. `env.module.main(...)` runs it."""
    FakeDeberta.loads = []
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
        "id\n" + "".join(f"test-{i}\n" for i in range(12))
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


@pytest.fixture
def report(env):
    for condition in "abcde":
        env("predict", condition)
    # The predictions of the main run, scored in batches of 4: one bio differs from condition a.
    saved = pd.read_csv(
        Path(env.results_dir).parent / "artifacts" / "predictions" / "mitigation_a_test.csv"
    )
    saved.loc[3, "predicted_occupation"] = (
        WRONG if saved.loc[3, "predicted_occupation"] != WRONG else "surgeon"
    )
    saved.to_csv(
        Path(env.results_dir).parent / "artifacts" / "predictions" / "deberta_test.csv", index=False
    )
    env("report")
    return json.loads((env.results_dir / "mitigation.json").read_text())


def test_every_condition_is_scored_on_the_test_bios_outside_the_showcase_subset(report):
    assert report["evaluation_set"] == {"test_bios": 120, "showcase_bios": 12, "bios_scored": 108}
    for condition in "abcde":
        assert report["conditions"][condition]["outside_showcase"]["n_bios"] == 108


def test_conditions_a_to_d_are_also_reported_on_the_full_test_split_and_e_is_not(report):
    for condition in "abcd":
        assert report["conditions"][condition]["full_test"]["n_bios"] == 120
    assert "full_test" not in report["conditions"]["e"]


def test_accuracy_is_the_share_of_right_predictions_on_the_bios_outside_the_showcase_subset(report):
    # Bios 12..119 have markers by index % 4: she (0), he (1), born (2), plain (3). The original
    # model errs on she and born: half of the 108 bios. The scrubbed texts of b, c and d drop she,
    # he; the models that still err on born get 27 of 108 wrong. Condition e deletes born.
    outside = lambda c: report["conditions"][c]["outside_showcase"]["accuracy"]["value"]  # noqa: E731
    assert outside("a") == pytest.approx(54 / 108)
    assert outside("b") == pytest.approx(81 / 108)
    assert outside("c") == pytest.approx(81 / 108)
    assert outside("d") == pytest.approx(81 / 108)
    assert outside("e") == pytest.approx(81 / 108)


def test_every_metric_has_a_bootstrap_interval_and_the_rms_gap_has_its_permutation_null(report):
    block = report["conditions"]["a"]["outside_showcase"]
    for name in ("accuracy", "macro_f1", "rms_tpr_gap"):
        assert block[name]["ci_low"] <= block[name]["value"] <= block[name]["ci_high"]
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
    assert change["accuracy"]["value"] == pytest.approx(27 / 108)
    assert (
        change["accuracy"]["ci_low"] <= change["accuracy"]["value"] <= change["accuracy"]["ci_high"]
    )
    assert set(change) == {"accuracy", "macro_f1", "rms_tpr_gap"}


def test_the_scoring_check_compares_condition_a_with_the_predictions_of_the_main_run(report):
    assert report["scoring_check"]["agreement_with_main_run"] == pytest.approx(119 / 120)


def test_the_file_records_the_proxy_words_the_share_of_bios_they_touch_and_the_limitation(report):
    assert report["proxy_words"] == PROXY
    # One bio in four has the marker "born", one in four "she" or "he" each.
    assert report["bios_changed"]["gendered_word_share"] == pytest.approx(0.5, abs=0.01)
    assert report["bios_changed"]["proxy_word_share"] == pytest.approx(0.25, abs=0.01)
    assert "one training seed" in report["limitation"].lower()
    assert report["settings"]["n_bootstrap"] == 40
    assert report["settings"]["scoring_batch_size"] == 64


def test_the_report_names_the_conditions_that_have_no_predictions_yet(env):
    env("predict", "a")
    with pytest.raises(FileNotFoundError, match="b, c, d, e"):
        env("report")
