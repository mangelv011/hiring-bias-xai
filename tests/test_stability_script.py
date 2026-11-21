import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import BagOfWords

from hiring_bias_xai import attribution, data
from hiring_bias_xai import explanation_checks as checks

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stability.py"
NURSE = data.OCCUPATIONS.index("nurse")
WEIGHTS = {f"w{i}": float(i - 4) for i in range(16)}
VARIANTS = ("deberta", "deberta-seed7", "deberta-unk", "deberta-random")
SAMPLE = ["test-1", "test-3", "test-4"]


def _text(start, extra=None):
    words = [f"w{(start + i) % 16}" for i in range(16)]
    if extra:
        words[:2] = extra
    return " ".join(words)


BIOS = [(f"test-{i}", _text(i), "nurse") for i in range(6)]


class AnyAlpha(BagOfWords):
    """Nurse logit: the word weights plus 40 when the bio has alpha at all (a second alpha is
    redundant), so a word's leave-one-out value is its weight and alpha's is 0."""

    def score(self, text):
        logits = super().score(text)
        if "alpha" in text.split():
            logits[NURSE] += 40.0
        return logits


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(): the script's main() on six synthetic bios of 16 words.

    The saved attributions of the three runs are made by the linear fake, so each equals the word
    weights; `go.save(variant, classifier)` replaces one run. `go.fake` is the model that scores
    the leave-one-out values and can be replaced before go().
    """
    results = tmp_path / "results"
    results.mkdir()
    ids = "".join(f"{bio_id}\n" for bio_id, _, _ in BIOS)
    (results / "showcase_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    (results / "stability_sample_ids.csv").write_bytes(
        ("id\n" + "".join(f"{bio_id}\n" for bio_id in SAMPLE)).encode()
    )
    (results / "shapley_pilot.json").write_text(json.dumps({"inference_batch_size": 16}))
    test = pd.DataFrame(BIOS, columns=["id", "text", "occupation"]).assign(gender="female")
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": test)
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("stability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loads = []

    def go():
        monkeypatch.setattr(
            module, "_load", lambda batch_size: (loads.append(batch_size), go.fake)[1]
        )
        module.main(results)
        return json.loads((results / "stability.json").read_text())

    def save(variant, classifier, texts=None):
        for bio_id, text, occupation in BIOS:
            (result,) = attribution.shapley(
                classifier, [(texts or {}).get(bio_id, text)], [occupation], 3, "delete", 0
            )
            attribution.save(attribution.path(variant, bio_id), result)

    go.fake = BagOfWords({"nurse": WEIGHTS})
    go.save = save
    go.loads = loads
    go.module = module
    go.results = results
    for variant in VARIANTS:
        save(variant, BagOfWords({"nurse": WEIGHTS}))
    return go


def test_when_every_run_agrees_all_three_checks_pass_with_perfect_agreement(run):
    record = run()
    for name in ("seed", "removal", "method"):
        check = record["checks"][name]
        assert check["spearman"]["mean"] == pytest.approx(1.0)
        assert check["spearman"]["median"] == pytest.approx(1.0)
        assert check["top10_overlap"]["mean"] == 1.0
        assert check["top10_overlap"]["median"] == 1.0
        assert check["passed"] is True


def test_the_seed_and_removal_checks_use_the_sample_and_the_method_check_the_whole_subset(run):
    record = run()
    assert record["checks"]["seed"]["n_bios"] == 3
    assert record["checks"]["removal"]["n_bios"] == 3
    assert record["checks"]["method"]["n_bios"] == 6
    assert record["sample"]["n_bios"] == 3
    assert record["sample"]["ids_file"] == "stability_sample_ids.csv"


def test_the_model_is_loaded_with_the_inference_batch_size_of_the_pilot(run):
    run()
    assert run.loads == [16]


def test_a_failed_check_is_kept_and_reported_with_its_numbers(run):
    run.save("deberta-seed7", BagOfWords({"nurse": {w: -v for w, v in WEIGHTS.items()}}))
    record = run()
    seed = record["checks"]["seed"]
    assert seed["passed"] is False
    assert seed["spearman"]["median"] == pytest.approx(-1.0)
    assert record["checks"]["removal"]["passed"] is True


def test_the_intervals_contain_the_mean_and_the_median(run):
    run.save(
        "deberta-unk",
        BagOfWords({"nurse": {w: v * (1 + (i % 3)) for i, (w, v) in enumerate(WEIGHTS.items())}}),
    )
    check = run()["checks"]["removal"]
    for metric in ("spearman", "top10_overlap"):
        for statistic in ("mean", "median"):
            low, high = check[metric][f"{statistic}_ci"]
            assert low <= check[metric][statistic] <= high


def test_each_check_records_its_expectation_its_thresholds_and_its_rule(run):
    record = run()
    assert record["settings"]["top"] == 10
    assert record["settings"]["word_ranking"]
    thresholds = {}
    for name, check in record["checks"].items():
        assert check["expected"]
        assert check["rule"]
        thresholds[name] = (
            check["thresholds"]["median_spearman"],
            check["thresholds"]["median_top10_overlap"],
        )
    assert thresholds == {"seed": (0.9, 0.7), "removal": (0.8, 0.5), "method": (0.7, 0.5)}


def test_a_check_passes_only_when_both_medians_reach_their_thresholds(run):
    # Same order of the values, so the Spearman median is 1, but the top words change.
    scaled = {w: (100.0 if w in ("w0", "w1", "w2") else 1.0) * v for w, v in WEIGHTS.items()}
    run.save("deberta-seed7", BagOfWords({"nurse": scaled}))
    check = run()["checks"]["seed"]
    assert check["spearman"]["median"] < 1.0 or check["top10_overlap"]["median"] < 1.0
    assert check["passed"] == (
        check["spearman"]["median"] >= 0.9 and check["top10_overlap"]["median"] >= 0.7
    )


def test_the_method_check_compares_the_main_run_with_leave_one_out_on_the_true_occupation(run):
    run.fake = BagOfWords({"nurse": {w: -v for w, v in WEIGHTS.items()}})
    method = run()["checks"]["method"]
    assert method["spearman"]["median"] == pytest.approx(-1.0)
    assert method["passed"] is False


def test_a_word_repeated_in_a_bio_gets_credit_in_shapley_but_not_in_leave_one_out(run):
    texts = {"test-2": _text(2, extra=["alpha", "alpha"])}
    redundant = AnyAlpha({"nurse": WEIGHTS})
    run.save("deberta", redundant, texts)
    run.fake = redundant
    example = run()["disagreement_example"]
    assert example["bio_id"] == "test-2"
    assert example["word"] == "alpha"
    assert example["n_occurrences"] == 2
    assert example["spearman"] < 1.0
    values = [entry for entry in example["occurrences"]]
    assert [entry["position"] for entry in values] == [0, 1]
    assert [entry["loo"] for entry in values] == pytest.approx([0.0, 0.0], abs=1e-5)
    assert [entry["shapley"] for entry in values] == pytest.approx([20.0, 20.0], abs=1e-3)
    assert example["drop_all_occurrences"] == pytest.approx(40.0, abs=1e-3)
    assert len(example["top_words"]) == 8
    assert example["rule"]


def _pairs(spearman, overlap):
    return list(zip(spearman, overlap, strict=True))


def test_the_pass_rule_uses_medians_not_means(run):
    check = run.module._check
    # median 1, mean 0.2: passes on the median
    assert check("seed", _pairs([1, 1, 1, -1, -1], [1.0] * 5))["passed"] is True
    # median 0.79 under the 0.8 threshold, mean 0.87 over it: fails on the median
    assert check("removal", _pairs([0.79, 0.79, 0.79, 1, 1], [1.0] * 5))["passed"] is False
    # the same for the overlap: median 0.4 under 0.5, mean 0.64 over
    assert check("removal", _pairs([1.0] * 5, [0.4, 0.4, 0.4, 1, 1]))["passed"] is False
    assert check("removal", _pairs([1.0] * 5, [1, 1, 1, 0, 0]))["passed"] is True


def test_a_median_equal_to_its_threshold_passes(run):
    assert run.module._check("seed", _pairs([0.9] * 5, [0.7] * 5))["passed"] is True
    assert run.module._check("seed", _pairs([0.9] * 5, [0.6] * 5))["passed"] is False


def test_an_undefined_correlation_is_counted_and_left_out_not_read_as_zero(run):
    check = run.module._check("seed", _pairs([1.0, 1.0, np.nan], [1.0] * 3))
    assert check["spearman"]["n_undefined"] == 1
    assert check["spearman"]["mean"] == 1.0
    assert check["n_bios"] == 3


def test_a_run_made_on_other_words_than_the_main_run_stops_the_check(run):
    other = {"test-1": "x " * 16}
    run.save("deberta-seed7", BagOfWords({"nurse": WEIGHTS}), other)
    with pytest.raises(ValueError, match="test-1"):
        run()


def test_an_interval_runs_from_its_lower_to_its_upper_bound(run):
    pairs = _pairs([0.2, 0.5, 0.9, 0.7, 0.4, 0.6], [0.1, 0.5, 0.9, 0.7, 0.3, 0.6])
    check = run.module._check("seed", pairs)
    for metric in ("spearman", "top10_overlap"):
        for statistic in ("mean", "median"):
            low, high = check[metric][f"{statistic}_ci"]
            assert low < high


def test_a_run_on_the_same_words_but_for_another_occupation_stops_the_check(run):
    (other,) = attribution.shapley(
        BagOfWords({"nurse": WEIGHTS}), [BIOS[1][1]], ["surgeon"], 3, "delete", 0
    )
    attribution.save(attribution.path("deberta-unk", "test-1"), other)
    with pytest.raises(ValueError, match="test-1"):
        run()


def test_the_settings_record_the_seeds_the_resamples_and_the_target(run):
    record = run()
    assert record["settings"]["bootstrap_seed"] == 42
    assert record["settings"]["n_boot"] == 1000
    assert record["settings"]["target"] == "true occupation"
    assert record["sample"]["seed"] == 42


def test_the_example_names_the_true_occupation_and_lists_the_largest_words_by_absolute_value(run):
    saved = attribution.load(attribution.path("deberta", "test-2"))
    shapley = np.arange(1.0, 17.0)
    shapley[4] = -50.0  # the largest in absolute value, but negative
    case = checks.Case("test-2", saved.words, shapley, np.zeros(16))
    example = run.module._explain(case, 0.1, run.fake)
    assert example["true_occupation"] == "nurse"
    assert example["word"] == saved.words[4]
    assert [entry["shapley"] for entry in example["top_words"]] == [
        -50.0,
        16.0,
        15.0,
        14.0,
        13.0,
        12.0,
        11.0,
        10.0,
    ]


def test_the_example_counts_every_occurrence_of_the_top_word_including_earlier_ones(run):
    words = ["python", *[f"w{i}" for i in range(10)], "Python,", *[f"v{i}" for i in range(4)]]
    case = checks.Case("test-2", words, np.eye(16)[11] * 5, np.zeros(16))  # the top word is word 11
    example = run.module._explain(case, 0.0, run.fake)
    assert [entry["position"] for entry in example["occurrences"]] == [0, 11]


def test_the_randomisation_compares_the_random_model_with_the_main_run_and_has_no_verdict(run):
    reversed_weights = BagOfWords({"nurse": {w: -v for w, v in WEIGHTS.items()}})
    run.save("deberta-random", reversed_weights)
    record = run()
    random = record["randomisation"]
    assert random["n_bios"] == 3
    assert random["spearman"]["median"] == pytest.approx(-1.0)
    assert random["top10_overlap"]["median"] < 1.0
    for metric in ("spearman", "top10_overlap"):
        low, high = random[metric]["median_ci"]
        assert low <= random[metric]["median"] <= high
    assert "random" in random["comparison"]
    assert random["expected"]
    assert not {"passed", "rule", "thresholds"} & random.keys()
    assert set(record["checks"]) == {"seed", "removal", "method"}


def test_a_random_run_made_on_other_words_than_the_main_run_stops_the_check(run):
    run.save("deberta-random", BagOfWords({"nurse": WEIGHTS}), {"test-1": "x " * 16})
    with pytest.raises(ValueError, match="test-1"):
        run()
