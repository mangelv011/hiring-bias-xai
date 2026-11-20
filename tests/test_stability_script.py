import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import BagOfWords

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stability.py"
NURSE = data.OCCUPATIONS.index("nurse")
WEIGHTS = {f"w{i}": float(i - 4) for i in range(16)}
VARIANTS = ("deberta", "deberta-seed7", "deberta-unk")
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
