import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import BagOfWords, Hashed

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shapley_pilot.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("shapley_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Rule(Hashed):
    """Predicts nurse or surgeon from `rule(text)`; its logits are the Hashed ones."""

    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def predict(self, texts):
        return [self.rule(text) for text in texts]


def _unk_confuses(text):
    if "[UNK]" in text:
        return "surgeon"
    return "nurse" if "nurse" in text else "surgeon"


def _short_confuses(text):
    if len(text.split()) < 8:
        return "surgeon"
    return "nurse" if "nurse" in text else "surgeon"


def _dev_frame(n=40):
    # Five copies of the occupation word between five words that occur in no other bio.
    rows = []
    for i in range(n):
        label = ("nurse", "surgeon")[i % 2]
        words = [w for j in range(5) for w in (label, f"filler{i}x{j}")]
        rows.append((f"dev-{i}", " ".join(words), label))
    return pd.DataFrame(rows, columns=["id", "text", "occupation"]).assign(gender="male")


def _subset_frame(n=12):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    rows = [(f"test-{i}", " ".join(words[: 4 + i % 6]), "nurse") for i in range(n)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation"]).assign(gender="male")


@pytest.fixture
def pilot(monkeypatch, tmp_path):
    """Return go(model): the script's main() on small synthetic frames, giving the JSON.

    Sizes are shrunk through the module constants; `go.module` allows changing more.
    """
    results = tmp_path / "results"
    results.mkdir()
    subset = _subset_frame()
    (results / "showcase_subset_ids.csv").write_bytes(
        ("id\n" + "".join(f"{i}\n" for i in subset["id"])).encode()
    )
    frames = {"dev": _dev_frame(), "test": subset}
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": frames[split])
    module = _load_script()
    monkeypatch.setattr(module, "REMOVAL_BIOS", 30)
    monkeypatch.setattr(module, "REMOVAL_DRAWS", 3)
    monkeypatch.setattr(module, "COST_BIOS", 6)
    monkeypatch.setattr(module, "PERMUTATIONS", (1, 2, 3))

    def go(model):
        monkeypatch.setattr(module, "_load_main", lambda batch_size: model)
        module.main(results)
        return json.loads((results / "shapley_pilot.json").read_text())

    go.module = module
    go.results = results
    return go


def test_the_removal_with_the_smaller_macro_f1_drop_is_primary_and_the_other_is_secondary(pilot):
    report = pilot(Rule(_unk_confuses))["removal_pilot"]
    assert report["primary_removal"] == "delete"
    assert report["secondary_removal"] == "unk"
    assert report["macro_f1_drop_mean"]["delete"] == pytest.approx(0.0)
    assert report["macro_f1_drop_mean"]["unk"] > 0.1


def test_the_pilot_can_pick_unk_when_deleting_hurts_more(pilot):
    report = pilot(Rule(_short_confuses))["removal_pilot"]
    assert report["primary_removal"] == "unk"
    assert report["secondary_removal"] == "delete"
    assert report["macro_f1_drop_mean"]["unk"] == pytest.approx(0.0)
    assert report["macro_f1_drop_mean"]["delete"] > 0.1


def test_a_tie_in_the_macro_f1_drop_goes_to_delete(pilot):
    report = pilot(Rule(lambda text: "nurse" if "nurse" in text else "surgeon"))["removal_pilot"]
    assert report["macro_f1_drop_mean"] == {"delete": 0.0, "unk": 0.0}
    assert report["primary_removal"] == "delete"


def test_the_removal_pilot_reports_its_design_and_states_its_rule(pilot):
    report = pilot(Rule(_unk_confuses))["removal_pilot"]
    assert report["n_bios"] == 30
    assert report["dropped_share"] == 0.3
    assert report["n_draws"] == 3
    assert report["macro_f1_unperturbed"] == pytest.approx(1.0)
    assert len(report["macro_f1_drops"]["delete"]) == 3
    assert "smaller" in report["rule"]
    assert "macro-F1" in report["rule"]


def test_thirty_percent_of_the_words_are_dropped_with_the_same_words_for_both_removals(pilot):
    seen = []

    class Spy(Rule):
        def predict(self, texts):
            seen.append(list(texts))
            return super().predict(texts)

    pilot(Spy(_unk_confuses))
    # Calls: the unperturbed bios, then per draw the deleted and the [UNK] versions.
    unperturbed, *draws = seen
    assert len(draws) == 2 * 3
    assert len({tuple(deleted) for deleted in draws[0::2]}) == 3  # each draw drops other words
    for deleted, replaced in zip(draws[0::2], draws[1::2], strict=True):
        for original, gone, unk in zip(unperturbed, deleted, replaced, strict=True):
            assert len(original.split()) == 10
            assert len(gone.split()) == 7
            assert len(unk.split()) == 10
            assert unk.count("[UNK]") == 3
            assert [w for w in unk.split() if w != "[UNK]"] == gone.split()


WEIGHTS = dict(
    zip(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa".split(),
        [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.7, -0.3, 2.2],
        strict=True,
    )
)


def _linear_model():
    model = BagOfWords({"nurse": WEIGHTS})
    model.predict = lambda texts: ["nurse"] * len(texts)
    return model


def test_the_cost_pilot_reports_seconds_per_bio_and_seed_agreement_for_each_count(pilot):
    report = pilot(_linear_model())["cost_pilot"]
    sampled = _subset_frame().sample(6, random_state=42)
    assert report["n_bios"] == 6
    assert report["seeds"] == [0, 1]
    assert sorted(report["by_permutations"]) == ["1", "2", "3"]
    for entry in report["by_permutations"].values():
        assert entry["seconds_per_bio"] > 0
        assert entry["mean_words"] == pytest.approx(sampled["text"].str.split().str.len().mean())
        assert entry["median_spearman"] == pytest.approx(1.0)  # linear scores: exact values
    assert "0.9" in report["rule"]
    assert report["threshold"] == 0.9


def test_the_cost_pilot_uses_the_primary_removal(pilot):
    class Spy(Rule):
        def logits(self, texts):
            seen.extend(texts)
            return super().logits(texts)

    seen = []
    unk_primary = pilot(Spy(_short_confuses))
    assert unk_primary["removal_pilot"]["primary_removal"] == "unk"
    assert any(text.startswith("[UNK]") for text in seen)
    seen.clear()
    delete_primary = pilot(Spy(_unk_confuses))
    assert delete_primary["removal_pilot"]["primary_removal"] == "delete"
    assert not any("[UNK]" in text for text in seen)
    assert "" in seen


@pytest.mark.parametrize(
    ("median_spearman", "expected"),
    [
        ({5: 0.7, 10: 0.91, 20: 0.95}, (10, True)),
        ({5: 0.95, 10: 0.97, 20: 0.99}, (5, True)),
        ({5: 0.9, 10: 0.5, 20: 0.99}, (5, True)),
        ({5: 0.7, 10: 0.89, 20: 0.8}, (20, False)),
        ({5: float("nan"), 10: 0.92, 20: 0.99}, (10, True)),
    ],
)
def test_the_smallest_count_with_median_spearman_of_at_least_0_9_is_chosen(
    median_spearman, expected
):
    module = _load_script()
    assert module.choose_permutations(median_spearman, 0.9) == expected


def test_the_chosen_count_and_the_projection_for_the_full_run_follow_the_stated_rules(pilot):
    result = pilot(_linear_model())
    cost = result["cost_pilot"]
    assert cost["chosen_permutations"] == 1  # every count has Spearman 1
    assert cost["threshold_met"] is True
    projection = cost["full_run"]
    seconds = cost["by_permutations"]["1"]["seconds_per_bio"]
    assert projection["n_bios"] == 12
    assert projection["projected_seconds"] == pytest.approx(seconds * 12)
    assert projection["projected_hours"] == pytest.approx(seconds * 12 / 3600)
    assert projection["limit_hours"] == 3
    assert projection["method"] == "shapley"
    assert "3 hours" in projection["rule"]
    assert result["chosen"] == {
        "method": "shapley",
        "removal": result["removal_pilot"]["primary_removal"],
        "n_permutations": 1,
    }
    assert result["inference_batch_size"] == 64


def test_a_projected_run_over_the_limit_falls_back_to_leave_one_out(pilot, monkeypatch):
    monkeypatch.setattr(pilot.module, "FULL_RUN_LIMIT_SECONDS", 0.0)
    result = pilot(_linear_model())
    assert result["cost_pilot"]["full_run"]["method"] == "leave_one_out"
    assert result["chosen"]["method"] == "leave_one_out"
    assert result["chosen"]["removal"] == result["removal_pilot"]["primary_removal"]
    assert "n_permutations" not in result["chosen"]


def test_when_no_count_reaches_the_threshold_the_largest_is_chosen_and_the_report_says_so(pilot):
    result = pilot(_noisy_model())
    cost = result["cost_pilot"]
    assert cost["threshold_met"] is False
    assert cost["chosen_permutations"] == 3
    assert result["chosen"]["n_permutations"] == 3


def _noisy_model():
    model = Hashed()
    model.predict = lambda texts: ["nurse"] * len(texts)
    return model


def test_the_pilot_file_is_written_like_the_other_result_files(pilot):
    pilot(_linear_model())
    raw = (pilot.results / "shapley_pilot.json").read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    assert json.loads(raw)["runtime_seconds"] >= 0


def test_the_model_is_warmed_up_on_the_first_bios_before_any_timed_run(pilot):
    class Spy(Rule):
        pass

    spy = Spy(_unk_confuses)
    pilot(spy)
    sampled = _subset_frame().sample(6, random_state=42)
    assert spy.calls[0] == list(sampled["text"])[:8]


def test_the_default_pilot_sizes_are_the_ones_the_rules_state():
    module = _load_script()
    assert (module.REMOVAL_BIOS, module.REMOVAL_DRAWS, module.DROPPED_SHARE) == (200, 5, 0.3)
    assert (module.COST_BIOS, module.PERMUTATIONS, module.SEEDS) == (50, (5, 10, 20, 40), (0, 1))
    assert (module.SPEARMAN_THRESHOLD, module.FULL_RUN_LIMIT_SECONDS) == (0.9, 3 * 3600)
    assert module.BATCH_SIZE == 64


@pytest.mark.parametrize(
    ("median_spearman", "expected"),
    [
        ({20: 0.95, 5: 0.7, 10: 0.92}, (10, True)),
        ({20: 0.95, 10: None, 5: None}, (20, True)),
        ({5: None, 10: None}, (10, False)),
    ],
)
def test_the_count_choice_ignores_the_order_of_the_dict_and_missing_medians(
    median_spearman, expected
):
    assert _load_script().choose_permutations(median_spearman, 0.9) == expected
