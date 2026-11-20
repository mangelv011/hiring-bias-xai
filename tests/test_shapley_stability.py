import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import BagOfWords, Hashed
from scipy.stats import spearmanr

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shapley_stability.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("shapley_stability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agreement_ranks_the_largest_words_apart_from_the_rest():
    # Mean |value| over the two seeds is [8.5, 8.5, 2.5, 2.5, 2.5, 2.5]: words 0 and 1 are the top
    # two. Among them the seeds agree on the order (9 > -8 and 8 > -9). Among the rest they
    # disagree completely (1 2 3 4 against 4 3 2 1).
    one = np.array([9.0, -8.0, 1.0, 2.0, 3.0, 4.0])
    other = np.array([8.0, -9.0, 4.0, 3.0, 2.0, 1.0])
    result = _load_script().agreement(one, other, top=2, shown=2)
    assert result["spearman_all"] == pytest.approx(spearmanr(one, other).statistic)
    assert result["spearman_top"] == pytest.approx(1.0)
    assert result["spearman_rest"] == pytest.approx(-1.0)
    assert result["top_overlap"] == 1.0


def test_the_overlap_is_the_share_of_the_shown_words_that_both_seeds_rank_first_by_size():
    # Largest |value| of the first seed: words 0 and 1. Of the second: words 0 and 2.
    result = _load_script().agreement(
        np.array([9.0, -8.0, 1.0, 0.0]), np.array([9.0, 1.0, -8.0, 0.0]), top=2, shown=2
    )
    assert result["top_overlap"] == 0.5


def test_a_constant_part_or_too_few_words_gives_nan_for_that_correlation():
    module = _load_script()
    result = module.agreement(
        np.array([1.0, 1.0, 1.0, 5.0]), np.array([2.0, 2.0, 2.0, 7.0]), top=1, shown=1
    )
    assert np.isnan(result["spearman_top"])  # one word is too few
    assert np.isnan(result["spearman_rest"])  # the other three are equal
    assert result["top_overlap"] == 1.0


def _subset_frame(n=12):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    rows = [(f"test-{i}", " ".join(words[: 4 + i % 6]), "nurse") for i in range(n)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation"]).assign(gender="male")


@pytest.fixture
def stability(monkeypatch, tmp_path):
    """Return go(model, removal): the script's main() on a small synthetic subset."""
    results = tmp_path / "results"
    results.mkdir()
    subset = _subset_frame()
    (results / "showcase_subset_ids.csv").write_bytes(
        ("id\n" + "".join(f"{i}\n" for i in subset["id"])).encode()
    )
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    module = _load_script()
    monkeypatch.setattr(module, "BIOS", 6)
    monkeypatch.setattr(module, "PERMUTATIONS", (1, 2))
    monkeypatch.setattr(module, "TOP", 3)

    def go(model, removal="delete"):
        (results / "shapley_pilot.json").write_text(json.dumps({"chosen": {"removal": removal}}))
        monkeypatch.setattr(module, "_load_baseline", lambda: model)
        module.main(results)
        return json.loads((results / "shapley_stability.json").read_text())

    go.module = module
    return go


WEIGHTS = dict(
    zip(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa".split(),
        [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.7, -0.3, 2.2],
        strict=True,
    )
)


def test_on_a_linear_model_every_agreement_is_one(stability):
    report = stability(BagOfWords({"nurse": WEIGHTS}))
    assert report["model"] == "baseline"
    assert sorted(report["by_permutations"]) == ["1", "2"]
    for entry in report["by_permutations"].values():
        assert entry["median_spearman_all"] == pytest.approx(1.0)
        assert entry["median_spearman_top"] == pytest.approx(1.0)
        assert entry["mean_top_overlap"] == pytest.approx(1.0)
        assert entry["n_bios"] == 6


def test_the_report_states_its_rule_and_design(stability):
    report = stability(Hashed())
    assert report["n_bios"] == 6
    assert report["seeds"] == [0, 1]
    assert report["removal"] == "delete"
    assert "top" in report["rule"]
    assert "baseline" in report["rule"]


def test_the_removal_comes_from_the_pilot_choice(stability):
    class Spy(BagOfWords):
        def logits(self, texts):
            seen.extend(texts)
            return super().logits(texts)

    seen = []
    stability(Spy({"nurse": WEIGHTS}), removal="unk")
    assert any("[UNK]" in text for text in seen)


def test_the_default_design_matches_the_cost_pilot_and_the_run():
    module = _load_script()
    assert (module.BIOS, module.PERMUTATIONS, module.SEEDS) == (50, (5, 10, 20, 40), (0, 1))
    assert (module.TOP, module.SHOWN) == (10, 8)


def test_the_top_words_are_those_with_the_largest_values_of_both_seeds_together():
    # Seed one has one large word (0), seed two two (2 and 3). Together the top two are 0 and 2;
    # looking at seed one alone would take 0 and 1.
    result = _load_script().agreement(
        np.array([10.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0, 9.0]), top=2, shown=2
    )
    assert result["spearman_top"] == pytest.approx(-1.0)


def test_the_overlap_looks_at_the_shown_words_of_each_seed_not_at_the_top_group():
    # Largest word of seed one: 1. Of seed two: 0. With a top group of 3 the second seed's list
    # would contain word 1 too.
    result = _load_script().agreement(
        np.array([8.0, 9.0, 1.0, 0.0]), np.array([9.0, 1.0, 8.0, 0.0]), top=3, shown=1
    )
    assert result["top_overlap"] == 0.0


def test_medians_skip_undefined_bios_and_are_none_when_none_is_defined():
    module = _load_script()
    assert module._median([1.0, 0.0, 0.0]) == 0.0
    assert module._median([float("nan"), 1.0, 3.0]) == 2.0
    assert module._median([float("nan")]) is None


def test_the_bios_are_a_seeded_random_sample_of_the_subset(stability, monkeypatch):
    from hiring_bias_xai import attribution

    seen = []
    real = attribution.shapley

    def spy(model, texts, targets, *args, **kwargs):
        seen.append(list(texts))
        return real(model, texts, targets, *args, **kwargs)

    monkeypatch.setattr(attribution, "shapley", spy)
    stability(BagOfWords({"nurse": WEIGHTS}))
    assert seen[0] == list(_subset_frame().sample(6, random_state=42)["text"])


def test_the_report_columns_are_the_medians_of_the_agreement_between_the_two_seeds(stability):
    # Worked out here from two runs with the seeds 0 and 1, on the same six bios.
    report = stability(Hashed(), removal="unk")
    sample = _subset_frame().sample(6, random_state=42)
    texts, targets = list(sample["text"]), list(sample["occupation"])
    assert report["removal"] == "unk"
    assert (report["top"], report["shown"]) == (3, 8)
    for count, entry in report["by_permutations"].items():
        one, other = (
            attribution.shapley(Hashed(), texts, targets, int(count), "unk", seed)
            for seed in (0, 1)
        )
        rows = [spearmanr(a.values, b.values).statistic for a, b in zip(one, other, strict=True)]
        assert entry["median_spearman_all"] == pytest.approx(np.median(rows))
        top = []
        for a, b in zip(one, other, strict=True):
            size = np.abs(a.values) + np.abs(b.values)
            chosen = np.argsort(-size, kind="stable")[:3]
            top.append(spearmanr(a.values[chosen], b.values[chosen]).statistic)
        assert entry["median_spearman_top"] == pytest.approx(np.nanmedian(top))
