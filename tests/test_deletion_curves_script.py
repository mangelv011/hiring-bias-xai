import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import BagOfWords

from hiring_bias_xai import attribution, data
from hiring_bias_xai import explanation_checks as checks

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deletion_curves.py"
SCREENERS = ("deberta", "baseline")
WEIGHTS = {f"w{i}": float(i - 6) for i in range(20)}  # w19 helps nurse most; a bio sums to +70


def _bio_text(start):
    return " ".join(f"w{(start + i) % 20}" for i in range(20))


BIOS = [(f"test-{i}", _bio_text(3 * i), "nurse") for i in range(4)]


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(): the script's main() on four synthetic bios of 20 words.

    The saved attributions of both screeners come from a linear fake, so the Shapley value of a
    word is its weight. `go.attribution_weights` can be replaced before saving to make them lie.
    `go.fakes` are the classifiers that score the deletion steps.
    """
    results = tmp_path / "results"
    results.mkdir()
    ids = "".join(f"{bio_id}\n" for bio_id, _, _ in BIOS)
    (results / "showcase_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    (results / "shapley_pilot.json").write_text(json.dumps({"inference_batch_size": 16}))
    test = pd.DataFrame(BIOS, columns=["id", "text", "occupation"]).assign(gender="female")
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": test)
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("deletion_curves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fakes = {name: BagOfWords({"nurse": WEIGHTS}) for name in SCREENERS}
    loads = []
    monkeypatch.setattr(
        module, "_load", lambda name, batch_size: (loads.append((name, batch_size)), fakes[name])[1]
    )

    def save_attributions(weights):
        explainer = BagOfWords({"nurse": weights})
        for name in SCREENERS:
            for bio_id, text, occupation in BIOS:
                (result,) = attribution.shapley(explainer, [text], [occupation], 4, "delete", 0)
                attribution.save(attribution.path(name, bio_id), result)

    def go():
        module.main(results)
        return results

    save_attributions(WEIGHTS)
    go.save_attributions = save_attributions
    go.fakes = fakes
    go.loads = loads
    go.results = results
    go.module = module
    return go


def _curves(results):
    return pd.read_csv(results / "deletion_curves.csv")


def test_the_csv_has_a_row_per_screener_order_and_fraction_with_the_documented_columns(run):
    table = _curves(run())
    assert list(table.columns) == [
        "screener",
        "order",
        "fraction",
        "mean_agreement",
        "agreement_ci_low",
        "agreement_ci_high",
        "mean_logodds_drop",
        "drop_ci_low",
        "drop_ci_high",
    ]
    assert len(table) == 2 * 3 * 11
    assert set(table["order"]) == {"morf", "lerf", "random"}
    assert sorted(table["fraction"].unique()) == pytest.approx([i / 20 for i in range(11)])


def test_nothing_is_removed_at_fraction_zero(run):
    start = _curves(run()).query("fraction == 0")
    assert (start["mean_agreement"] == 1).all()
    assert start["mean_logodds_drop"].abs().max() < 1e-6
    assert (start["drop_ci_low"] == start["drop_ci_high"]).all()


def test_removing_the_top_words_hurts_more_than_random_and_random_more_than_the_bottom(run):
    end = _curves(run()).query("fraction == 0.5").set_index(["screener", "order"])
    for name in SCREENERS:
        drop = end["mean_logodds_drop"]
        assert drop[name, "morf"] > drop[name, "random"] > drop[name, "lerf"]
        agreement = end["mean_agreement"]
        assert agreement[name, "morf"] < agreement[name, "random"] <= agreement[name, "lerf"]


def test_each_bio_gets_ten_random_orders_the_eleven_steps_and_its_own_seed(run, monkeypatch):
    calls = []
    real = checks.deletion_curve

    def spy(classifier, words, values, target, percents, n_random, rng):
        calls.append((list(percents), n_random, rng.bit_generator.state["state"]["state"]))
        return real(classifier, words, values, target, percents, n_random, rng)

    monkeypatch.setattr(checks, "deletion_curve", spy)
    run()
    assert len(calls) == 2 * len(BIOS)
    assert all(percents == list(range(0, 55, 5)) for percents, _, _ in calls)
    assert all(n_random == 10 for _, n_random, _ in calls)
    expected = [
        np.random.default_rng((42, i)).bit_generator.state["state"]["state"] for i in range(4)
    ]
    assert [state for _, _, state in calls[:4]] == expected


def test_the_models_are_loaded_with_the_inference_batch_size_of_the_pilot(run):
    run()
    assert run.loads == [("deberta", 16), ("baseline", 16)]


def test_words_are_removed_with_unk_not_deleted(run):
    run()
    texts = [text for call in run.fakes["deberta"].calls for text in call]
    assert any("[UNK]" in text for text in texts)
    assert all(len(text.split()) == 20 for text in texts)


def _areas(results):
    return json.loads((results / "deletion_areas.json").read_text())


def test_areas_are_the_mean_of_the_curve_over_the_eleven_fractions(run):
    results = run()
    table = _curves(results)
    areas = _areas(results)["screeners"]["deberta"]
    for order in ("morf", "lerf", "random"):
        curve = table.query("screener == 'deberta' and order == @order")
        assert areas["logodds_drop"]["area"][order]["value"] == pytest.approx(
            curve["mean_logodds_drop"].mean()
        )
        assert areas["agreement"]["area"][order]["value"] == pytest.approx(
            curve["mean_agreement"].mean()
        )


def test_the_differences_of_areas_carry_bootstrap_intervals_that_contain_the_estimate(run):
    areas = _areas(run())["screeners"]["deberta"]["logodds_drop"]
    area = {order: areas["area"][order]["value"] for order in ("morf", "lerf", "random")}
    for key, expected in (
        ("morf_minus_random", area["morf"] - area["random"]),
        ("random_minus_lerf", area["random"] - area["lerf"]),
    ):
        entry = areas[key]
        assert entry["value"] == pytest.approx(expected)
        assert entry["ci_low"] <= entry["value"] <= entry["ci_high"]
    assert areas["area"]["morf"]["ci_low"] <= area["morf"] <= areas["area"]["morf"]["ci_high"]


def test_each_screener_records_what_was_expected_the_rule_and_whether_it_passed(run):
    record = _areas(run())
    assert record["removal"] == "unk"
    assert record["n_bios"] == 4
    assert "half up" in record["rounding"]
    for name in SCREENERS:
        entry = record["screeners"][name]
        assert entry["expected"]
        assert entry["rule"]
        assert entry["passed"] is True


def test_a_screener_whose_attributions_do_not_beat_random_fails_and_the_failure_is_kept(run):
    run.save_attributions({word: -weight for word, weight in WEIGHTS.items()})
    entry = _areas(run())["screeners"]["deberta"]
    assert entry["passed"] is False
    assert entry["logodds_drop"]["morf_minus_random"]["value"] < 0


def test_a_saved_attribution_of_other_words_than_the_bio_stops_the_run(run):
    other = attribution.load(attribution.path("deberta", "test-1"))
    attribution.save(attribution.path("deberta", "test-0"), other)
    with pytest.raises(ValueError, match="test-0"):
        run()


def test_the_random_curve_is_the_mean_over_bios_of_each_bios_ten_random_orders(run):
    table = _curves(run()).query("screener == 'deberta' and order == 'random'")
    expected = []
    for position, (bio_id, _, _) in enumerate(BIOS):
        saved = attribution.load(attribution.path("deberta", bio_id))
        curve = checks.deletion_curve(
            run.fakes["deberta"],
            saved.words,
            saved.values,
            saved.target,
            range(0, 55, 5),
            10,
            np.random.default_rng((42, position)),
        )
        expected.append(curve.drop[2:].mean(axis=0))
    np.testing.assert_allclose(table["mean_logodds_drop"], np.mean(expected, axis=0), atol=1e-9)


def _per_bio(morf, random, lerf, n=6, spread=0.0):
    """Curves of shape (bios, 3 orders, 11 steps) with the given level per order; the sign of
    `spread` alternates by bio and is added to MoRF."""
    levels = np.array([morf, lerf, random])[:, None] * np.ones(11)
    bios = np.stack([levels.copy() for _ in range(n)])
    bios[::2, 0] += spread
    bios[1::2, 0] -= spread
    return bios


def test_a_screener_passes_only_when_both_intervals_of_the_drop_are_above_zero(run):
    record = run.module._record
    good = record("deberta", _per_bio(0.2, 0.5, 0.8), _per_bio(5.0, 2.0, 0.0))
    assert good["passed"] is True
    assert good["passed_agreement"] is True


def test_a_difference_of_exactly_zero_is_not_a_pass(run):
    record = run.module._record("deberta", _per_bio(0.5, 0.5, 0.5), _per_bio(2.0, 2.0, 0.0))
    assert record["passed"] is False  # MoRF equals random: the interval is [0, 0]


def test_one_good_difference_is_not_enough(run):
    record = run.module._record("deberta", _per_bio(0.5, 0.5, 0.5), _per_bio(5.0, 2.0, 3.0))
    assert record["logodds_drop"]["morf_minus_random"]["ci_low"] > 0
    assert record["logodds_drop"]["random_minus_lerf"]["ci_high"] < 0
    assert record["passed"] is False


def test_an_interval_that_straddles_zero_fails_both_rules(run):
    # MoRF beats random on half of the bios and loses on the other half
    drop = _per_bio(2.0, 2.0, 0.0, spread=2.0)
    agreement = _per_bio(0.5, 0.5, 0.9, spread=0.3)
    record = run.module._record("deberta", agreement, drop)
    area = record["logodds_drop"]["morf_minus_random"]
    assert area["ci_low"] < 0 < area["ci_high"]
    assert record["passed"] is False
    assert record["passed_agreement"] is False


def test_an_agreement_difference_of_exactly_zero_is_not_below_zero(run):
    record = run.module._record("deberta", _per_bio(0.5, 0.5, 0.9), _per_bio(5.0, 2.0, 0.0))
    assert record["passed"] is True
    assert record["passed_agreement"] is False  # MoRF equals random: the interval is [0, 0]


def test_each_curve_row_has_its_lower_bound_under_its_mean_under_its_upper_bound(run):
    rng = np.random.default_rng(0)
    agreement, drop = rng.uniform(size=(2, 8, 3, 11))
    table = run.module._curve_table("deberta", agreement, drop)
    for metric, low, high in (
        ("mean_agreement", "agreement_ci_low", "agreement_ci_high"),
        ("mean_logodds_drop", "drop_ci_low", "drop_ci_high"),
    ):
        assert (table[low] < table[metric]).all()
        assert (table[metric] < table[high]).all()


def test_the_json_records_the_ten_random_orders_and_the_eleven_steps(run):
    record = _areas(run())
    assert record["n_random_orders"] == 10
    assert record["percents"] == list(range(0, 55, 5))


def test_the_curve_intervals_use_the_default_number_of_resamples(run, monkeypatch):
    sizes = []
    real = checks.bootstrap_ci

    def spy(statistic, n, *args, **kwargs):
        sizes.append(kwargs.get("n_boot", args[0] if args else checks.N_BOOT))
        return real(statistic, n, *args, **kwargs)

    monkeypatch.setattr(checks, "bootstrap_ci", spy)
    run()
    assert set(sizes) == {1000}


def test_the_baseline_model_is_loaded_without_touching_the_main_model(monkeypatch):
    spec = importlib.util.spec_from_file_location("deletion_curves_real", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # its own _load, not the fixture's fake
    sentinel = object()
    monkeypatch.setattr(module.BaselineModel, "load", classmethod(lambda cls: sentinel))
    monkeypatch.setattr(
        module.DebertaModel, "load", classmethod(lambda cls, size: pytest.fail("main model"))
    )
    assert module._load("baseline", 16) is sentinel
