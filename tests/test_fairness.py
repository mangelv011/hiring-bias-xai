import json
import time

import numpy as np
import pytest

from hiring_bias_xai import config, fairness

N, S = "nurse", "surgeon"

# Hand-built example: 2 occupations x 2 genders, 12 bios.
#   true     gender  count  predicted
#   nurse    female  4      nurse, nurse, nurse, surgeon
#   nurse    male    2      nurse, surgeon
#   surgeon  female  2      surgeon, nurse
#   surgeon  male    4      surgeon x4
ROWS = (
    [(N, "female", p) for p in (N, N, N, S)]
    + [(N, "male", p) for p in (N, S)]
    + [(S, "female", p) for p in (S, N)]
    + [(S, "male", S)] * 4
)
SHARE_WOMEN = {N: 0.9, S: 0.1}


def _report(copies=1, **kwargs):
    y_true, gender, y_pred = zip(*(ROWS * copies), strict=True)
    return fairness.evaluate(y_true, y_pred, gender, SHARE_WOMEN, **kwargs)


def test_accuracy_has_a_confidence_interval_around_the_point_estimate():
    accuracy = _report(n_boot=200)["accuracy"]
    assert accuracy["ci_low"] <= accuracy["value"] <= accuracy["ci_high"]


def _metrics(report):
    """Every {"value": ...} entry of a report."""
    for key, entry in report.items():
        if key == "value":
            yield report
        elif isinstance(entry, dict):
            yield from _metrics(entry)


def test_each_resample_has_as_many_bios_as_the_data():
    # 4 of 5 bios right. A resample of 5 gives accuracy k/5 with k ~ Binomial(5, 0.8):
    # P(k <= 1) = 0.007 and P(k <= 2) = 0.058, so the 2.5th percentile is 2/5; k = 5 has
    # probability 0.33, so the 97.5th is 1. Resamples of 4 bios would give multiples of 1/4.
    y_true = ["a"] * 5
    y_pred = ["a"] * 4 + ["b"]
    accuracy = fairness.evaluate(y_true, y_pred, ["female", "male"] * 2 + ["male"], {"a": 0.5})[
        "accuracy"
    ]
    assert (accuracy["ci_low"], accuracy["ci_high"]) == pytest.approx((0.4, 1.0))


def test_shares_are_matched_to_occupations_by_name_not_by_dict_order():
    # Same three occupations as the Pearson example above, with the dict in another order.
    y_true = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    y_pred = ["a", "a", "a", "x", "b", "x", "b", "x", "c", "x", "c", "c"]
    gender = ["female", "female", "male", "male"] * 3
    report = fairness.evaluate(y_true, y_pred, gender, {"c": 0.9, "a": 0.2, "b": 0.5})
    assert report["gap_share_pearson"]["value"] == pytest.approx(-7 / 20 / (37 / 300) ** 0.5)


def test_every_metric_has_an_interval_around_its_point_estimate():
    metrics = list(_metrics(_report(copies=10, n_boot=200)))
    assert len(metrics) == 2 + 3 * 2 + 2  # accuracy, F1, 3 per occupation x 2, RMS, Pearson
    for metric in metrics:
        assert metric["ci_low"] <= metric["value"] <= metric["ci_high"]


def test_resamples_without_a_cell_are_ignored_and_counted_for_the_metrics_that_need_it():
    # One woman among 20 "a" bios: a resample of 40 bios leaves her out with probability
    # (39/40)^40 = 0.36, so about 72 of 200 resamples have no female "a". All predictions are
    # right, so the interval of her TPR is exactly [1, 1] once those resamples are ignored.
    # The last "b" bio, a man, is predicted "a", so the gaps differ between resamples. The gap of
    # "a" and the RMS gap are NaN exactly when her TPR is; the correlation also is when both gaps
    # coincide. Accuracy and macro-F1 do not depend on a gender cell.
    y_true = ["a"] * 20 + ["b"] * 20
    y_pred = ["a"] * 20 + ["b"] * 19 + ["a"]
    gender = ["female"] + ["male"] * 19 + ["female", "male"] * 10
    report = fairness.evaluate(y_true, y_pred, gender, {"a": 0.05, "b": 0.5}, n_boot=200)
    tpr_woman = report["per_occupation"]["a"]["tpr_female"]
    assert 20 < tpr_woman["n_dropped"] < 150
    assert (tpr_woman["ci_low"], tpr_woman["ci_high"]) == (1.0, 1.0)
    assert report["per_occupation"]["a"]["tpr_gap"]["n_dropped"] == tpr_woman["n_dropped"]
    assert report["rms_tpr_gap"]["n_dropped"] == tpr_woman["n_dropped"]
    assert report["gap_share_pearson"]["n_dropped"] >= tpr_woman["n_dropped"]
    assert report["per_occupation"]["a"]["tpr_male"].get("n_dropped") is None
    assert "n_dropped" not in report["accuracy"]
    assert "n_dropped" not in report["macro_f1"]


def test_accuracy_interval_matches_the_normal_approximation():
    # 200 of 400 right: standard error sqrt(0.5 * 0.5 / 400) = 0.025, so the 95% interval is
    # 0.5 -+ 1.96 * 0.025 = [0.451, 0.549]. A bootstrap with 2,000 resamples lands within 0.005.
    y_true = ["a"] * 400
    y_pred = ["a"] * 200 + ["b"] * 200
    gender = ["female", "male"] * 200
    accuracy = fairness.evaluate(y_true, y_pred, gender, {"a": 0.5}, n_boot=2000)["accuracy"]
    assert accuracy["ci_low"] == pytest.approx(0.451, abs=0.005)
    assert accuracy["ci_high"] == pytest.approx(0.549, abs=0.005)


def test_the_same_seed_gives_the_same_report_and_another_seed_another_interval():
    y_true = ["a"] * 400
    y_pred = ["a"] * 200 + ["b"] * 200
    gender = ["female", "male"] * 200

    def accuracy(seed):
        report = fairness.evaluate(y_true, y_pred, gender, {"a": 0.5}, n_boot=200, seed=seed)
        return report["accuracy"]

    assert accuracy(1) == accuracy(1)
    assert accuracy(1)["ci_low"] != accuracy(2)["ci_low"]


def test_accuracy_is_the_share_of_bios_with_the_right_occupation():
    # Correct: nurse 3 + 1, surgeon 1 + 4 = 9 of 12.
    assert _report()["accuracy"]["value"] == pytest.approx(9 / 12)


def test_accuracy_macro_f1_and_rms_gap_do_not_weight_occupations_by_their_size():
    # "a" has 4 bios, "b" has 2: women (a, a | b), men (a, b | a). Bios 3 and 5 are wrong.
    # accuracy 4/6 (the mean of the recalls 3/4 and 1/2 would be 5/8).
    # F1 a = 2*3 / (2*3 + 1 + 1) = 3/4, F1 b = 2*1 / (2*1 + 1 + 1) = 1/2: macro-F1 5/8
    # (weighted by size it would be 2/3).
    # gaps: a = 1 - 1/2 = 1/2, b = 1 - 0 = 1: RMS sqrt((1/4 + 1) / 2) = sqrt(5/8)
    # (weighted by size it would be sqrt(1/2)).
    y_true = ["a"] * 4 + ["b"] * 2
    y_pred = ["a", "a", "a", "b", "b", "a"]
    gender = ["female", "female", "male", "male", "female", "male"]
    report = fairness.evaluate(y_true, y_pred, gender, {"a": 0.3, "b": 0.7}, n_boot=10)
    assert report["accuracy"]["value"] == pytest.approx(4 / 6)
    assert report["macro_f1"]["value"] == pytest.approx(5 / 8)
    assert report["rms_tpr_gap"]["value"] == pytest.approx((5 / 8) ** 0.5)


def test_the_defaults_are_a_thousand_resamples_and_the_project_seed():
    assert _report() == _report(n_boot=1000, seed=config.SEED)


def test_macro_f1_averages_the_f1_of_each_occupation():
    # F1 = 2 TP / (2 TP + FP + FN).
    # nurse: TP 4, FP 1 (a female surgeon), FN 2 (a female and a male nurse) -> 8 / 11.
    # surgeon: TP 5, FP 2, FN 1 -> 10 / 13.
    assert _report()["macro_f1"]["value"] == pytest.approx((8 / 11 + 10 / 13) / 2)


def test_tpr_is_the_share_of_true_bios_of_an_occupation_predicted_as_it():
    per_occupation = _report()["per_occupation"]
    # nurse: 3 of 4 women, 1 of 2 men. surgeon: 1 of 2 women, 4 of 4 men.
    assert per_occupation[N]["tpr_female"]["value"] == pytest.approx(3 / 4)
    assert per_occupation[N]["tpr_male"]["value"] == pytest.approx(1 / 2)
    assert per_occupation[S]["tpr_female"]["value"] == pytest.approx(1 / 2)
    assert per_occupation[S]["tpr_male"]["value"] == pytest.approx(4 / 4)


def test_per_occupation_counts_the_true_bios_of_each_gender():
    per_occupation = _report()["per_occupation"]
    assert [per_occupation[N]["n_female"], per_occupation[N]["n_male"]] == [4, 2]
    assert [per_occupation[S]["n_female"], per_occupation[S]["n_male"]] == [2, 4]


def test_tpr_gap_is_the_female_tpr_minus_the_male_tpr():
    per_occupation = _report()["per_occupation"]
    assert per_occupation[N]["tpr_gap"]["value"] == pytest.approx(3 / 4 - 1 / 2)
    assert per_occupation[S]["tpr_gap"]["value"] == pytest.approx(1 / 2 - 4 / 4)


def test_a_model_that_misses_more_women_than_men_has_a_negative_gap():
    report = fairness.evaluate(
        [N] * 4,
        [N, S, N, N],  # women: one of two found; men: both found
        ["female", "female", "male", "male"],
        {N: 0.5},
    )
    assert report["per_occupation"][N]["tpr_gap"]["value"] == pytest.approx(-0.5)


def test_rms_tpr_gap_is_the_root_mean_square_of_the_gaps():
    # Gaps are +1/4 (nurse) and -1/2 (surgeon): sqrt((1/16 + 1/4) / 2) = sqrt(5/32).
    assert _report()["rms_tpr_gap"]["value"] == pytest.approx((5 / 32) ** 0.5)


def test_gap_share_pearson_is_one_when_the_gap_rises_with_the_share_of_women():
    # Gaps (nurse, surgeon) = (+1/4, -1/2) and shares = (0.9, 0.1): two points on a rising line.
    assert _report()["gap_share_pearson"]["value"] == pytest.approx(1.0)


def test_gap_share_pearson_follows_the_gaps_and_shares_of_every_occupation():
    # Three occupations of 2 women and 2 men; "x" is a wrong answer. Gaps are +1/2, 0, -1/2:
    #   a: women 2 of 2, men 1 of 2   b: women 1 of 2, men 1 of 2   c: women 1 of 2, men 2 of 2
    # With shares 0.2, 0.5, 0.9 (mean 8/15) the sums are: gap x share = 1/2 * -1/3
    # + -1/2 * 11/30 = -7/20, gap^2 = 1/2, share^2 = 37/150, so r = -7/20 / sqrt(37/300).
    y_true = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    y_pred = ["a", "a", "a", "x", "b", "x", "b", "x", "c", "x", "c", "c"]
    gender = ["female", "female", "male", "male"] * 3
    report = fairness.evaluate(y_true, y_pred, gender, {"a": 0.2, "b": 0.5, "c": 0.9})
    assert report["gap_share_pearson"]["value"] == pytest.approx(-7 / 20 / (37 / 300) ** 0.5)


def test_the_report_is_json_ready():
    report = _report(copies=10, n_boot=50)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_a_thousand_resamples_of_a_full_test_split_take_well_under_a_minute():
    rng = np.random.default_rng(0)
    n, k = 99_069, 28
    occupations = np.array([f"occ{i:02d}" for i in range(k)])
    y_true = occupations[rng.integers(0, k, n)]
    y_pred = np.where(rng.random(n) < 0.8, y_true, occupations[rng.integers(0, k, n)])
    gender = np.where(rng.random(n) < 0.46, "female", "male")
    started = time.perf_counter()
    fairness.evaluate(y_true, y_pred, gender, dict.fromkeys(occupations, 0.5), n_boot=1000)
    assert time.perf_counter() - started < 60
