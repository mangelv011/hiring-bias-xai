import pytest

from hiring_bias_xai import fairness

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


def _report(**kwargs):
    y_true, gender, y_pred = zip(*ROWS, strict=True)
    return fairness.evaluate(y_true, y_pred, gender, SHARE_WOMEN, **kwargs)


def test_accuracy_has_a_confidence_interval_around_the_point_estimate():
    accuracy = _report(n_boot=200)["accuracy"]
    assert accuracy["ci_low"] <= accuracy["value"] <= accuracy["ci_high"]


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
