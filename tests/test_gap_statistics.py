import numpy as np
import pytest

from hiring_bias_xai import fairness, gap_statistics


def test_fisher_ci_of_r_half_with_28_points():
    # z = atanh(0.5), se = 1 / sqrt(28 - 3) = 0.2, interval tanh(z -+ 1.959964 * 0.2)
    low, high = gap_statistics.fisher_ci(0.5, 28)
    assert (low, high) == pytest.approx((0.15603, 0.73582), abs=1e-5)


def test_fisher_ci_of_r_point_nine_with_10_points():
    low, high = gap_statistics.fisher_ci(0.9, 10)
    assert (low, high) == pytest.approx((0.62393, 0.97636), abs=1e-5)


def test_fisher_ci_of_a_perfect_correlation_is_the_point():
    assert gap_statistics.fisher_ci(1.0, 28) == (1.0, 1.0)
    assert gap_statistics.fisher_ci(-1.0, 28) == (-1.0, -1.0)


def test_holm_adjusts_a_hand_checked_list():
    # Sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04; the running
    # maximum makes the last 0.06. Back in input order:
    adjusted = gap_statistics.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one():
    assert gap_statistics.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_two_proportion_p_of_90_against_80_hits_in_100_each():
    # pooled 0.85, se = sqrt(0.85 * 0.15 * 2 / 100) = 0.0505, z = 1.9803
    assert gap_statistics.two_proportion_p(90, 100, 80, 100) == pytest.approx(0.04767, abs=1e-5)


def test_two_proportion_p_does_not_depend_on_the_order_of_the_groups():
    forward = gap_statistics.two_proportion_p(90, 100, 80, 100)
    assert gap_statistics.two_proportion_p(80, 100, 90, 100) == pytest.approx(forward)


def test_two_proportion_p_is_one_when_both_groups_are_all_hits_or_all_misses():
    assert gap_statistics.two_proportion_p(10, 10, 20, 20) == 1.0
    assert gap_statistics.two_proportion_p(0, 10, 0, 20) == 1.0


def _line(n=28):
    share = np.linspace(0.1, 0.9, n)
    return 2 * share - 1, share


def test_a_perfect_correlation_has_a_degenerate_interval_in_every_estimate():
    gap, share = _line()
    out = gap_statistics.correlations(gap, share, n_boot=200)
    assert out["pearson"] == pytest.approx(
        {
            "value": 1.0,
            "fisher_ci_low": 1.0,
            "fisher_ci_high": 1.0,
            "bootstrap_ci_low": 1.0,
            "bootstrap_ci_high": 1.0,
        }
    )
    assert out["spearman"] == pytest.approx({"value": 1.0, "ci_low": 1.0, "ci_high": 1.0})


def test_spearman_is_one_for_a_monotone_curve_where_pearson_is_not():
    share = np.linspace(0.1, 0.9, 28)
    out = gap_statistics.correlations(share**8, share, n_boot=50)
    assert out["spearman"]["value"] == pytest.approx(1.0)
    assert out["pearson"]["value"] < 0.95


def test_the_point_values_are_the_correlations_of_the_data():
    gap = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    share = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    out = gap_statistics.correlations(gap, share, n_boot=10)
    # Pearson: sxy = 15.5, sxx = syy = 17.5 -> 31/35. Spearman is the same on these ranks.
    assert out["pearson"]["value"] == pytest.approx(31 / 35)
    assert out["spearman"]["value"] == pytest.approx(31 / 35)


def test_the_occupation_bootstrap_intervals_are_wide_for_noisy_points_and_reproducible():
    rng = np.random.default_rng(1)
    share = np.linspace(0.1, 0.9, 28)
    gap = share + rng.normal(0, 0.4, 28)
    out = gap_statistics.correlations(gap, share, n_boot=500)
    assert out == gap_statistics.correlations(gap, share, n_boot=500)
    assert out != gap_statistics.correlations(gap, share, n_boot=500, seed=7)
    for estimate, low, high in (
        ("pearson", "bootstrap_ci_low", "bootstrap_ci_high"),
        ("spearman", "ci_low", "ci_high"),
    ):
        assert out[estimate][low] < out[estimate]["value"] < out[estimate][high]
    assert out["pearson"]["bootstrap_ci_high"] - out["pearson"]["bootstrap_ci_low"] > 0.3


def test_the_fisher_interval_uses_the_number_of_occupations():
    gap = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    share = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    out = gap_statistics.correlations(gap, share, n_boot=10)["pearson"]
    low, high = gap_statistics.fisher_ci(31 / 35, 6)
    assert (out["fisher_ci_low"], out["fisher_ci_high"]) == pytest.approx((low, high))


def test_leave_one_out_finds_the_influential_occupation():
    # Four occupations on the line y = x and "e" at (5, -5). Without "e" r = 1; the lowest r,
    # -0.7467, comes from leaving out "d".
    share = np.array([1, 2, 3, 4, 5.0])
    gap = np.array([1, 2, 3, 4, -5.0])
    out = gap_statistics.leave_one_out(gap, share, ["a", "b", "c", "d", "e"])
    assert out["max"]["left_out"] == "e"
    assert out["max"]["r"] == pytest.approx(1.0)
    assert out["min"]["left_out"] == "d"
    assert out["min"]["r"] == pytest.approx(-0.74673, abs=1e-5)


def _bios(spec):
    """Bios from {occupation: (hits_women, n_women, hits_men, n_men)}; a miss predicts "other"."""
    y_true, y_pred, gender = [], [], []
    for occupation, (hits_f, n_f, hits_m, n_m) in spec.items():
        for g, hits, n in (("female", hits_f, n_f), ("male", hits_m, n_m)):
            y_true += [occupation] * n
            y_pred += [occupation] * hits + ["other"] * (n - hits)
            gender += [g] * n
    return np.array(y_true), np.array(y_pred), np.array(gender)


def test_a_planted_gap_in_every_occupation_gives_the_smallest_possible_p():
    spec = {"a": (20, 20, 0, 20), "b": (20, 20, 0, 20), "c": (20, 20, 0, 20)}
    out = gap_statistics.rms_permutation_test(*_bios(spec), n_perm=99)
    assert out["observed"] == pytest.approx(1.0)
    assert out["p_value"] == pytest.approx(1 / 100)
    assert out["n_permutations"] == 99
    assert out["null_mean"] < 0.4
    assert out["null_p95"] < 0.6


def test_hits_that_do_not_depend_on_gender_give_p_one_and_a_null_of_zero():
    # "a" is all hits and "b" all misses; "a" is mostly women, "b" mostly men. Shuffling gender
    # inside an occupation changes nothing; shuffling across occupations would give gaps.
    spec = {"a": (45, 45, 5, 5), "b": (0, 5, 0, 45)}
    out = gap_statistics.rms_permutation_test(*_bios(spec), n_perm=50)
    assert out["observed"] == 0.0
    assert out["null_mean"] == 0.0
    assert out["p_value"] == 1.0


def test_the_observed_rms_gap_matches_the_fairness_module():
    spec = {"a": (30, 40, 10, 20), "b": (5, 10, 18, 30)}
    y_true, y_pred, gender = _bios(spec)
    report = fairness.evaluate(y_true, y_pred, gender, {"a": 0.5, "b": 0.5}, n_boot=2)
    out = gap_statistics.rms_permutation_test(y_true, y_pred, gender, n_perm=10)
    assert out["observed"] == pytest.approx(report["rms_tpr_gap"]["value"])


def test_the_null_is_reproducible_and_depends_on_the_seed():
    y_true, y_pred, gender = _bios({"a": (30, 40, 10, 20), "b": (5, 10, 18, 30)})
    first = gap_statistics.rms_permutation_test(y_true, y_pred, gender, n_perm=100)
    assert first == gap_statistics.rms_permutation_test(y_true, y_pred, gender, n_perm=100)
    other = gap_statistics.rms_permutation_test(y_true, y_pred, gender, n_perm=100, seed=1)
    assert other["null_mean"] != first["null_mean"]


def test_without_a_gender_effect_p_values_are_not_concentrated_near_zero():
    # 40 datasets with hits at 70% for both genders; p should be roughly uniform, so about 5%
    # fall below 0.05. A p that is too small too often would show as more than a quarter here.
    rng = np.random.default_rng(3)
    small = 0
    for _ in range(40):
        spec = {o: (rng.binomial(30, 0.7), 30, rng.binomial(30, 0.7), 30) for o in "abcd"}
        out = gap_statistics.rms_permutation_test(*_bios(spec), n_perm=200)
        small += out["p_value"] < 0.05
    assert small <= 10


def test_the_null_summary_is_the_mean_and_95th_percentile_of_the_permuted_rms_gaps():
    # One occupation with 1 hit among 2 women and 1 among 2 men. Shuffling the 2 hits over the
    # 4 bios puts 0, 1 or 2 of them on women with probability 1/6, 4/6, 1/6, so the gap is -1, 0
    # or 1: the null rms is 1 with probability 1/3 and 0 otherwise.
    y_true, y_pred, gender = _bios({"a": (1, 2, 1, 2)})
    out = gap_statistics.rms_permutation_test(y_true, y_pred, gender, n_perm=2000)
    assert out["null_p95"] == 1.0
    assert out["null_mean"] == pytest.approx(1 / 3, abs=0.04)


SPEC = {
    "a": (45, 50, 20, 25),
    "b": (30, 50, 45, 60),
    "c": (10, 20, 70, 80),
    "d": (60, 80, 20, 30),
}
SHARES = {"a": 0.2, "b": 0.4, "c": 0.7, "d": 0.9}


def test_compute_agrees_with_the_fairness_module_and_holds_every_statistic():
    y_true, y_pred, gender = _bios(SPEC)
    report = fairness.evaluate(y_true, y_pred, gender, SHARES, n_boot=2)
    out = gap_statistics.compute(y_true, y_pred, gender, SHARES, n_boot=20, n_perm=20)
    assert out["n_occupations"] == 4
    assert out["pearson"]["value"] == pytest.approx(report["gap_share_pearson"]["value"])
    assert out["rms_tpr_gap"]["observed"] == pytest.approx(report["rms_tpr_gap"]["value"])
    assert out["rms_tpr_gap"]["n_permutations"] == 20
    assert sorted(out["spearman"]) == ["ci_high", "ci_low", "value"]
    assert sorted(out["leave_one_out"]) == ["max", "min"]
    for name, row in out["occupations"].items():
        assert row["share_women"] == SHARES[name]
        assert row["tpr_gap"] == pytest.approx(report["per_occupation"][name]["tpr_gap"]["value"])
        assert 0 <= row["p_raw"] <= row["p_holm"] <= 1
    assert out["n_significant_holm"] == sum(
        row["p_holm"] < 0.05 for row in out["occupations"].values()
    )
    assert 0 < out["n_significant_holm"] < 4  # some gaps survive Holm, not all


def test_occupation_tests_report_raw_and_holm_adjusted_p_and_the_gap():
    # a: 90% of 100 women against 80% of 100 men, raw p = 0.04767. b: 5 of 10 against 50 of 100,
    # no gap, p = 1. c: 0 of 100 against 40 of 100, p < 1e-9. Holm over 3: sorted p are c, a, b,
    # factors 3, 2, 1, so a gets 2 * 0.04767 and b stays 1.
    spec = {"a": (90, 100, 80, 100), "b": (5, 10, 50, 100), "c": (0, 100, 40, 100)}
    out = gap_statistics.occupation_tests(*_bios(spec))
    assert list(out) == ["a", "b", "c"]
    assert out["a"]["tpr_gap"] == pytest.approx(0.1)
    assert out["a"]["p_raw"] == pytest.approx(0.04767, abs=1e-5)
    assert out["a"]["p_holm"] == pytest.approx(2 * 0.04767, abs=1e-4)
    assert out["b"]["tpr_gap"] == pytest.approx(0.0)
    assert out["b"]["p_raw"] == 1.0
    assert out["b"]["p_holm"] == 1.0
    assert out["c"]["tpr_gap"] == pytest.approx(-0.4)
    assert out["c"]["p_raw"] < 1e-9
    assert out["c"]["p_holm"] == pytest.approx(3 * out["c"]["p_raw"])
