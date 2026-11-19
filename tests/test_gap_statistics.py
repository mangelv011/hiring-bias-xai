import numpy as np
import pytest

from hiring_bias_xai import gap_statistics


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
    # Pearson: sxy = 14, sxx = syy = 17.5 -> 0.8. Spearman is the same on these ranks.
    assert out["pearson"]["value"] == pytest.approx(0.8)
    assert out["spearman"]["value"] == pytest.approx(0.8)


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
    low, high = gap_statistics.fisher_ci(0.8, 6)
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
