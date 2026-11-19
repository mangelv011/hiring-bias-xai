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
