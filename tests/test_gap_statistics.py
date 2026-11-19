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
