import numpy as np
import pytest

from hiring_bias_xai import explanation_checks as checks


@pytest.mark.parametrize(
    ("n_words", "percent", "expected"),
    [
        (10, 0, 0),
        (10, 5, 1),  # 0.5 words rounds half up
        (10, 15, 2),  # 1.5 words rounds half up
        (58, 5, 3),  # 2.9 words
        (58, 10, 6),  # 5.8 words
        (59, 10, 6),  # 5.9 words
        (60, 7, 4),  # 4.2 words
        (17, 50, 9),  # 8.5 words rounds half up
        (1, 50, 1),
    ],
)
def test_the_share_of_words_to_remove_rounds_half_up_to_a_whole_word(n_words, percent, expected):
    assert checks.words_to_remove(n_words, percent) == expected


VALUES = np.array([0.5, -1.0, 3.0, 0.5, 2.0])


def test_morf_removes_the_largest_value_first_and_lerf_the_smallest_ties_by_position():
    orders = checks.removal_orders(VALUES, 3, np.random.default_rng(0))
    assert orders.morf.tolist() == [2, 4, 0, 3, 1]
    assert orders.lerf.tolist() == [1, 0, 3, 4, 2]


def test_random_orders_are_distinct_permutations_of_the_words_and_follow_the_seed():
    first = checks.removal_orders(VALUES, 10, np.random.default_rng(7)).random
    again = checks.removal_orders(VALUES, 10, np.random.default_rng(7)).random
    other = checks.removal_orders(VALUES, 10, np.random.default_rng(8)).random
    assert first.shape == (10, 5)
    assert all(sorted(row) == [0, 1, 2, 3, 4] for row in first)
    assert len({tuple(row) for row in first}) > 1
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
