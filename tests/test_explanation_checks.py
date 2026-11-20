import numpy as np
import pytest
from fakes import BagOfWords

from hiring_bias_xai import data
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


NURSE = data.OCCUPATIONS.index("nurse")
WORDS = ["alpha", "beta", "gamma", "delta"]
WEIGHTS = {"alpha": 3.0, "beta": 2.0, "gamma": -1.0}  # the values of a linear bio are its weights


def _curve(percents=(0, 25, 50), n_random=2, seed=0, classifier=None):
    classifier = classifier or BagOfWords({"nurse": WEIGHTS})
    values = np.array([3.0, 2.0, -1.0, 0.0])
    return classifier, checks.deletion_curve(
        classifier, WORDS, values, NURSE, percents, n_random, np.random.default_rng(seed)
    )


def test_the_drop_is_the_fall_in_true_class_log_odds_from_the_full_bio_per_order_and_step():
    _, curve = _curve()
    # Rows: MoRF, LeRF, two random orders. Full bio: 3 + 2 - 1 = 4.
    # MoRF removes alpha (-3 -> drop 3), then beta (-2 -> drop 5).
    # LeRF removes gamma (+1 -> drop -1), then delta (no weight -> drop -1).
    assert curve.drop.shape == (4, 3)
    np.testing.assert_allclose(curve.drop[0], [0, 3, 5], atol=1e-6)
    np.testing.assert_allclose(curve.drop[1], [0, -1, -1], atol=1e-6)
    assert (curve.drop[:, 0] == 0).all()


def test_agreement_is_whether_the_call_is_still_the_call_on_the_full_bio():
    _, curve = _curve()
    # Full bio: nurse logit 4 beats the zero logits. After alpha goes it is 1: still nurse.
    # After alpha and beta it is -1: some zero-logit occupation wins.
    assert curve.agreement[0].tolist() == [1, 1, 0]
    assert curve.agreement[1].tolist() == [1, 1, 1]


def test_each_random_order_is_a_row_and_the_seed_fixes_them():
    _, one = _curve(n_random=3, seed=5)
    _, again = _curve(n_random=3, seed=5)
    assert one.drop.shape == (5, 3)
    np.testing.assert_array_equal(one.drop, again.drop)


def test_all_steps_of_a_bio_go_to_the_classifier_in_one_call_with_each_text_once():
    classifier, _ = _curve(n_random=4)
    (texts,) = classifier.calls
    assert len(texts) == len(set(texts))
    assert " ".join(WORDS) in texts


def test_removed_words_become_unk_and_the_rest_keep_their_place():
    classifier, _ = _curve(percents=(25,), n_random=0)
    (texts,) = classifier.calls
    assert sorted(texts) == sorted(
        ["[UNK] beta gamma delta", "alpha beta [UNK] delta", "alpha beta gamma delta"]
    )
