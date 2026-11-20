import numpy as np
import pandas as pd
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
    classifier, _ = _curve(percents=(25,), n_random=1)
    (texts,) = classifier.calls
    assert {"[UNK] beta gamma delta", "alpha beta [UNK] delta", "alpha beta gamma delta"} <= set(
        texts
    )


def test_the_bootstrap_interval_of_a_mean_is_the_2_5th_to_97_5th_percentile_of_resampled_means():
    values = np.arange(100.0)
    low, high = checks.bootstrap_ci(lambda rows: values[rows].mean(), len(values), 2000, seed=1)
    # standard error 28.9 / 10 = 2.89, so the 95% interval is about 49.5 +- 5.7
    assert low == pytest.approx(43.8, abs=0.7)
    assert high == pytest.approx(55.2, abs=0.7)


def test_the_bootstrap_follows_the_seed_and_resamples_bios_not_values():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    one = checks.bootstrap_ci(lambda rows: values[rows].mean(), 4, 50, seed=3)
    again = checks.bootstrap_ci(lambda rows: values[rows].mean(), 4, 50, seed=3)
    assert one == again
    seen = []
    checks.bootstrap_ci(lambda rows: seen.append(rows.copy()) or 0.0, 4, 5, seed=3)
    assert all(rows.shape == (4,) and rows.min() >= 0 and rows.max() <= 3 for rows in seen)


def test_the_bootstrap_of_a_curve_gives_one_bound_per_step():
    curves = np.array([[0.0, 1.0], [0.0, 3.0]])
    low, high = checks.bootstrap_ci(lambda rows: curves[rows].mean(axis=0), 2, 100, seed=0)
    assert low.shape == high.shape == (2,)
    np.testing.assert_allclose(low[0], 0)
    assert 1.0 <= low[1] <= high[1] <= 3.0


def test_identical_values_agree_completely():
    values = np.arange(12.0)
    assert checks.rank_agreement(values, values) == (1.0, 1.0)


def test_reversed_values_correlate_at_minus_one_and_share_the_top_ten_in_the_middle():
    values = np.arange(1.0, 13.0)  # 12 words: the top 10 by size are words 2 to 11
    spearman, overlap = checks.rank_agreement(values, values[::-1])
    assert spearman == pytest.approx(-1.0)
    assert overlap == 0.8  # reversed, the top 10 are words 0 to 9: eight in common


def test_the_top_words_are_the_largest_in_absolute_value_so_a_sign_flip_still_overlaps():
    one = np.array([3.0, -3.0, 0.1, 0.2])
    other = np.array([3.0, 3.0, 0.2, 0.1])
    assert checks.rank_agreement(one, other, top=2)[1] == 1.0


def test_a_bio_with_fewer_words_than_the_top_size_overlaps_over_all_its_words():
    assert checks.rank_agreement(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))[1] == 1.0


def test_a_correlation_that_is_undefined_is_nan():
    spearman, _ = checks.rank_agreement(np.zeros(5), np.arange(5.0))
    assert np.isnan(spearman)
    assert np.isnan(checks.rank_agreement(np.array([1.0]), np.array([2.0]))[0])


class _AnyAlpha(BagOfWords):
    """Nurse logit 5 when the bio has the word alpha at all: a second alpha is redundant."""

    def score(self, text):
        logits = np.zeros(28)
        logits[NURSE] = 5.0 if "alpha" in text.split() else 0.0
        return logits


def test_leave_one_out_is_the_fall_in_log_odds_when_that_one_word_is_deleted():
    classifier = BagOfWords({"nurse": WEIGHTS})
    values = checks.loo_values(classifier, WORDS, NURSE)
    np.testing.assert_allclose(values, [3.0, 2.0, -1.0, 0.0], atol=1e-6)
    assert len(classifier.calls) == 1


def test_a_redundant_word_has_no_leave_one_out_value_but_its_group_does():
    words = ["alpha", "beta", "alpha"]
    classifier = _AnyAlpha({})
    np.testing.assert_allclose(checks.loo_values(classifier, words, NURSE), [0, 0, 0], atol=1e-6)
    drop = checks.drop_without(classifier, words, [[0], [0, 2]], NURSE)
    np.testing.assert_allclose(drop, [0.0, 5.0], atol=1e-6)


def test_deleting_a_group_rejoins_the_other_words_with_single_spaces():
    classifier = BagOfWords({})
    checks.drop_without(classifier, ["a", "b", "c"], [[1]], NURSE)
    assert classifier.calls == [["a b c", "a c"]]


def _subset(sizes):
    rows = [
        (occupation, gender) for (occupation, gender), size in sizes.items() for _ in range(size)
    ]
    return pd.DataFrame(rows, columns=["occupation", "gender"])


SIZES = {("nurse", "male"): 6, ("nurse", "female"): 9, ("model", "male"): 7, ("model", "female"): 8}


def test_the_sample_takes_equal_shares_from_each_occupation_and_gender_remainder_first():
    subset = _subset(SIZES)
    picked = subset.iloc[checks.stratified_sample(subset, 10, seed=1)]
    counts = picked.groupby(["occupation", "gender"]).size()
    # 10 over 4 strata: 2 each, and the 2 left over go to the first two strata in sorted order
    assert counts.to_dict() == {
        ("model", "female"): 3,
        ("model", "male"): 3,
        ("nurse", "female"): 2,
        ("nurse", "male"): 2,
    }


def test_the_sample_is_sorted_positions_without_repeats_that_follow_the_seed():
    subset = _subset(SIZES)
    one = checks.stratified_sample(subset, 10, seed=1)
    assert one.tolist() == sorted(set(one.tolist()))
    np.testing.assert_array_equal(one, checks.stratified_sample(subset, 10, seed=1))
    assert not np.array_equal(one, checks.stratified_sample(subset, 10, seed=2))


def test_a_stratum_smaller_than_its_share_is_an_error_not_a_smaller_sample():
    subset = _subset({("nurse", "male"): 1, ("nurse", "female"): 9})
    with pytest.raises(ValueError, match="nurse"):
        checks.stratified_sample(subset, 10, seed=1)
