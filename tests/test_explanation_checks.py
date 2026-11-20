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


def _case(bio_id, words, shapley, loo):
    return checks.Case(bio_id, words, np.array(shapley, dtype=float), np.array(loo, dtype=float))


def _words(n, repeated="python"):
    return [repeated, "x1", repeated, *[f"w{i}" for i in range(n - 3)]]


def _ramp(n):
    return list(np.linspace(0.0, 1.0, n))


def test_the_example_is_the_lowest_spearman_among_bios_whose_top_word_occurs_twice():
    n = 16
    shapley = [3.0, *_ramp(n - 1)]  # the top word is word 0, "python", which occurs twice
    agree = _case("test-1", _words(n), shapley, shapley)
    swapped = list(shapley)
    swapped[5], swapped[6] = swapped[6], swapped[5]
    mild = _case("test-2", _words(n), shapley, swapped)
    worst = _case("test-3", _words(n), shapley, [3.0, *_ramp(n - 1)[::-1]])
    picked, spearman = checks.disagreement_example([agree, mild, worst])
    assert picked.bio_id == "test-3"
    assert spearman < checks.rank_agreement(mild.shapley, mild.loo)[0] < 1.0


def test_a_tie_goes_to_the_first_bio():
    n = 16
    shapley = [3.0, *_ramp(n - 1)]
    loo = [3.0, *_ramp(n - 1)[::-1]]
    twin = [_case("test-1", _words(n), shapley, loo), _case("test-2", _words(n), shapley, loo)]
    assert checks.disagreement_example(twin)[0].bio_id == "test-1"


@pytest.mark.parametrize(
    "words",
    [
        [f"w{i}" for i in range(16)],  # the top word occurs once
        _words(14),  # under 15 units
        [*_words(16)[:-1], "123456"],  # six digits in a row
    ],
)
def test_a_bio_that_breaks_a_rule_is_never_the_example(words):
    n = len(words)
    shapley = [3.0, *_ramp(n - 1)]
    broken = _case("test-1", words, shapley, [3.0, *_ramp(n - 1)[::-1]])
    fine = _case("test-2", _words(16), [3.0, *_ramp(15)], [3.0, *_ramp(15)[::-1]][:16])
    assert checks.disagreement_example([broken, fine])[0].bio_id == "test-2"


def test_the_top_word_is_the_largest_absolute_value_and_matches_without_case_or_punctuation():
    words = ["Python,", "x1", "python.", *[f"w{i}" for i in range(13)]]
    shapley = [-3.0, *_ramp(15)]  # a negative top value still counts
    case = _case("test-1", words, shapley, [3.0, *_ramp(15)[::-1]])
    assert checks.disagreement_example([case])[0].bio_id == "test-1"


def test_no_example_when_no_bio_qualifies():
    assert checks.disagreement_example([]) is None


def test_a_bio_whose_correlation_is_undefined_is_skipped():
    n = 16
    flat = _case("test-1", _words(n), [3.0, *_ramp(n - 1)], [0.0] * n)
    assert checks.disagreement_example([flat]) is None


def test_a_resample_whose_statistic_is_nan_is_left_out_of_the_interval():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    def mean_unless_first_is_zero(rows):
        return np.nan if rows[0] == 0 else values[rows].mean()

    low, high = checks.bootstrap_ci(mean_unless_first_is_zero, 4, 200, seed=2)
    assert np.isfinite(low) and np.isfinite(high)


def test_the_bootstrap_draws_one_thousand_resamples_by_default():
    seen = []
    checks.bootstrap_ci(lambda rows: seen.append(0) or 0.0, 4)
    assert len(seen) == 1000


def test_the_top_words_of_the_second_vector_are_also_ranked_by_absolute_value():
    one = np.array([3.0, 3.0, 0.1, 0.2])
    other = np.array([3.0, -3.0, 0.2, 0.1])
    assert checks.rank_agreement(one, other, top=2)[1] == 1.0


def test_a_constant_vector_gives_nan_without_a_warning_from_either_side():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(checks.rank_agreement(np.arange(5.0), np.zeros(5))[0])
        assert np.isnan(checks.rank_agreement(np.zeros(5), np.arange(5.0))[0])
        assert np.isnan(checks.rank_agreement(np.array([1.0]), np.array([2.0]))[0])


def test_a_group_with_one_bio_fewer_than_its_share_is_an_error_and_one_with_its_share_is_not():
    short = _subset({("nurse", "male"): 4, ("nurse", "female"): 5})
    with pytest.raises(ValueError, match="nurse"):
        checks.stratified_sample(short, 10, seed=1)
    exact = _subset({("nurse", "male"): 5, ("nurse", "female"): 5})
    assert len(checks.stratified_sample(exact, 10, seed=1)) == 10


def test_a_bio_of_exactly_15_units_can_be_the_example_and_one_of_14_cannot():
    def case(bio_id, n, loo_sign):
        shapley = [3.0, *_ramp(n - 1)]
        words = ["python", *[f"w{i}" for i in range(n - 2)], "python"]
        return _case(bio_id, words, shapley, [3.0, *(np.array(_ramp(n - 1)) * loo_sign)])

    fourteen = case("test-1", 14, -1)  # Spearman near -1: it would win if it were eligible
    fifteen = case("test-2", 15, 1)
    assert checks.disagreement_example([fourteen, fifteen])[0].bio_id == "test-2"
    assert checks.disagreement_example([fifteen])[0].bio_id == "test-2"


@pytest.mark.parametrize(("digits", "eligible"), [("12345", True), ("123456", False)])
def test_five_digits_in_a_row_are_fine_and_six_are_a_possible_identifier(digits, eligible):
    n = 16
    words = ["python", "python", digits, *[f"w{i}" for i in range(n - 3)]]
    shapley = [3.0, *_ramp(n - 1)]
    case = _case("test-1", words, shapley, [3.0, *_ramp(n - 1)[::-1]])
    assert (checks.disagreement_example([case]) is not None) is eligible


def test_the_default_bootstrap_seed_is_the_project_seed():
    values = np.arange(30.0)

    def statistic(rows):
        return values[rows].mean()

    assert checks.bootstrap_ci(statistic, 30, 100) == checks.bootstrap_ci(
        statistic, 30, 100, seed=42
    )
    assert checks.bootstrap_ci(statistic, 30, 100) != checks.bootstrap_ci(
        statistic, 30, 100, seed=7
    )


def test_ties_in_the_second_vector_go_to_the_words_that_come_first_in_a_long_bio():
    n = 200
    other = np.tile([1.0, 0.0], n // 2)  # the ones tie: the ten shown must be words 0, 2, ..., 18
    one = np.zeros(n)
    one[0:20:2] = np.arange(10.0, 0.0, -1.0)  # the same ten words, in the first vector
    assert checks.rank_agreement(one, other)[1] == 1.0
    assert checks.rank_agreement(other, one)[1] == 1.0


def test_a_punctuation_only_unit_is_its_own_word_so_two_different_dashes_are_not_a_repeat():
    n = 16
    words = ["-", "--", *[f"w{i}" for i in range(n - 2)]]
    case = _case("test-1", words, [3.0, *_ramp(n - 1)], [3.0, *_ramp(n - 1)[::-1]])
    assert checks.disagreement_example([case]) is None
    words[1] = "-"
    again = _case("test-1", words, [3.0, *_ramp(n - 1)], [3.0, *_ramp(n - 1)[::-1]])
    assert checks.disagreement_example([again]) is not None
