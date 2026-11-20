import math

import numpy as np
import pytest
from fakes import BagOfWords

from hiring_bias_xai import attribution, gap_decomposition
from hiring_bias_xai.config import SEED


def _attribution(text, weights, occupation="surgeon"):
    (saved,) = attribution.shapley(
        BagOfWords({occupation: weights}), [text], [occupation], 4, "delete", seed=0
    )
    return saved


def test_a_bio_sums_the_values_of_a_word_over_its_occurrences_and_keys_them_by_core():
    saved = _attribution(
        "She, care Care. care -", {"She,": 1.5, "care": 1.0, "Care.": 0.5, "-": 0.25}
    )
    bio = gap_decomposition.bio_from_attribution(saved, "female")
    assert bio.gender == "female"
    # "She," keeps its core "she"; three spellings of care add up; a unit without letters or
    # digits keeps its own text.
    assert bio.values == pytest.approx({"she": 1.5, "care": 2.5, "-": 0.25})
    assert bio.score == pytest.approx(4.25 - math.log(27))
    assert bio.empty == pytest.approx(-math.log(27))


# Hand-built occupation: women F1-F3, men M1-M2; the empty bio scores -1 for everyone, so the full
# score is -1 plus the sum of the values.
#   word  women's mean  men's mean  D       bios with it  I
#   a     (2+4+0)/3     (1+0)/2     3/2     3             (2+4+1)/3 = 7/3
#   b     (1+0+3)/3     (0+1)/2     5/6     3             (1+3+1)/3 = 5/3
#   c     (0+0+1)/3     (2+0)/2     -2/3    2             (1+2)/2   = 3/2
def _bio(gender, values):
    return gap_decomposition.Bio(gender, values, -1.0 + sum(values.values()), -1.0)


BIOS = [
    _bio("female", {"a": 2.0, "b": 1.0}),
    _bio("female", {"a": 4.0}),
    _bio("female", {"b": 3.0, "c": 1.0}),
    _bio("male", {"a": 1.0, "c": 2.0}),
    _bio("male", {"b": 1.0}),
]


def _by_word(table):
    return table.set_index("word")


def test_contribution_is_the_womens_mean_minus_the_mens_with_absent_words_counting_zero():
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=1).table)
    assert table["contribution"].to_dict() == pytest.approx({"a": 3 / 2, "b": 5 / 6, "c": -2 / 3})
    assert table["n_bios"].to_dict() == {"a": 3, "b": 3, "c": 2}


def test_importance_is_the_mean_over_the_bios_that_contain_the_word_of_both_genders():
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=1).table)
    assert table["importance"].to_dict() == pytest.approx({"a": 7 / 3, "b": 5 / 3, "c": 3 / 2})


def test_each_word_has_its_frequency_and_mean_value_when_present_for_women_and_men():
    # a: women's bios F1 and F2 (2 of 3, values 2 and 4), men's M1 (1 of 2, value 1);
    # b: women F1, F3 (values 1 and 3), men M2 (value 1); c: women F3 (value 1), men M1 (value 2).
    # D = frequency_women * value_women - frequency_men * value_men.
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=1).table)
    assert table["freq_female"].to_dict() == pytest.approx({"a": 2 / 3, "b": 2 / 3, "c": 1 / 3})
    assert table["freq_male"].to_dict() == pytest.approx({"a": 1 / 2, "b": 1 / 2, "c": 1 / 2})
    assert table["value_female"].to_dict() == pytest.approx({"a": 3.0, "b": 2.0, "c": 1.0})
    assert table["value_male"].to_dict() == pytest.approx({"a": 1.0, "b": 1.0, "c": 2.0})


def test_the_contribution_splits_into_a_usage_part_and_a_weight_part_that_add_up_to_it():
    # usage = (f_women - f_men) * mean of the two mean values; weight = mean of the two
    # frequencies * (value_women - value_men).
    #   a: (2/3 - 1/2) * (3 + 1)/2 = 1/3      (2/3 + 1/2)/2 * (3 - 1) = 7/6
    #   b: (2/3 - 1/2) * (2 + 1)/2 = 1/4      (2/3 + 1/2)/2 * (2 - 1) = 7/12
    #   c: (1/3 - 1/2) * (1 + 2)/2 = -1/4     (1/3 + 1/2)/2 * (1 - 2) = -5/12
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=1).table)
    assert table["usage"].to_dict() == pytest.approx({"a": 1 / 3, "b": 1 / 4, "c": -1 / 4})
    assert table["weight"].to_dict() == pytest.approx({"a": 7 / 6, "b": 7 / 12, "c": -5 / 12})
    assert (table["usage"] + table["weight"]).to_numpy() == pytest.approx(
        table["contribution"].to_numpy()
    )


def test_a_word_one_gender_never_uses_has_no_mean_value_there_and_the_split_still_adds_up():
    # guard: the split counts a missing mean value as 0, which its zero frequency makes exact
    bios = [_bio("female", {"x": 2.0}), _bio("male", {"y": 1.0})]
    table = _by_word(gap_decomposition.decompose(bios, min_bios=1).table)
    assert np.isnan(table.loc["x", "value_male"])
    assert table.loc["x", "freq_male"] == 0
    assert table.loc["x", ["usage", "weight"]].to_numpy() == pytest.approx([1.0, 1.0])
    assert table.loc["y", ["usage", "weight"]].to_numpy() == pytest.approx([-0.5, -0.5])


GENDERED_BIOS = [
    _bio("female", {"she": 5.0, "x": 2.0}),
    _bio("female", {"she": 4.0}),
    _bio("male", {"he": 1.0, "x": 1.0}),
    _bio("male", {"x": 0.0}),
]
# D: she = 9/2, he = -1/2, x = 1 - 1/2 = 1/2; the total is 9/2 and the gendered words carry 4.
# Every resample has a total gap above 2, so no share divides by zero.


def test_gendered_words_are_flagged_and_their_share_of_the_total_gap_is_reported():
    result = gap_decomposition.decompose(GENDERED_BIOS, min_bios=1)
    assert _by_word(result.table)["gendered"].to_dict() == {"she": True, "he": True, "x": False}
    assert result.summary["gendered_share"]["value"] == pytest.approx(8 / 9)


def test_top_share_is_the_sum_of_the_largest_absolute_contributions_over_the_total_gap():
    # D: a = 3/2, b = 5/6, c = -2/3; total 5/3. The top 2 by |D| are a and b: (3/2 + 5/6) / (5/3).
    result = gap_decomposition.decompose(BIOS, min_bios=1, top=2)
    assert result.summary["top_share"]["value"] == pytest.approx(7 / 5)
    # The top word is chosen by |D|, not by D: p = 1, q = -3, total -2, so q alone is 3/2.
    bios = [_bio("female", {"p": 1.0}), _bio("male", {"q": 3.0}), _bio("male", {"q": 3.0})]
    summary = gap_decomposition.decompose(bios, min_bios=1, top=1).summary
    assert summary["top_share"]["value"] == pytest.approx(-3 / -2)
    assert summary["top_words"] == ["q"]


def _resamples(bios, n_boot, seed):
    """The bootstrap of the documented procedure, drawn independently of the module: women's
    bios first, then men's, each resampled with replacement within its gender."""
    rng = np.random.default_rng(seed)
    women = [b for b in bios if b.gender == "female"]
    men = [b for b in bios if b.gender == "male"]
    draw_women = rng.integers(0, len(women), size=(n_boot, len(women)))
    draw_men = rng.integers(0, len(men), size=(n_boot, len(men)))
    return [
        ([women[i] for i in w], [men[i] for i in m])
        for w, m in zip(draw_women, draw_men, strict=True)
    ]


def _contributions(women, men):
    """D of every word by plain arithmetic over two lists of bios."""
    words = {word for bio in women + men for word in bio.values}
    return {
        w: sum(b.values.get(w, 0.0) for b in women) / len(women)
        - sum(b.values.get(w, 0.0) for b in men) / len(men)
        for w in words
    }


def _interval(values):
    return np.percentile(values, [2.5, 97.5])


def test_the_gaps_have_95_percent_intervals_from_resampling_bios_within_each_gender():
    resamples = _resamples(BIOS, 50, seed=7)
    score = [
        np.mean([b.score for b in women]) - np.mean([b.score for b in men])
        for women, men in resamples
    ]
    decomposed = [sum(_contributions(women, men).values()) for women, men in resamples]
    summary = gap_decomposition.decompose(BIOS, min_bios=1, n_boot=50, seed=7).summary
    assert [summary["score_gap"]["ci_low"], summary["score_gap"]["ci_high"]] == pytest.approx(
        _interval(score)
    )
    assert [
        summary["decomposed_gap"]["ci_low"],
        summary["decomposed_gap"]["ci_high"],
    ] == pytest.approx(_interval(decomposed))


def test_top_share_has_two_intervals_one_that_reselects_the_top_words_and_one_with_them_fixed():
    # D of "a" and "b" are close, so which one is the largest changes between resamples.
    bios = [
        _bio("female", {"a": 3.0, "b": 0.0}),
        _bio("female", {"a": 0.0, "b": 3.0}),
        _bio("female", {"a": 1.0, "b": 1.0}),
        _bio("male", {"a": 0.0, "b": 0.0}),
        _bio("male", {"a": 0.0, "b": 1.0}),
    ]
    resamples = _resamples(bios, 60, seed=3)
    reselected, fixed, tops = [], [], set()
    for women, men in resamples:
        d = _contributions(women, men)
        best = max(d, key=lambda w: abs(d[w]))
        tops.add(best)
        reselected.append(d[best] / sum(d.values()))
        fixed.append(d["a"] / sum(d.values()))  # "a" is the top word of the full sample
    assert tops == {"a", "b"}  # the data does make the top word change
    summary = gap_decomposition.decompose(bios, min_bios=1, top=1, n_boot=60, seed=3).summary
    assert summary["top_words"] == ["a"]
    share = summary["top_share"]
    assert [share["ci_low"], share["ci_high"]] == pytest.approx(_interval(reselected))
    assert [share["fixed_ci_low"], share["fixed_ci_high"]] == pytest.approx(_interval(fixed))


def test_the_gendered_sum_is_the_sum_of_the_gendered_contributions_with_an_interval_and_needs_no_division():
    # D: she = 9/2, he = -1/2, x = 1/2; the gendered words add up to 4.
    summary = gap_decomposition.decompose(GENDERED_BIOS, min_bios=1, n_boot=40, seed=5).summary
    sums = []
    for women, men in _resamples(GENDERED_BIOS, 40, seed=5):
        d = _contributions(women, men)
        sums.append(d.get("she", 0) + d.get("he", 0))
    gendered = summary["gendered_sum"]
    assert gendered["value"] == pytest.approx(4.0)
    assert [gendered["ci_low"], gendered["ci_high"]] == pytest.approx(_interval(sums))


def test_the_gendered_sum_is_defined_when_the_total_gap_is_exactly_zero():
    # D: she = 1, a = -1, total 0: the shares are empty, the gendered sum is 1.
    bios = [_bio("female", {"she": 1.0, "a": -1.0}), _bio("male", {"she": 0.0, "a": 0.0})]
    summary = gap_decomposition.decompose(bios, min_bios=1, n_boot=10).summary
    assert summary["gendered_share"]["value"] is None
    assert summary["gendered_sum"]["value"] == pytest.approx(1.0)


def test_the_gendered_share_has_an_interval_from_the_same_resamples():
    shares = []
    for women, men in _resamples(GENDERED_BIOS, 40, seed=5):
        d = _contributions(women, men)
        shares.append((d.get("she", 0) + d.get("he", 0)) / sum(d.values()))
    share = gap_decomposition.decompose(GENDERED_BIOS, min_bios=1, n_boot=40, seed=5).summary[
        "gendered_share"
    ]
    assert [share["ci_low"], share["ci_high"]] == pytest.approx(_interval(shares))


@pytest.mark.filterwarnings("error")
def test_shares_are_flagged_unstable_when_the_interval_of_the_total_gap_includes_zero():
    stable = gap_decomposition.decompose(GENDERED_BIOS, min_bios=1).summary
    assert stable["gap_unstable"] is False
    # Women's values cancel out and men's are 0: the total gap is exactly 0, so no share exists.
    bios = [
        _bio("female", {"a": 1.0}),
        _bio("female", {"a": -1.0}),
        _bio("male", {"a": 0.0}),
        _bio("male", {"a": 0.0}),
    ]
    unstable = gap_decomposition.decompose(bios, min_bios=1).summary
    assert unstable["gap_unstable"] is True
    assert unstable["decomposed_gap"]["value"] == 0
    assert unstable["top_share"]["value"] is None
    assert unstable["gendered_share"]["value"] is None


def test_each_contribution_has_a_95_percent_interval_from_the_same_resamples():
    draws = [_contributions(women, men) for women, men in _resamples(BIOS, 40, seed=9)]
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=1, n_boot=40, seed=9).table)
    for word in "abc":
        low, high = _interval([d.get(word, 0.0) for d in draws])
        assert table.loc[word, ["contribution_ci_low", "contribution_ci_high"]].to_numpy() == (
            pytest.approx([low, high])
        )


def test_the_resampling_is_reproducible_with_the_seed_and_defaults_to_the_project_seed():
    # guard: the oracle tests already pin the draws for a given seed; this pins the default
    first = gap_decomposition.decompose(BIOS, min_bios=1, n_boot=30, seed=1).summary
    again = gap_decomposition.decompose(BIOS, min_bios=1, n_boot=30, seed=1).summary
    other = gap_decomposition.decompose(BIOS, min_bios=1, n_boot=30, seed=2).summary
    assert first == again
    assert first["score_gap"] != other["score_gap"]
    default = gap_decomposition.decompose(BIOS, min_bios=1, n_boot=30).summary
    assert default == gap_decomposition.decompose(BIOS, min_bios=1, n_boot=30, seed=SEED).summary


def test_a_gap_that_is_zero_in_every_resample_is_unstable_because_its_interval_touches_zero():
    bios = [_bio("female", {"a": 0.0}), _bio("male", {"a": 0.0})]
    summary = gap_decomposition.decompose(bios, min_bios=1, n_boot=10).summary
    assert (summary["decomposed_gap"]["ci_low"], summary["decomposed_gap"]["ci_high"]) == (0, 0)
    assert summary["gap_unstable"] is True


def test_the_top_words_of_each_resample_are_chosen_by_absolute_contribution():
    # Women never vary and men never vary, so every resample has D: p = 1, q = -4, total -3.
    # The largest |D| is q: its share is -4 / -3 in every resample, not p's 1 / -3.
    bios = [
        _bio("female", {"p": 1.0}),
        _bio("female", {"p": 1.0}),
        _bio("male", {"q": 4.0}),
        _bio("male", {"q": 4.0}),
    ]
    share = gap_decomposition.decompose(bios, min_bios=1, top=1, n_boot=20).summary["top_share"]
    assert (share["ci_low"], share["ci_high"]) == pytest.approx((4 / 3, 4 / 3))


def test_a_share_of_a_total_that_is_exactly_zero_is_empty_even_when_its_part_is_not():
    # D: a = 1, b = -1, total 0; the top word a alone is 1 / 0.
    bios = [_bio("female", {"a": 1.0, "b": -1.0}), _bio("male", {"a": 0.0, "b": 0.0})]
    summary = gap_decomposition.decompose(bios, min_bios=1, top=1, n_boot=10).summary
    assert summary["top_share"]["value"] is None


def test_the_decomposition_error_is_the_size_of_the_disagreement_with_the_scores():
    # The values add up to 1 for the woman and 0 for the man, but her full score is 5 above her
    # empty score: sum of D = 1, women-minus-men mean of full minus empty = 5, error 4.
    woman = gap_decomposition.Bio("female", {"a": 1.0}, 5.0, 0.0)
    man = gap_decomposition.Bio("male", {"a": 0.0}, 0.0, 0.0)
    summary = gap_decomposition.decompose([woman, man], min_bios=1, n_boot=5).summary
    assert summary["decomposition_error"] == pytest.approx(4.0)


def test_intervals_ignore_resamples_whose_total_is_zero():
    # Women's values cancel out and men's are 0: half of the resamples have a total of exactly 0
    # (no share) and the others a total of +-1, where the share of "a" is 1 and a gendered share 0.
    bios = [
        _bio("female", {"a": 1.0}),
        _bio("female", {"a": -1.0}),
        _bio("male", {"a": 0.0}),
        _bio("male", {"a": 0.0}),
    ]
    summary = gap_decomposition.decompose(bios, min_bios=1, top=1, n_boot=200).summary
    top = summary["top_share"]
    assert [top["ci_low"], top["ci_high"], top["fixed_ci_low"], top["fixed_ci_high"]] == [1] * 4
    assert [summary["gendered_share"]["ci_low"], summary["gendered_share"]["ci_high"]] == [0, 0]


def test_the_score_gap_interval_resamples_the_full_scores_not_the_sums_of_values():
    empties = [-1.0, -2.0, -1.0, -0.5, -0.9]  # vary inside each gender
    bios = [
        gap_decomposition.Bio(b.gender, b.values, sum(b.values.values()) + e, e)
        for b, e in zip(BIOS, empties, strict=True)
    ]
    scores = [
        np.mean([b.score for b in women]) - np.mean([b.score for b in men])
        for women, men in _resamples(bios, 40, seed=4)
    ]
    summary = gap_decomposition.decompose(bios, min_bios=1, n_boot=40, seed=4).summary
    gap = summary["score_gap"]
    assert [gap["ci_low"], gap["ci_high"]] == pytest.approx(_interval(scores))


def test_the_top_words_are_listed_by_absolute_contribution_not_alphabetically():
    # D: y = 3, x = -1
    bios = [_bio("female", {"y": 3.0}), _bio("male", {"x": 1.0})]
    summary = gap_decomposition.decompose(bios, min_bios=1, top=2).summary
    assert summary["top_words"] == ["y", "x"]


def test_the_defaults_are_the_study_settings_20_bios_the_top_10_and_1000_resamples():
    assert (gap_decomposition.MIN_BIOS, gap_decomposition.TOP, gap_decomposition.N_BOOT) == (
        20,
        10,
        1000,
    )
    # 11 words in all 22 bios (women 11 x c_i, men 0) and one word in 19 bios with D = 0.
    bios = []
    for i in range(22):
        values = {f"c{k}": float(k + 1) if i < 11 else 0.0 for k in range(11)}
        if i < 19:
            values["rare"] = 0.0
        bios.append(_bio("female" if i < 11 else "male", values))
    table = gap_decomposition.decompose(bios, n_boot=10).table
    assert sorted(table["word"]) == sorted(f"c{k}" for k in range(11))  # rare: 19 bios < 20
    assert table["top"].sum() == 10


def test_the_contributions_add_up_to_the_gap_of_full_minus_empty_scores_and_drift_is_reported():
    # Same values as BIOS, but the empty bio scores -1.0 for women and -0.9 for men (as with a
    # classifier whose empty-bio logits vary by 1e-3 with the batch): the score gap of the full
    # bios is 5/3 - 1.0 + 0.9, while the contributions still add up to 5/3.
    bios = [
        gap_decomposition.Bio(b.gender, b.values, sum(b.values.values()) + e, e)
        for b, e in zip(BIOS, [-1.0] * 3 + [-0.9] * 2, strict=True)
    ]
    summary = gap_decomposition.decompose(bios, min_bios=1).summary
    assert summary["score_gap"]["value"] == pytest.approx(5 / 3 - 0.1)
    assert summary["decomposed_gap"]["value"] == pytest.approx(5 / 3)
    assert summary["empty_score_drift"] == pytest.approx(-0.1)
    assert summary["decomposition_error"] < 1e-12


def test_a_word_present_with_value_zero_is_in_the_bios_that_contain_it():
    # guard: n_bios and importance already count presence, not a non-zero value
    bios = [
        _bio("female", {"z": 0.0}),
        _bio("male", {"z": 0.0, "a": 1.0}),
        _bio("male", {"a": 3.0}),
    ]
    table = _by_word(gap_decomposition.decompose(bios, min_bios=1).table)
    assert table["n_bios"].to_dict() == {"z": 2, "a": 2}
    assert table["importance"].to_dict() == pytest.approx({"z": 0.0, "a": 2.0})


def test_a_row_is_kept_for_a_word_in_enough_bios_or_in_the_top_and_importance_needs_the_bios():
    table = _by_word(gap_decomposition.decompose(BIOS, min_bios=3, top=1).table)
    # c is in 2 bios and is not the largest |D|: no row. b is in 3: a row. a is the top word.
    assert list(table.index) == ["a", "b"]
    assert table["top"].to_dict() == {"a": True, "b": False}
    wide = _by_word(gap_decomposition.decompose(BIOS, min_bios=3, top=3).table)
    assert list(wide.index) == ["a", "b", "c"]  # c is in the top 3: a row, without importance
    assert wide["importance"].isna().to_dict() == {"a": False, "b": False, "c": True}
