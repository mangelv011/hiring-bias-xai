import math

import pytest
from fakes import BagOfWords

from hiring_bias_xai import attribution, gap_decomposition


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
    bios = [_bio("female", {"z": 0.0}), _bio("male", {"z": 0.0, "a": 1.0}), _bio("male", {"a": 3.0})]
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


