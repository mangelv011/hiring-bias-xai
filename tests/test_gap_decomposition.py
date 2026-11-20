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


