import pandas as pd

from hiring_bias_xai import proxy_words

WOMEN = (0.5, 0.0)  # (freq_female, freq_male): a word used by women only
MEN = (0.0, 0.5)


def _table(*rows):
    """Rows of (word, contribution, ci_low, ci_high, gendered[, (freq_female, freq_male)]) of one
    occupation. Without the frequencies a word is used by women only."""
    return pd.DataFrame(
        [(*row[:5], *(row[5] if len(row) > 5 else WOMEN)) for row in rows],
        columns=[
            "word",
            "contribution",
            "contribution_ci_low",
            "contribution_ci_high",
            "gendered",
            "freq_female",
            "freq_male",
        ],
    )


def _words(table, gendered_sum=1.0, **kwargs):
    return proxy_words.proxy_rows(table, gendered_sum, **kwargs)["word"].tolist()


def test_a_gendered_word_is_never_a_proxy_word():
    table = _table(("she", 9.0, 8.0, 10.0, True), ("alpha", 1.0, 0.5, 1.5, False))
    assert _words(table) == ["alpha"]


def test_a_word_whose_interval_includes_zero_is_not_a_proxy_word():
    table = _table(
        ("alpha", 1.0, 0.5, 1.5, False),
        ("beta", 2.0, -0.1, 4.0, False),
        ("gamma", 2.0, 0.0, 4.0, False),
        ("delta", 2.0, -4.0, 0.0, False),
    )
    assert _words(table) == ["alpha"]


def test_a_proxy_word_pushes_the_gap_the_way_the_gendered_sum_does():
    table = _table(
        ("up", 1.0, 0.5, 1.5, False),
        ("down", -1.0, -1.5, -0.5, False, MEN),
    )
    assert _words(table, gendered_sum=0.3) == ["up"]
    assert _words(table, gendered_sum=-0.3) == ["down"]


def test_with_a_gendered_sum_of_exactly_zero_no_word_pushes_the_same_way():
    table = _table(("up", 1.0, 0.5, 1.5, False), ("down", -1.0, -1.5, -0.5, False, MEN))
    assert _words(table, gendered_sum=0.0) == []


def test_only_the_five_words_with_the_largest_absolute_contribution_are_kept_largest_first():
    # Seven passing words with contributions 1 to 7 in either sign; the gendered sum is positive,
    # so the negative ones drop out first and the five largest positive ones remain.
    table = _table(
        *[(f"w{n}", float(n), n - 0.5, n + 0.5, False) for n in range(1, 8)],
        ("big_negative", -50.0, -51.0, -49.0, False),
    )
    assert _words(table) == ["w7", "w6", "w5", "w4", "w3"]


def test_the_number_of_words_kept_is_the_top_argument():
    table = _table(*[(f"w{n}", float(n), n - 0.5, n + 0.5, False) for n in range(1, 8)])
    assert _words(table, top=2) == ["w7", "w6"]


def test_equal_absolute_contributions_are_ordered_alphabetically():
    table = _table(
        ("delta", 2.0, 1.0, 3.0, False),
        ("alpha", 2.0, 1.0, 3.0, False),
        ("charlie", 2.0, 1.0, 3.0, False),
        ("bravo", 2.0, 1.0, 3.0, False),
    )
    assert _words(table, top=3) == ["alpha", "bravo", "charlie"]


def test_the_largest_negative_contribution_ranks_by_its_size():
    table = _table(
        ("small", -1.0, -1.5, -0.5, False, MEN),
        ("large", -3.0, -3.5, -2.5, False, MEN),
        ("medium", -2.0, -2.5, -1.5, False, MEN),
    )
    assert _words(table, gendered_sum=-1.0) == ["large", "medium", "small"]


def test_a_word_used_about_equally_by_both_genders_is_not_a_proxy_word():
    # Smoothed ratios: (0.5 + 0.01) / (0.4 + 0.01) = 1.2 and (0.5 + 0.01) / (0.17 + 0.01) = 2.8,
    # below 3; (0.5 + 0.01) / (0.15 + 0.01) = 3.2 is above it.
    table = _table(
        ("same", 1.0, 0.5, 1.5, False, (0.5, 0.4)),
        ("below", 1.0, 0.5, 1.5, False, (0.5, 0.17)),
        ("above", 1.0, 0.5, 1.5, False, (0.5, 0.15)),
    )
    assert _words(table) == ["above"]


def test_the_ratio_is_smoothed_so_a_word_a_group_never_uses_does_not_divide_by_zero():
    # (0.05 + 0.01) / (0 + 0.01) = 6.
    assert _words(_table(("rare", 1.0, 0.5, 1.5, False, (0.05, 0.0)))) == ["rare"]


def test_a_word_the_other_gender_uses_more_is_not_a_proxy_word_even_if_it_pushes_the_gap():
    # The gap favours women (positive sum) and the contribution is positive, but men use the
    # word three times as often: it is not a way for the model to read that a bio is a woman's.
    table = _table(("mens", 1.0, 0.5, 1.5, False, (0.2, 0.6)), ("womens", 1.0, 0.5, 1.5, False))
    assert _words(table) == ["womens"]


def test_the_gender_that_must_use_the_word_more_is_the_one_the_gap_favours():
    table = _table(
        ("women_word", -1.0, -1.5, -0.5, False, WOMEN),
        ("men_word", -1.0, -1.5, -0.5, False, MEN),
    )
    assert _words(table, gendered_sum=-1.0) == ["men_word"]
    table = _table(
        ("women_word", 1.0, 0.5, 1.5, False, WOMEN),
        ("men_word", 1.0, 0.5, 1.5, False, MEN),
    )
    assert _words(table, gendered_sum=1.0) == ["women_word"]


def test_a_word_used_in_fewer_than_five_percent_of_the_favoured_gender_bios_is_not_a_proxy_word():
    table = _table(
        ("five", 1.0, 0.5, 1.5, False, (0.05, 0.0)),
        ("under", 1.0, 0.5, 1.5, False, (0.049, 0.0)),
        ("men_under", -1.0, -1.5, -0.5, False, (0.0, 0.049)),
        ("men_five", -1.0, -1.5, -0.5, False, (0.0, 0.05)),
    )
    assert _words(table, gendered_sum=1.0) == ["five"]
    assert _words(table, gendered_sum=-1.0) == ["men_five"]


def test_masking_deletes_the_words_of_the_list_whatever_their_case_and_punctuation():
    text = "Born in Ohio, She is Ranked (currently) first."
    assert proxy_words.mask_words(text, {"born", "ranked", "currently"}) == "in Ohio, She is first."


def test_masking_keeps_words_that_only_contain_a_word_of_the_list():
    assert proxy_words.mask_words("borne ranking the-born", {"born", "ranked"}) == (
        "borne ranking the-born"
    )


def test_masking_joins_what_is_left_with_single_spaces():
    assert proxy_words.mask_words("a  born\n\nb", {"born"}) == "a b"


def test_masking_with_an_empty_list_changes_only_the_spacing():
    assert proxy_words.mask_words("a  b", set()) == "a b"


def test_a_smoothed_ratio_of_exactly_three_is_enough():
    # (0.5 + 0.01) / (0.16 + 0.01) is 3 in floating point.
    assert _words(_table(("edge", 1.0, 0.5, 1.5, False, (0.5, 0.16)))) == ["edge"]


def test_the_smoothing_pulls_the_ratio_of_two_small_shares_towards_one():
    # Unsmoothed 0.31 / 0.10 is 3.1; smoothed (0.31 + 0.01) / (0.10 + 0.01) is 2.9.
    assert _words(_table(("shrunk", 1.0, 0.5, 1.5, False, (0.31, 0.10)))) == []


def test_a_word_that_pushes_against_the_gendered_sum_is_not_a_proxy_word_even_if_men_use_it():
    # The gap favours men (negative sum), men use the word, but its contribution is positive.
    table = _table(("against", 1.0, 0.5, 1.5, False, MEN), ("with", -1.0, -1.5, -0.5, False, MEN))
    assert _words(table, gendered_sum=-1.0) == ["with"]


def test_the_rows_returned_have_the_columns_of_the_table_they_came_from():
    table = _table(("alpha", 1.0, 0.5, 1.5, False))
    assert list(proxy_words.proxy_rows(table, 1.0).columns) == list(table.columns)
