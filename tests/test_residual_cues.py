import pytest

from hiring_bias_xai.counterfactual import RESIDUAL_NOUNS, residual_cues


@pytest.mark.parametrize(
    "word",
    [
        "mother", "father", "wife", "husband", "actress", "actor", "businesswoman",
        "businessman", "daughter", "son", "sister", "brother", "girl", "boy", "woman",
        "man", "lady", "gentleman", "queen", "king", "chairwoman", "chairman",
    ],
)  # fmt: skip
def test_the_nouns_the_brief_names_are_residual_cues(word):
    assert residual_cues(f"She is a {word} of two.") == [word]


def test_nouns_are_matched_as_whole_words_in_any_case_and_plural_forms_count():
    assert residual_cues("MOTHER of two; Women in tech; the father's lab") == [
        "father",
        "mother",
        "women",
    ]
    assert residual_cues("A human, a German, a Roman and a manager at Mansion Inc.") == []


def test_a_bio_without_cues_has_none():
    assert residual_cues("She builds bridges.") == []


def test_every_cue_is_a_lower_case_word_and_listed_once():
    assert len(set(RESIDUAL_NOUNS)) == len(RESIDUAL_NOUNS)
    assert all(word.isalpha() and word.islower() for word in RESIDUAL_NOUNS)
