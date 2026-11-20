import pytest

from hiring_bias_xai.counterfactual import RESIDUAL_NOUNS, first_names_in, residual_cues

NAMES = frozenset({"Maria", "Mark", "Will", "June", "Grace", "Lee"})


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


def test_a_first_name_of_the_list_used_as_a_proper_noun_is_a_cue():
    assert first_names_in("I work with Maria and Dr. Lee.", NAMES) == ["Lee", "Maria"]


def test_a_listed_name_that_is_a_verb_a_month_or_in_lower_case_is_not_a_cue():
    text = "We mark the date. Will you join? In June 2020 she showed grace, said maria."
    assert first_names_in(text, NAMES) == []


def test_residual_cues_adds_the_names_to_the_nouns():
    assert residual_cues("Maria is a mother.", NAMES) == ["Maria", "mother"]
    assert residual_cues("Maria is a mother.") == ["mother"]
