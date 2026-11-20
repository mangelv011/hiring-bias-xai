import pytest

from hiring_bias_xai.counterfactual import swap_gender


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("he runs", "she runs"),
        ("she runs", "he runs"),
        ("ask him", "ask her"),
        ("he did it himself", "she did it herself"),
        ("she did it herself", "he did it himself"),
        ("the desk is hers", "the desk is his"),
    ],
)
def test_pronouns_swap_both_ways(text, swapped):
    assert swap_gender(text) == swapped


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("He runs", "She runs"),
        ("SHE RUNS", "HE RUNS"),
        ("Ask Him. HIMSELF", "Ask Her. HERSELF"),
        ("sHe", "he"),
    ],
)
def test_capitalisation_is_kept(text, swapped):
    assert swap_gender(text) == swapped


def test_words_that_contain_pronouns_stay():  # guard: the regex already needs whole words
    text = "There the sheriff in the theme history, Herb, shed, Hershey, other, ashes"
    assert swap_gender(text) == text


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("he's here", "she's here"),
        ("She'd go", "He'd go"),
        ("he'll go", "she'll go"),
        ("she’s here", "he’s here"),
        ("he've gone", "she've gone"),
    ],
)
def test_contractions_swap_the_pronoun_and_keep_the_suffix(text, swapped):
    assert swap_gender(text) == swapped


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("Mr. Lee", "Ms. Lee"),
        ("Ms. Lee", "Mr. Lee"),
        ("Mrs Lee", "Mr Lee"),
        (
            "MS in computer science, 50 ms, MR imaging, mr",
            "MS in computer science, 50 ms, MR imaging, mr",
        ),
    ],
)
def test_titles_swap_but_degrees_and_units_stay(text, swapped):
    assert swap_gender(text) == swapped


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("She loves her job.", "He loves his job."),
        ("I told her the news.", "I told him the news."),
        ("The patients trust her", "The patients trust him"),
        ("Her work helped her.", "His work helped him."),
        ("He taught her students.", "She taught his students."),
        ("Call her.", "Call him."),
    ],
)
def test_her_is_his_as_a_possessive_and_him_as_an_object(text, swapped):
    assert swap_gender(text) == swapped


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("He loves his job.", "She loves her job."),
        ("The book is his.", "The book is hers."),
        ("That one is his and this one is hers.", "That one is hers and this one is his."),
        ("His is the best.", "Hers is the best."),
        ("He took her and his friends out.", "She took him and her friends out."),
        ("his own team", "her own team"),
    ],
)
def test_his_is_her_before_a_noun_and_hers_on_its_own(text, swapped):
    assert swap_gender(text) == swapped


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("(he) works", "(she) works"),
        ("she—her work", "he—his work"),
        ("herself,he", "himself,she"),
        ('"she" said', '"he" said'),
    ],
)
def test_pronouns_glued_to_punctuation_are_swapped(text, swapped):
    assert swap_gender(text) == swapped


def test_a_text_without_pronouns_is_returned_as_it_is():  # guard; nothing matches
    text = "Dr. Lee  teaches\nchemistry.\t"
    assert swap_gender(text) == text
    assert swap_gender("") == ""


def test_swapping_twice_restores_a_text_with_unambiguous_pronouns():
    text = "She says he trusts himself, and Mr. Lee thanks her mother for his gift."
    assert swap_gender(swap_gender(text)) == text.replace("Mr.", "Ms.").replace("Ms.", "Mr.")


def test_a_her_or_his_inside_a_longer_token_such_as_a_web_address_is_left_alone():
    assert swap_gender("See praxis-his.com for his book.") == "See praxis-his.com for her book."
    assert swap_gender("See www.her-site.org and her book.") == "See www.her-site.org and his book."


@pytest.mark.parametrize(
    ("text", "swapped"),
    [
        ("The book is his", "The book is hers"),
        ("The choice was his in the end.", "The choice was hers in the end."),
        ("She said it was his because he asked.", "He said it was hers because she asked."),
        ("That one is his to keep.", "That one is hers to keep."),
    ],
)
def test_his_is_hers_at_the_end_of_the_text_and_before_a_preposition_conjunction_or_particle(
    text, swapped
):
    assert swap_gender(text) == swapped
