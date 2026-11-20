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
        ("MS in computer science, 50 ms, MR imaging, mr", "MS in computer science, 50 ms, MR imaging, mr"),
    ],
)
def test_titles_swap_but_degrees_and_units_stay(text, swapped):
    assert swap_gender(text) == swapped
