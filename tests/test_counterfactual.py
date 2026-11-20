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
        ("sHe", "she"),
    ],
)
def test_capitalisation_is_kept(text, swapped):
    assert swap_gender(text) == swapped


def test_words_that_contain_pronouns_stay():  # guard: the regex already needs whole words
    text = "There the sheriff in the theme history, Herb, shed, Hershey, other, ashes"
    assert swap_gender(text) == text
