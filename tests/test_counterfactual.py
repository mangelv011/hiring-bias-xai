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
