import numpy as np
import pytest

from hiring_bias_xai import explanation_checks as checks


@pytest.mark.parametrize(
    ("n_words", "percent", "expected"),
    [
        (10, 0, 0),
        (10, 5, 1),  # 0.5 words rounds half up
        (10, 15, 2),  # 1.5 words rounds half up
        (58, 5, 3),  # 2.9 words
        (58, 10, 6),  # 5.8 words
        (59, 10, 6),  # 5.9 words
        (60, 7, 4),  # 4.2 words
        (17, 50, 9),  # 8.5 words rounds half up
        (1, 50, 1),
    ],
)
def test_the_share_of_words_to_remove_rounds_half_up_to_a_whole_word(n_words, percent, expected):
    assert checks.words_to_remove(n_words, percent) == expected
