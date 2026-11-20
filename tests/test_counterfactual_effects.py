import numpy as np
import pytest

from hiring_bias_xai import counterfactual_effects as effects

SHARE = {"nurse": 0.9, "surgeon": 0.15, "model": 0.8}


def test_a_flip_moves_toward_the_new_gender_when_the_share_of_women_follows_it():
    # A woman's bio becomes a man's: toward means a more male-dominated occupation.
    toward = effects.moves_toward_new_gender(
        gender=["female", "female", "male", "male"],
        before=["nurse", "surgeon", "surgeon", "nurse"],
        after=["surgeon", "nurse", "nurse", "surgeon"],
        share_women=SHARE,
    )
    assert toward.tolist() == [True, False, True, False]


def test_a_flip_between_occupations_with_the_same_share_does_not_move_toward():
    toward = effects.moves_toward_new_gender(
        gender=["female"], before=["nurse"], after=["nurse"], share_women=SHARE
    )
    assert toward.tolist() == [False]


def test_a_prediction_outside_the_share_table_is_an_error():
    with pytest.raises(KeyError):
        effects.moves_toward_new_gender(["male"], ["nurse"], ["poet"], SHARE)

