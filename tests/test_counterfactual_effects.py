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



def test_summarise_gives_rates_and_means_with_intervals_that_contain_them():
    rng = np.random.default_rng(0)
    flip = rng.random(200) < 0.3
    change = rng.normal(-0.5, 1.0, 200)
    summary = effects.summarise(flip, change, n_boot=200)
    assert summary["n"] == 200
    assert summary["n_flips"] == int(flip.sum())
    assert summary["flip_rate"] == pytest.approx(flip.mean())
    assert summary["mean_log_odds_change"] == pytest.approx(change.mean())
    for name in ("flip_rate", "mean_log_odds_change"):
        assert summary[f"{name}_ci_low"] < summary[name] < summary[f"{name}_ci_high"]


def test_summarise_is_deterministic_and_has_no_direction_without_one():
    flip, change = np.array([True, False, False, True]), np.array([1.0, 0.0, -1.0, 2.0])
    first, second = effects.summarise(flip, change, n_boot=50), effects.summarise(flip, change, n_boot=50)
    np.testing.assert_equal(first, second)
    assert np.isnan(first["toward_share"])
    assert np.isnan(first["toward_share_ci_low"])


def test_summarise_degenerate_samples_have_degenerate_intervals():
    summary = effects.summarise(np.ones(5, dtype=bool), np.full(5, 2.0), n_boot=20)
    assert (summary["flip_rate_ci_low"], summary["flip_rate_ci_high"]) == (1.0, 1.0)
    assert (summary["mean_log_odds_change_ci_low"], summary["mean_log_odds_change_ci_high"]) == (
        2.0,
        2.0,
    )


def test_the_toward_share_is_taken_over_the_flips_and_empty_without_flips():
    flip = np.array([True, True, True, False, False])
    toward = np.array([True, True, False, False, False])
    summary = effects.summarise(flip, np.zeros(5), toward, n_boot=100)
    assert summary["toward_share"] == pytest.approx(2 / 3)
    assert summary["toward_share_ci_low"] <= 2 / 3 <= summary["toward_share_ci_high"]
    none = effects.summarise(np.zeros(5, dtype=bool), np.zeros(5), np.zeros(5, dtype=bool), 20)
    assert np.isnan(none["toward_share"])
    assert np.isnan(none["toward_share_ci_low"])
