import pandas as pd
import pytest

from hiring_bias_xai import counterfactual_effects as effects

SHARE = {"nurse": 0.9, "surgeon": 0.15, "model": 0.8}


def _scored():
    """Eight bios in two occupations with the predictions and log-odds before and after."""
    rows = [
        # occupation, gender, flagged, original, swapped, removed, lo (orig, swap, removed)
        ("nurse", "female", False, "nurse", "surgeon", "nurse", (2.0, 1.0, 1.5)),
        ("nurse", "female", True, "nurse", "nurse", "nurse", (2.0, 2.5, 2.0)),
        ("nurse", "male", False, "surgeon", "nurse", "surgeon", (0.0, 1.0, -1.0)),
        ("nurse", "male", False, "nurse", "nurse", "model", (1.0, 1.0, 0.0)),
        ("surgeon", "female", False, "nurse", "surgeon", "nurse", (-1.0, 0.0, -1.0)),
        ("surgeon", "male", False, "surgeon", "surgeon", "surgeon", (3.0, 2.0, 3.0)),
        ("surgeon", "male", True, "surgeon", "nurse", "surgeon", (3.0, 1.0, 2.0)),
        ("surgeon", "female", False, "surgeon", "model", "surgeon", (1.0, 1.0, 1.0)),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["occupation", "gender", "flagged", "original", "swapped", "removed", "lo"],
    )
    for position, name in enumerate(("lo_original", "lo_swapped", "lo_removed")):
        frame[name] = [row[position] for row in frame["lo"]]
    return frame.drop(columns="lo")


@pytest.fixture(scope="module")
def table():
    return effects.table(_scored(), SHARE, "deberta", n_boot=50)


def _row(table, occupation, subset, condition):
    (row,) = table[
        (table["occupation"] == occupation)
        & (table["subset"] == subset)
        & (table["condition"] == condition)
    ].to_dict("records")
    return row


def test_there_is_a_row_per_occupation_and_pooled_subset_and_condition(table):
    assert len(table) == 3 * 2 * 2
    assert set(table["screener"]) == {"deberta"}
    assert set(table["occupation"]) == {"nurse", "surgeon", "pooled"}
    assert set(table["subset"]) == {"all", "unflagged"}
    assert set(table["condition"]) == {"swap", "removal"}


def test_flip_rate_and_log_odds_change_are_taken_per_condition(table):
    swap = _row(table, "nurse", "all", "swap")
    assert (swap["n"], swap["n_flips"], swap["flip_rate"]) == (4, 2, 0.5)
    assert swap["mean_log_odds_change"] == pytest.approx((-1.0 + 0.5 + 1.0 + 0.0) / 4)
    removal = _row(table, "nurse", "all", "removal")
    assert (removal["n"], removal["n_flips"], removal["flip_rate"]) == (4, 1, 0.25)
    assert removal["mean_log_odds_change"] == pytest.approx((-0.5 + 0.0 - 1.0 - 1.0) / 4)


def test_unflagged_leaves_out_the_flagged_bios_and_pooled_joins_the_occupations(table):
    assert _row(table, "nurse", "unflagged", "swap")["n"] == 3
    assert _row(table, "pooled", "all", "swap")["n"] == 8
    assert _row(table, "pooled", "unflagged", "swap")["n"] == 6
    pooled = _row(table, "pooled", "all", "removal")
    assert pooled["n_flips"] == 1


def test_the_flip_direction_follows_the_share_of_women_and_the_removal_gets_it_as_a_reference(
    table,
):
    nurse = _row(table, "nurse", "all", "swap")
    # Flips: a woman's nurse bio -> surgeon (toward male: yes); a man's surgeon bio -> nurse
    # (toward female: yes).
    assert nurse["toward_share"] == 1.0
    surgeon = _row(table, "surgeon", "all", "swap")
    # Flips: a woman's nurse call -> surgeon (yes); a man's surgeon bio -> nurse (yes); a
    # woman's surgeon bio -> model (no: it turns more female, not more male).
    assert surgeon["toward_share"] == pytest.approx(2 / 3)
    # The removal has no new gender; its flips are read as if it were the opposite one. The only
    # nurse flip is a man's nurse bio -> model (a share of 0.9 -> 0.8 is not toward female).
    removal = _row(table, "nurse", "all", "removal")
    assert removal["n_flips"] == 1
    assert removal["toward_share"] == 0.0
    assert nurse["direction_basis"] == removal["direction_basis"] == effects.DIRECTION_BASIS
