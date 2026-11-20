import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "word_units.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("word_units", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def units():
    return _load_script()


@pytest.mark.parametrize(
    "word",
    [
        "shoot...she",
        "Humor/Romance",
        "she/her/hers",
        "Conference.He.",
        "well-known",
        "cars---he",
        "age—his",
    ],
)
def test_a_unit_with_letters_on_both_sides_of_inner_punctuation_is_glued(units, word):
    assert units.is_glued(word)


@pytest.mark.parametrize(
    "word",
    ["she", "(She)", "it's", "it’s", "12,137", "2/20/2013", "etc.).", "Rated:", "“they", "C3PO"],
)
def test_other_units_are_not_glued(units, word):
    assert not units.is_glued(word)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("shoot...she", True),
        ("she/her/hers", True),
        ("Conference.He.", True),
        ("he's.His", True),
        ("Humor/Romance", False),
        ("sheriff/deputy", False),
    ],
)
def test_a_glued_unit_has_a_gendered_part_when_a_piece_between_punctuation_is_a_scrubbed_word(
    units, word, expected
):
    assert units.has_gendered_part(word) is expected


def test_the_statistics_count_units_glued_units_and_the_gendered_ones_among_them(units):
    # Units: She said shoot...she and Humor/Romance. He's here: 7 in all. Glued: shoot...she and
    # Humor/Romance. Gendered units, counting the glued one that holds "she": She, He's and
    # shoot...she, so a third of them are glued.
    stats = units.unit_statistics(["She said shoot...she and Humor/Romance. He's here"])
    assert stats == {
        "n_units": 7,
        "n_glued": 2,
        "share_glued": 2 / 7,
        "n_gendered_units": 2,
        "n_glued_with_gendered_part": 1,
        "share_of_gendered_units_glued": 1 / 3,
    }


def test_the_statistics_add_up_over_bios(units):
    stats = units.unit_statistics(["a b.c", "She x", "x/y he/she"])
    assert stats["n_units"] == 6
    assert stats["n_glued"] == 3
    assert stats["n_gendered_units"] == 1
    assert stats["n_glued_with_gendered_part"] == 1


def test_main_writes_the_statistics_of_the_showcase_subset(units, monkeypatch, tmp_path):
    subset = pd.DataFrame(
        {
            "id": ["test-1", "test-2"],
            "text": ["She said shoot...she", "He is here"],
            "occupation": ["a", "b"],
        }
    )
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    (tmp_path / "showcase_subset_ids.csv").write_bytes(b"id\ntest-1\ntest-2\n")
    units.main(tmp_path)
    record = json.loads((tmp_path / "word_units.json").read_text())
    assert record["n_units"] == 6
    assert record["n_glued"] == 1
    assert record["n_gendered_units"] == 2
    assert record["n_glued_with_gendered_part"] == 1
    assert "whitespace" in record["rule"]
