import time

import numpy as np
import pandas as pd
import pytest

from hiring_bias_xai import data


def test_occupations_are_the_28_names_in_code_order():
    assert data.OCCUPATIONS == (
        "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
        "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
        "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
        "photographer", "physician", "poet", "professor", "psychologist", "rapper",
        "software_engineer", "surgeon", "teacher", "yoga_teacher",
    )  # fmt: skip


GENDERED_WORDS = [
    "he", "she", "her", "his", "him", "hers", "himself", "herself", "mr", "mrs", "ms",
]  # fmt: skip


@pytest.mark.parametrize("word", GENDERED_WORDS)
@pytest.mark.parametrize("style", [str.lower, str.capitalize, str.upper])
def test_scrub_removes_each_gendered_word_in_any_case(word, style):
    assert data.scrub(f"a {style(word)} b") == "a b"


def test_scrub_strips_the_ends():
    assert data.scrub("She is a nurse. ") == "is a nurse."



@pytest.mark.parametrize(
    "word",
    ["the", "shell", "heritage", "hierarchy", "there", "sheriff", "history", "ashes", "this"],
)
def test_scrub_keeps_words_that_contain_a_pronoun(word):
    text = f"the {word} was here"
    assert data.scrub(text) == text


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("she's a nurse", "a nurse"),
        ("He'd say so", "say so"),
        ("she'll go", "go"),
        ("his' book", "book"),
        ("the nurse and her's", "the nurse and"),
    ],
)
def test_scrub_removes_attached_pronoun_suffixes(text, expected, apostrophe):
    assert data.scrub(text.replace("'", apostrophe)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr. Smith is a dentist", "Smith is a dentist"),
        ("Mrs. Lee, a nurse", "Lee, a nurse"),
        ("ms. Ray", "Ray"),
        ("Dr. Mr. X", "Dr. X"),
    ],
)
def test_scrub_removes_the_period_after_a_title(text, expected):
    assert data.scrub(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Worked hard (she said) daily.", "Worked hard ( said) daily."),
        ("Nurse, she.", "Nurse, ."),
        ("Ask him, her, or his-own", "Ask , , or -own"),
        ("a  b   she   c", "a b c"),
    ],
)
def test_scrub_handles_punctuation_next_to_the_word_and_collapses_spaces(text, expected):
    assert data.scrub(text) == expected


@pytest.mark.parametrize(
    "unit",
    ["he'", "he" + "'s" * 3 + "x", " ", "m", "hee", "’he’", "Mr", "h e "],
)
def test_scrub_is_linear_on_hostile_input(unit):
    small, large = unit * 2_000 + "x", unit * 40_000 + "x"
    start = time.perf_counter()
    data.scrub(small)
    t_small = time.perf_counter() - start
    start = time.perf_counter()
    data.scrub(large)
    t_large = time.perf_counter() - start
    assert t_large < 5.0
    assert t_large < 100 * max(t_small, 0.01)


def test_load_split_rejects_an_unknown_split_naming_the_allowed_ones():
    with pytest.raises(ValueError, match="train.*dev.*test"):
        data.load_split("validation")


def test_load_split_rejects_an_unknown_variant_naming_the_allowed_ones():
    with pytest.raises(ValueError, match="original.*scrubbed"):
        data.load_split("test", variant="masked")


def _raw(rows):
    """Raw split as the dataset delivers it: (text, profession code, gender code) rows."""
    return pd.DataFrame(rows, columns=["hard_text", "profession", "gender"])


@pytest.fixture
def fake_raw(monkeypatch):
    def install(**splits):
        monkeypatch.setattr(data, "_load_raw", lambda split: splits[split])

    return install


def test_load_split_test_returns_the_full_split_with_named_labels_and_ids(fake_raw):
    fake_raw(test=_raw([("first bio", 13, 1), ("second bio", 24, 0), ("third bio", 0, 1)]))
    df = data.load_split("test")
    assert list(df.columns) == ["id", "text", "occupation", "gender"]
    assert df.to_dict("list") == {
        "id": ["test-0", "test-1", "test-2"],
        "text": ["first bio", "second bio", "third bio"],
        "occupation": ["nurse", "software_engineer", "accountant"],
        "gender": ["female", "male", "female"],
    }


def test_load_split_scrubbed_variant_scrubs_the_text_and_keeps_the_rest(fake_raw):
    fake_raw(test=_raw([("She is a nurse.", 13, 1), ("Mr. Lee builds code", 24, 0)]))
    original = data.load_split("test")
    scrubbed = data.load_split("test", variant="scrubbed")
    assert list(scrubbed["text"]) == ["is a nurse.", "Lee builds code"]
    assert list(original["text"]) == ["She is a nurse.", "Mr. Lee builds code"]
    assert scrubbed.drop(columns="text").equals(original.drop(columns="text"))


def _synthetic(groups):
    """Raw split from {(profession, gender): size}, rows interleaved in a fixed shuffled order."""
    rows = [(p, g) for (p, g), size in groups.items() for _ in range(size)]
    order = np.random.default_rng(7).permutation(len(rows))
    return _raw([(f"bio {i}", *rows[k]) for i, k in enumerate(order)])


def _counts(df):
    return df.groupby(["occupation", "gender"]).size().to_dict()


def test_train_is_a_60000_bio_subsample_with_stratified_quotas(fake_raw):
    # n * size / total = 6 * size / 7: 30000, 6000, 12000.857, 11999.143. The floors sum to
    # 59999; the one leftover goes to the larger remainder (architect male).
    fake_raw(train=_synthetic({(0, 0): 35000, (0, 1): 7000, (1, 0): 14001, (1, 1): 13999}))
    df = data.load_split("train")
    assert len(df) == 60000
    assert _counts(df) == {
        ("accountant", "male"): 30000,
        ("accountant", "female"): 6000,
        ("architect", "male"): 12001,
        ("architect", "female"): 11999,
    }
