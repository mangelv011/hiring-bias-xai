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


def test_scrub_deletes_the_word_without_leaving_a_space_behind():
    assert data.scrub("(she) and x,he.") == "() and x,."


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


def test_dev_is_a_10000_bio_subsample_and_ties_go_to_the_first_group_in_sorted_order(fake_raw):
    # Three equal groups: floor(10000 / 3) = 3333 each, one leftover, all remainders tie.
    # Sorted (occupation, gender) order starts with accountant female.
    fake_raw(dev=_synthetic({(1, 0): 5000, (0, 0): 5000, (0, 1): 5000}))
    df = data.load_split("dev")
    assert _counts(df) == {
        ("accountant", "female"): 3334,
        ("accountant", "male"): 3333,
        ("architect", "male"): 3333,
    }


def test_subsample_draws_per_group_from_one_seeded_generator_and_keeps_input_order(
    fake_raw, monkeypatch
):
    # 10 accountant female rows (positions 0-9) then 6 accountant male rows (10-15); n = 5.
    # Quotas: floor(5 * 10 / 16) = 3 and floor(5 * 6 / 16) = 1; the leftover goes to the larger
    # remainder (male), so 3 and 2. With rng = default_rng(42), choice(arange(10), 3, replace=False)
    # gives [9, 0, 6] and the next choice(arange(10, 16), 2, replace=False) gives [14, 10].
    monkeypatch.setitem(data._SUBSAMPLE_SIZES, "train", 5)
    fake_raw(train=_raw([(f"bio {i}", 0, 1 if i < 10 else 0) for i in range(16)]))
    df = data.load_split("train")
    assert list(df["id"]) == ["train-0", "train-6", "train-9", "train-10", "train-14"]
    assert list(df["text"]) == ["bio 0", "bio 6", "bio 9", "bio 10", "bio 14"]
    assert list(df.index) == [0, 1, 2, 3, 4]
    assert df.equals(data.load_split("train"))


def test_xai_subset_is_a_2000_bio_stratified_sample_of_the_full_test_split(fake_raw):
    # n / total = 2000 / 8000 = 1/4: floors 1000, 499, 250, 250 sum to 1999; the leftover goes
    # to the largest remainder (accountant male, 6000 of 8000).
    fake_raw(test=_synthetic({(0, 1): 4001, (0, 0): 1999, (1, 0): 1000, (1, 1): 1000}))
    subset = data.xai_subset()
    assert _counts(subset) == {
        ("accountant", "female"): 1000,
        ("accountant", "male"): 500,
        ("architect", "male"): 250,
        ("architect", "female"): 250,
    }
    test = data.load_split("test").set_index("id")
    assert subset.set_index("id").equals(test.loc[subset["id"]])
    positions = [int(i.split("-")[1]) for i in subset["id"]]
    assert positions == sorted(positions)


def test_raw_split_is_downloaded_from_the_pinned_revision_into_the_artifacts_dir(
    monkeypatch, tmp_path
):
    calls = []

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame(
                {"hard_text": ["a bio"], "profession": [13], "gender": [1], "extra": ["x"]}
            )

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeDataset()

    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    raw = data._load_raw("dev")
    assert calls == [
        (
            ("LabHC/bias_in_bios",),
            {
                "revision": "052f01de644dba841176e0449528b41f27d94a61",
                "split": "dev",
                "cache_dir": str(tmp_path / "hf"),
            },
        )
    ]
    assert list(raw.columns) == ["hard_text", "profession", "gender"]
    assert raw.to_dict("records") == [{"hard_text": "a bio", "profession": 13, "gender": 1}]


def _bios(per_occupation):
    """Frame with `occupation` and `gender` columns from {occupation: (n_women, n_bios)}."""
    return pd.DataFrame(
        [
            (occupation, "female" if i < women else "male")
            for occupation, (women, total) in per_occupation.items()
            for i in range(total)
        ],
        columns=["occupation", "gender"],
    )


def test_share_women_is_the_fraction_of_female_bios_in_each_occupation():
    frame = _bios({"nurse": (2, 3), "surgeon": (0, 4), "model": (5, 5)})
    shares = data.share_women(frame)
    assert shares == {"model": 1.0, "nurse": 2 / 3, "surgeon": 0.0}
    assert all(type(share) is float for share in shares.values())


def test_showcase_occupations_take_the_three_most_imbalanced_each_way_among_large_ones():
    # Women's share in train: rapper .05, software_engineer .1, surgeon .2, dj 1/3, painter .4,
    # teacher .5, dietitian 2/3, paralegal .8, nurse .9. Comedian (share 0) and model (share 1)
    # have 499 test bios, one short of the cut-off; nurse has exactly 500.
    train = _bios(
        {
            "rapper": (5, 100), "software_engineer": (10, 100), "surgeon": (20, 100),
            "dj": (1, 3), "painter": (40, 100), "teacher": (50, 100), "dietitian": (2, 3),
            "paralegal": (80, 100), "nurse": (90, 100), "comedian": (0, 100), "model": (100, 100),
        }
    )  # fmt: skip
    test = _bios({occupation: (0, 500) for occupation in set(train["occupation"])})
    test = test[~test["occupation"].isin(["comedian", "model"])]
    test = pd.concat([test, _bios({"comedian": (0, 499), "model": (0, 499)})])
    assert data.showcase_occupations(train, test) == {
        "rule": (
            "Among occupations with at least 500 test bios, the three with the lowest and "
            "the three with the highest share of women in the training subsample."
        ),
        "male_dominated": [
            {"occupation": "rapper", "share_women": 0.05},
            {"occupation": "software_engineer", "share_women": 0.1},
            {"occupation": "surgeon", "share_women": 0.2},
        ],
        "female_dominated": [
            {"occupation": "nurse", "share_women": 0.9},
            {"occupation": "paralegal", "share_women": 0.8},
            {"occupation": "dietitian", "share_women": 0.6667},
        ],
    }


@pytest.mark.slow
def test_real_data_has_the_official_split_sizes_and_the_subsample_sizes():
    sizes = {split: len(data._load_raw(split)) for split in ("train", "dev", "test")}
    assert sizes == {"train": 257_478, "dev": 39_642, "test": 99_069}
    assert len(data.load_split("train")) == 60_000
    assert len(data.load_split("dev")) == 10_000
    assert len(data.load_split("test")) == 99_069
    assert len(data.xai_subset()) == 2_000


@pytest.mark.slow
def test_real_gender_code_1_is_female_and_0_is_male():
    raw = data._load_raw("dev")
    feminine = raw["hard_text"].str.contains(r"\b(?:she|her|hers|herself)\b", case=False)
    masculine = raw["hard_text"].str.contains(r"\b(?:he|his|him|himself)\b", case=False)
    code_1, code_0 = raw["gender"] == 1, raw["gender"] == 0
    assert feminine[code_1].mean() > 0.9
    assert not masculine[code_1].any()
    assert masculine[code_0].mean() > 0.9
    assert not feminine[code_0].any()


def test_subsample_as_large_as_the_split_keeps_every_bio(fake_raw, monkeypatch):
    monkeypatch.setitem(data._SUBSAMPLE_SIZES, "dev", 6)
    fake_raw(dev=_raw([(f"bio {i}", i % 2, i % 2) for i in range(6)]))
    df = data.load_split("dev")
    assert list(df["id"]) == [f"dev-{i}" for i in range(6)]


def test_subsample_larger_than_the_split_raises(fake_raw, monkeypatch):
    monkeypatch.setitem(data._SUBSAMPLE_SIZES, "dev", 7)
    fake_raw(dev=_raw([(f"bio {i}", 0, 1) for i in range(6)]))
    with pytest.raises(ValueError, match="7.*6"):
        data.load_split("dev")
