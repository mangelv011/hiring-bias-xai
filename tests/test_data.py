import pandas as pd
import pytest

from hiring_bias_xai import data


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She is a nurse and he is a doctor.", "is a nurse and is a doctor."),
        ("He and SHE and Him and hErself", "and and and"),
        ("Her bio and his book. Hers is long, himself too.", "bio and book. is long, too."),
        (
            "The shell of the heritage; hierarchy there. Sheriff history.",
            "The shell of the heritage; hierarchy there. Sheriff history.",
        ),
        ("This is her's and his' and she's here", "This is and and here"),
        ('They said (she) left, "he" stayed.', 'They said () left, "" stayed.'),
        ("Mr. Smith, Mrs Jones and Ms. Lee", "Smith, Jones and Lee"),
        ("Her’s and he’d", "and"),
        ("Ask him. Then she left.", "Ask . Then left."),
        ("line one\nshe line two", "line one\nline two"),
    ],
)
def test_scrub_removes_gendered_words_only(text, expected):
    assert data.scrub(text) == expected


def _synthetic_raw(n_rows=1000):
    rows = [
        {
            "id": f"x-{i}",
            "text": f"She wrote {i}",
            "occupation": ("nurse", "poet", "attorney")[i % 3],
            "gender": "female" if i % 7 < 2 else "male",
        }
        for i in range(n_rows)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(monkeypatch):
    raw = _synthetic_raw()
    monkeypatch.setattr(data, "_load_raw", lambda split: raw.assign(id=split + "-" + raw["id"]))
    monkeypatch.setitem(data._SUBSAMPLE_SIZES, "train", 101)
    monkeypatch.setitem(data._SUBSAMPLE_SIZES, "dev", 37)
    return raw


def test_load_split_returns_documented_columns(synthetic):
    df = data.load_split("dev")
    assert list(df.columns) == ["id", "text", "occupation", "gender"]


def test_train_and_dev_are_subsampled_to_exact_sizes_and_test_is_full(synthetic):
    assert len(data.load_split("train")) == 101
    assert len(data.load_split("dev")) == 37
    assert len(data.load_split("test")) == len(synthetic)


def test_subsample_is_deterministic_and_has_no_duplicate_ids(synthetic):
    first = data.load_split("train")
    second = data.load_split("train")
    pd.testing.assert_frame_equal(first, second)
    assert first["id"].is_unique


def test_subsample_follows_occupation_by_gender_proportions(synthetic):
    sample = data.load_split("train")
    full = synthetic.groupby(["occupation", "gender"]).size()
    got = sample.groupby(["occupation", "gender"]).size()
    expected = full / full.sum() * 101
    assert (got - expected).abs().max() < 1


def test_scrubbed_variant_scrubs_text_and_keeps_the_rest(synthetic):
    original = data.load_split("dev", variant="original")
    scrubbed = data.load_split("dev", variant="scrubbed")
    assert original["text"].str.startswith("She wrote").all()
    assert scrubbed["text"].str.startswith("wrote").all()
    pd.testing.assert_frame_equal(original.drop(columns="text"), scrubbed.drop(columns="text"))


def test_load_split_rejects_unknown_split_and_variant(synthetic):
    with pytest.raises(ValueError, match="split"):
        data.load_split("val")
    with pytest.raises(ValueError, match="variant"):
        data.load_split("dev", variant="masked")
