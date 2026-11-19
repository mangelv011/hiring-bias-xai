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


def test_xai_subset_is_a_stratified_subset_of_test(synthetic, monkeypatch):
    monkeypatch.setattr(data, "_XAI_SIZE", 53)
    subset = data.xai_subset()
    test = data.load_split("test")
    assert len(subset) == 53
    assert subset["id"].is_unique
    assert set(subset["id"]) <= set(test["id"])
    pd.testing.assert_frame_equal(subset, data.xai_subset())
    full = test.groupby(["occupation", "gender"]).size()
    got = subset.groupby(["occupation", "gender"]).size()
    assert (got - full / full.sum() * 53).abs().max() < 1


def test_xai_subset_larger_than_test_is_an_error(synthetic, monkeypatch):
    monkeypatch.setattr(data, "_XAI_SIZE", len(synthetic) + 1)
    with pytest.raises(ValueError, match="cannot sample"):
        data.xai_subset()


@pytest.mark.slow
def test_real_data_sizes_labels_and_gender_coding():
    train = data.load_split("train")
    assert len(train) == 60_000
    assert len(data.load_split("dev")) == 10_000
    assert len(data.load_split("test")) == 99_069
    assert len(data.xai_subset()) == 2_000
    assert set(train["occupation"]) == set(data.OCCUPATIONS)
    assert set(train["gender"]) == {"female", "male"}
    assert train["id"].str.startswith("train-").all()
    # Gender coding: female bios use she/her, male bios use he/his.
    female_text = train.loc[train["gender"] == "female", "text"].str.lower()
    assert female_text.str.contains(r"\b(?:she|her)\b").mean() > 0.85
    assert not female_text.str.contains(r"\b(?:he|his)\b").any()
    # Occupation coding.
    nurse_share = (train.loc[train["occupation"] == "nurse", "gender"] == "female").mean()
    assert nurse_share > 0.8
    assert (
        train.loc[train["occupation"] == "software_engineer", "text"]
        .str.contains("software", case=False)
        .mean()
        > 0.2
    )
    assert not data.load_split("train", "scrubbed")["text"].str.contains(r"\b(?:she|he)\b").any()


def _frame(counts):
    rows = [
        {"id": f"{o}-{g}-{i}", "text": "", "occupation": o, "gender": g}
        for (o, g), n in counts.items()
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_showcase_picks_three_per_side_among_occupations_with_enough_test_bios():
    shares = {"a": 5, "b": 10, "c": 20, "d": 50, "e": 80, "f": 90, "g": 95}  # women per 100
    train = _frame(
        {(o, g): n for o, w in shares.items() for g, n in (("female", w), ("male", 100 - w))}
    )
    test_counts = {o: 500 for o in shares}
    test_counts["a"] = 499  # the most male-dominated occupation is too small
    test = _frame({(o, "male"): n for o, n in test_counts.items()})
    result = data.showcase_occupations(train, test)
    assert [e["occupation"] for e in result["male_dominated"]] == ["b", "c", "d"]
    assert [e["occupation"] for e in result["female_dominated"]] == ["g", "f", "e"]
    assert result["male_dominated"][0]["share_women"] == 0.1
    assert "500" in result["rule"]
