import math
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data

GOLDEN_DEV_IDS = ["dev-x-46", "dev-x-64", "dev-x-66", "dev-x-83", "dev-x-90"]
GOLDEN_XAI_IDS = [
    "test-x-92", "test-x-97", "test-x-202", "test-x-432",
    "test-x-653", "test-x-694", "test-x-773", "test-x-864",
]  # fmt: skip


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
        ("She'll go and he'd stay", "go and stay"),
        ("a  she b", "a b"),
        ("It ended with her", "It ended with"),
        ("x\tshe\tleft", "x\tleft"),
    ],
)
def test_scrub_removes_gendered_words_only(text, expected):
    assert data.scrub(text) == expected


def test_scrub_runs_in_linear_time_on_hostile_input():
    # Bios come from third-party web data: long runs of spaces or near-matches must not stall.
    hostile = "he" * 200_000 + " " * 200_000 + "she's " * 50_000
    start = time.perf_counter()
    data.scrub(hostile)
    assert time.perf_counter() - start < 5


def test_download_is_pinned_to_a_revision_and_cached_under_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    calls = []

    class FakeDataset:
        def to_pandas(self):
            return pd.DataFrame({"hard_text": ["a", "b"], "profession": [13, 0], "gender": [1, 0]})

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeDataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    raw = data._load_raw("dev")
    ((args, kwargs),) = calls
    assert args == ("LabHC/bias_in_bios",)
    assert re.fullmatch(r"[0-9a-f]{40}", kwargs["revision"])
    assert kwargs["split"] == "dev"
    assert Path(kwargs["cache_dir"]).is_relative_to(tmp_path)
    assert raw.to_dict("list") == {
        "id": ["dev-0", "dev-1"],
        "text": ["a", "b"],
        "occupation": ["nurse", "accountant"],
        "gender": ["female", "male"],
    }


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
    original_order = [
        i for i in synthetic.pipe(lambda d: "train-" + d["id"]) if i in set(first["id"])
    ]
    assert list(first["id"]) == original_order


def test_samples_are_pinned_to_the_global_seed(synthetic, monkeypatch):
    # Regression pin on the numpy Generator stream: a change here means every
    # downstream result built on these samples changes too.
    assert list(data.load_split("dev")["id"][:5]) == GOLDEN_DEV_IDS
    monkeypatch.setattr(data, "_XAI_SIZE", 8)
    assert list(data.xai_subset()["id"]) == GOLDEN_XAI_IDS


@pytest.mark.parametrize("n", range(1, 40))
def test_sample_size_is_exact_and_groups_are_rounded_not_skewed(synthetic, monkeypatch, n):
    monkeypatch.setattr(data, "_XAI_SIZE", n)
    test = data.load_split("test")
    subset = data.xai_subset()
    full = test.groupby(["occupation", "gender"]).size()
    got = subset.groupby(["occupation", "gender"]).size().reindex(full.index, fill_value=0)
    exact = full * n / len(test)
    assert len(subset) == n
    assert ((got == exact.apply(math.floor)) | (got == exact.apply(math.ceil))).all()


def test_xai_subset_can_be_the_whole_test_split(synthetic, monkeypatch):
    monkeypatch.setattr(data, "_XAI_SIZE", len(synthetic))
    assert len(data.xai_subset()) == len(synthetic)


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
    test_ids = data.load_split("test")["id"]
    assert (test_ids.iloc[0], test_ids.iloc[-1]) == ("test-0", "test-99068")
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
    # (women, men) in the training subsample; names are not in share order.
    train_counts = {
        "d": (5, 95),
        "b": (1, 9),
        "f": (1, 2),
        "a": (20, 80),
        "e": (50, 50),
        "g": (80, 20),
        "c": (90, 10),
    }
    train = _frame(
        {
            (o, g): n
            for o, wm in train_counts.items()
            for g, n in zip(("female", "male"), wm, strict=True)
        }
    )
    test_counts = {o: 500 for o in train_counts}
    test_counts["d"] = 499  # the most male-dominated occupation is too small
    test = _frame({(o, "male"): n for o, n in test_counts.items()})
    result = data.showcase_occupations(train, test)
    assert [e["occupation"] for e in result["male_dominated"]] == ["b", "a", "f"]
    assert [e["occupation"] for e in result["female_dominated"]] == ["c", "g", "e"]
    assert result["male_dominated"][2]["share_women"] == 0.3333
    assert "500" in result["rule"]
