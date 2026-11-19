"""Loading, splitting and scrubbing Bias in Bios."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from hiring_bias_xai.config import SEED, artifacts_dir

OCCUPATIONS = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
)  # fmt: skip

_SPLITS = ("train", "dev", "test")
_VARIANTS = ("original", "scrubbed")

_DATASET = "LabHC/bias_in_bios"
_REVISION = "052f01de644dba841176e0449528b41f27d94a61"

_SUBSAMPLE_SIZES = {"train": 60_000, "dev": 10_000}
_SHOWCASE_PER_GROUP = 150

_SHOWCASE_MIN_TEST_BIOS = 500
_SHOWCASE_RULE = (
    "Among occupations with at least 500 test bios, the three with the lowest and "
    "the three with the highest share of women in the training subsample."
)

# One alternation with no nested quantifiers, so matching stays linear on hostile text.
_GENDERED = re.compile(
    r"\b(?:(?:he|she|her|his|him|hers|himself|herself)(?:['\u2019](?:s|d|ll)?)?|(?:mr|mrs|ms)\.?)(?!\w)",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    """Remove gendered pronouns and titles (De-Arteaga et al. 2019 list) as whole words."""
    return re.sub(" {2,}", " ", _GENDERED.sub("", text)).strip()


# Importing datasets is slow, so only the download path pays for it.
def _load_raw(split: str) -> pd.DataFrame:
    import datasets

    dataset = datasets.load_dataset(
        _DATASET, revision=_REVISION, split=split, cache_dir=str(artifacts_dir() / "hf")
    )
    return dataset.to_pandas()[["hard_text", "profession", "gender"]]


# Quotas are floor(n * size / total); leftover bios go to the largest remainders, ties to
# the first group in sorted order. Each group draws from one seeded generator, in that order.
def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"cannot sample {n} bios from a split of {len(df)}")
    groups = df.groupby(["occupation", "gender"])
    sizes = groups.size().to_numpy()
    # groupby sorts its keys, so sizes, remainders and indices line up by group.
    quotas = n * sizes // len(df)
    leftover = n - quotas.sum()
    by_remainder = np.argsort(-(n * sizes % len(df)), kind="stable")
    quotas[by_remainder[:leftover]] += 1
    rng = np.random.default_rng(SEED)
    chosen = [
        rng.choice(positions, size=quota, replace=False)
        for positions, quota in zip(groups.indices.values(), quotas, strict=True)
    ]
    return df.iloc[np.sort(np.concatenate(chosen))].reset_index(drop=True)


def load_split(split: str, variant: str = "original") -> pd.DataFrame:
    """Bios with columns id, text, occupation, gender, in official row order.

    train and dev are the training subsamples (60,000 and 10,000 bios); test is the full split.
    variant="scrubbed" applies `scrub` to the text.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {', '.join(_SPLITS)}; got {split!r}")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(_VARIANTS)}; got {variant!r}")
    raw = _load_raw(split)
    texts = raw["hard_text"]
    if variant == "scrubbed":
        texts = texts.map(scrub)
    df = pd.DataFrame(
        {
            "id": [f"{split}-{i}" for i in range(len(raw))],
            "text": texts,
            "occupation": [OCCUPATIONS[code] for code in raw["profession"]],
            "gender": ["female" if code == 1 else "male" for code in raw["gender"]],
        }
    )
    if split in _SUBSAMPLE_SIZES:
        df = _stratified_sample(df, _SUBSAMPLE_SIZES[split])
    return df


def showcase_subset(test: pd.DataFrame, occupations: Sequence[str]) -> pd.DataFrame:
    """150 test bios per occupation in `occupations` and gender, in official row order.

    Groups are visited in sorted (occupation, gender) order and each draws from one generator
    seeded with `SEED`, so the result does not depend on the order of `occupations`.
    """
    chosen_occupations = test[test["occupation"].isin(occupations)]
    groups = chosen_occupations.groupby(["occupation", "gender"])
    rng = np.random.default_rng(SEED)
    chosen = []
    for (occupation, gender), positions in groups.indices.items():
        if len(positions) < _SHOWCASE_PER_GROUP:
            raise ValueError(
                f"{occupation} {gender}: {len(positions)} test bios, {_SHOWCASE_PER_GROUP} needed"
            )
        chosen.append(rng.choice(positions, size=_SHOWCASE_PER_GROUP, replace=False))
    return chosen_occupations.iloc[np.sort(np.concatenate(chosen))].reset_index(drop=True)


def share_women(frame: pd.DataFrame) -> dict:
    """Share of female bios in each occupation of `frame`, keyed by occupation."""
    return (frame["gender"] == "female").groupby(frame["occupation"]).mean().to_dict()


def showcase_occupations(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """The rule and the three most male- and female-dominated eligible occupations, JSON-ready."""
    test_bios = test["occupation"].value_counts()
    large = test_bios.index[test_bios >= _SHOWCASE_MIN_TEST_BIOS]
    shares = pd.Series(share_women(train))
    shares = shares[shares.index.isin(large)].sort_values(kind="stable")

    def entries(picked: pd.Series) -> list[dict]:
        return [
            {"occupation": occupation, "share_women": round(float(share), 4)}
            for occupation, share in picked.items()
        ]

    return {
        "rule": _SHOWCASE_RULE,
        "male_dominated": entries(shares.head(3)),
        "female_dominated": entries(shares.tail(3).iloc[::-1]),
    }
