"""Loading, splitting and scrubbing Bias in Bios."""

import re

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
_XAI_SUBSET_SIZE = 2_000

_SHOWCASE_MIN_TEST_BIOS = 500
_SHOWCASE_RULE = (
    "Among occupations with at least 500 test bios, the three with the lowest and "
    "the three with the highest share of women in the training subsample."
)

_GENDERED = re.compile(
    r"\b(?:(?:he|she|her|his|him|hers|himself|herself)(?:['\u2019](?:s|d|ll)?)?|(?:mr|mrs|ms)\.?)(?!\w)",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    return re.sub(" {2,}", " ", _GENDERED.sub("", text)).strip()


def _load_raw(split: str) -> pd.DataFrame:
    import datasets

    dataset = datasets.load_dataset(
        _DATASET, revision=_REVISION, split=split, cache_dir=str(artifacts_dir() / "hf")
    )
    return dataset.to_pandas()[["hard_text", "profession", "gender"]]


def _stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"cannot sample {n} bios from a split of {len(df)}")
    groups = df.groupby(["occupation", "gender"])
    sizes = groups.size().to_numpy()
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


def xai_subset() -> pd.DataFrame:
    return _stratified_sample(load_split("test"), _XAI_SUBSET_SIZE)


def showcase_occupations(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    test_bios = test["occupation"].value_counts()
    large = test_bios.index[test_bios >= _SHOWCASE_MIN_TEST_BIOS]
    share_women = (train["gender"] == "female").groupby(train["occupation"]).mean()
    share_women = share_women[share_women.index.isin(large)].sort_values(kind="stable")

    def entries(shares: pd.Series) -> list[dict]:
        return [
            {"occupation": occupation, "share_women": round(float(share), 4)}
            for occupation, share in shares.items()
        ]

    return {
        "rule": _SHOWCASE_RULE,
        "male_dominated": entries(share_women.head(3)),
        "female_dominated": entries(share_women.tail(3).iloc[::-1]),
    }
