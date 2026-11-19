"""Loading, splitting and scrubbing Bias in Bios."""

import re

import numpy as np
import pandas as pd

from hiring_bias_xai.config import SEED

OCCUPATIONS = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
)  # fmt: skip

_GENDERED = re.compile(
    r"\b(?:(?:he|she|her|his|him|hers|himself|herself)(?:['\u2019](?:s|d|ll)?)?|(?:mr|mrs|ms)\.?)(?!\w)",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    return re.sub(" {2,}", " ", _GENDERED.sub("", text)).strip()


_SPLITS = ("train", "dev", "test")
_VARIANTS = ("original", "scrubbed")


def _load_raw(split: str) -> pd.DataFrame:
    raise NotImplementedError


_SUBSAMPLE_SIZES = {"train": 60_000, "dev": 10_000}


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
