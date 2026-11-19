"""Loading, splitting and scrubbing Bias in Bios."""

import re

import numpy as np
import pandas as pd

from hiring_bias_xai.config import SEED, artifacts_dir

_DATASET = "LabHC/bias_in_bios"
# Pinned: a moved or edited upstream dataset must not change results silently.
_REVISION = "052f01de644dba841176e0449528b41f27d94a61"

# Label coding from the dataset card at the pinned revision (README.md of the dataset repo).
# The int -> name mapping is not stored in the data itself.
OCCUPATIONS = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist",
    "dietitian", "dj", "filmmaker", "interior_designer", "journalist", "model", "nurse",
    "painter", "paralegal", "pastor", "personal_trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software_engineer", "surgeon", "teacher",
    "yoga_teacher",
)  # fmt: skip
# Card: Male = 0, Female = 1. Checked against the pronouns in the train split (see data card).
_GENDERS = ("male", "female")

_SUBSAMPLE_SIZES = {"train": 60_000, "dev": 10_000}
_SPLITS = ("train", "dev", "test")
_VARIANTS = ("original", "scrubbed")

# De-Arteaga et al. (2019), arXiv:1901.09451, section 4: "we deleted the subject's first name,
# along with the words he, she, her, his, him, hers, himself, herself, mr, mrs, and ms".
# The bios carry a first name only inside the text, so only the listed words are removed.
_PRONOUNS = ("he", "she", "her", "his", "him", "hers", "himself", "herself")
_TITLES = ("mr", "mrs", "ms")
# A pronoun may carry a possessive or contraction ("her's", "his'", "he'd"); a title, its period.
_SCRUB_PATTERN = re.compile(
    rf"\b(?:(?:{'|'.join(_PRONOUNS)})\b(?:['’](?:s|d|ll|ve)?)?|(?:{'|'.join(_TITLES)})\b\.?)[ \t]*",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    """Remove gendered pronouns and titles as whole words, case-insensitively."""
    return re.sub(" {2,}", " ", _SCRUB_PATTERN.sub("", text)).strip()


def _load_raw(split: str) -> pd.DataFrame:
    """The full official split with columns id, text, occupation, gender."""
    from datasets import load_dataset

    ds = load_dataset(
        _DATASET, revision=_REVISION, split=split, cache_dir=str(artifacts_dir() / "hf")
    )
    raw = ds.to_pandas()
    return pd.DataFrame(
        {
            "id": [f"{split}-{i}" for i in range(len(raw))],
            "text": raw["hard_text"],
            "occupation": raw["profession"].map(dict(enumerate(OCCUPATIONS))),
            "gender": raw["gender"].map(dict(enumerate(_GENDERS))),
        }
    )


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n rows, each occupation x gender group in proportion to its size, totals exact.

    Each group gets floor(n * size / total); the leftover rows go to the groups with the
    largest remainders, ties broken by group order.
    """
    if n > len(df):
        raise ValueError(f"cannot sample {n} rows from {len(df)}")
    groups = df.groupby(["occupation", "gender"]).indices
    keys = sorted(groups)
    scaled = np.array([len(groups[k]) * n for k in keys])
    quota = scaled // len(df)
    leftover = n - quota.sum()
    quota[np.argsort(-(scaled % len(df)), kind="stable")[:leftover]] += 1
    rng = np.random.default_rng(seed)
    picked = np.concatenate(
        [rng.choice(groups[k], size=q, replace=False) for k, q in zip(keys, quota, strict=True)]
    )
    return df.iloc[np.sort(picked)].reset_index(drop=True)


def load_split(split: str, variant: str = "original") -> pd.DataFrame:
    """Bios with columns id, text, occupation, gender.

    Train and dev are the stratified subsamples (60,000 and 10,000 bios); test is the full
    official split. `variant="scrubbed"` applies `scrub` to the text.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")
    df = _load_raw(split)
    if split in _SUBSAMPLE_SIZES:
        df = _stratified_sample(df, _SUBSAMPLE_SIZES[split], SEED)
    if variant == "scrubbed":
        df = df.assign(text=df["text"].map(scrub))
    return df
