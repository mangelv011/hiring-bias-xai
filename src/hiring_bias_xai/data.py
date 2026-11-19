"""Loading, splitting and scrubbing Bias in Bios."""

import re

import pandas as pd

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


def load_split(split: str, variant: str = "original") -> pd.DataFrame:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {', '.join(_SPLITS)}; got {split!r}")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(_VARIANTS)}; got {variant!r}")
    raw = _load_raw(split)
    return pd.DataFrame(
        {
            "id": [f"{split}-{i}" for i in range(len(raw))],
            "text": raw["hard_text"],
            "occupation": [OCCUPATIONS[code] for code in raw["profession"]],
            "gender": ["female" if code == 1 else "male" for code in raw["gender"]],
        }
    )
