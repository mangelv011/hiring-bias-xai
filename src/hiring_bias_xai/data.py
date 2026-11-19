"""Loading, splitting and scrubbing Bias in Bios."""

import re

OCCUPATIONS = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
)  # fmt: skip

_GENDERED = re.compile(
    r"\b(?:(?:he|she|her|his|him|hers|himself|herself)(?:['’](?:s|d|ll)?)?|mr|mrs|ms)(?!\w)",
    re.IGNORECASE,
)


def scrub(text: str) -> str:
    return re.sub(" {2,}", " ", _GENDERED.sub("", text)).strip()
