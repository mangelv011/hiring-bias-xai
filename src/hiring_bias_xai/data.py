"""Loading, splitting and scrubbing Bias in Bios."""

import re

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
