"""Counterfactual swaps of pronouns and name headers."""

import re

_SWAPPED = {
    "he": "she",
    "she": "he",
    "him": "her",
    "himself": "herself",
    "herself": "himself",
    "hers": "his",
}

_PRONOUN = re.compile(r"\b(" + "|".join(_SWAPPED) + r")(?!\w)", re.IGNORECASE)


def swap_gender(text: str) -> str:
    """The bio with its gendered pronouns swapped, he for she and the reverse."""
    return _PRONOUN.sub(lambda match: _SWAPPED[match.group(1).lower()], text)
