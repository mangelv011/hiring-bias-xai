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


def _with_case_of(model: str, word: str) -> str:
    """`word` in lower case, Capitalised or ALL CAPS, as `model` is; anything else is lower case."""
    if len(model) > 1 and model.isupper():
        return word.upper()
    if model[0].isupper() and model[1:].islower():
        return word.capitalize()
    return word


def swap_gender(text: str) -> str:
    """The bio with its gendered pronouns swapped, he for she and the reverse."""
    return _PRONOUN.sub(
        lambda match: _with_case_of(match.group(1), _SWAPPED[match.group(1).lower()]), text
    )
