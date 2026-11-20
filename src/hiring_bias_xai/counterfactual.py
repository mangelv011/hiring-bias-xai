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

# Only the Capitalised titles: "MS" is a degree and "ms" a unit.
_TITLES = {"Mr": "Ms", "Ms": "Mr", "Mrs": "Mr"}

_GENDERED = re.compile(r"\b((?i:" + "|".join(_SWAPPED) + r")|" + "|".join(_TITLES) + r")(?!\w)")


def _with_case_of(model: str, word: str) -> str:
    """`word` in lower case, Capitalised or ALL CAPS, as `model` is; anything else is lower case."""
    if len(model) > 1 and model.isupper():
        return word.upper()
    if model[0].isupper() and model[1:].islower():
        return word.capitalize()
    return word


def _replacement(match: re.Match) -> str:
    word = match.group(1)
    if word in _TITLES:
        return _TITLES[word]
    return _with_case_of(word, _SWAPPED[word.lower()])


def swap_gender(text: str) -> str:
    """The bio with its gendered pronouns and titles swapped, he for she and the reverse."""
    return _GENDERED.sub(_replacement, text)
