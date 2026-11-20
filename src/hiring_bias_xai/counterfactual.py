"""Counterfactual swaps of pronouns and name headers."""

import functools
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

# "her" has two readings, possessive and object, so a part-of-speech tagger decides.
_AMBIGUOUS = ("her",)

_GENDERED = re.compile(
    r"\b((?i:" + "|".join((*_SWAPPED, *_AMBIGUOUS)) + r")|" + "|".join(_TITLES) + r")(?!\w)"
)


@functools.cache
def _tagger():
    """spaCy's small English pipeline with only the tagger and its tag-to-POS mapping."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def _with_case_of(model: str, word: str) -> str:
    """`word` in lower case, Capitalised or ALL CAPS, as `model` is; anything else is lower case."""
    if len(model) > 1 and model.isupper():
        return word.upper()
    if model[0].isupper() and model[1:].islower():
        return word.capitalize()
    return word


def swap_gender(text: str) -> str:
    """The bio with its gendered pronouns and titles swapped, he for she and the reverse.

    "her" becomes "his" when it is a possessive and "him" when it is an object, as the tagger
    reads it.
    """
    tags = {}
    if any(match.group(1).lower() in _AMBIGUOUS for match in _GENDERED.finditer(text)):
        tags = {token.idx: token.tag_ for token in _tagger()(text)}

    def replacement(match: re.Match) -> str:
        word = match.group(1)
        if word in _TITLES:
            return _TITLES[word]
        if word.lower() == "her":
            return _with_case_of(word, "his" if tags[match.start()] == "PRP$" else "him")
        return _with_case_of(word, _SWAPPED[word.lower()])

    return _GENDERED.sub(replacement, text)
