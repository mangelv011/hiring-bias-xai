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

# "her" and "his" have two readings each, so a part-of-speech tagger decides.
_AMBIGUOUS = ("her", "his")

# Parts of speech that cannot continue a noun phrase: a "his" before one is a pronoun.
_PHRASE_BREAKS = {"PUNCT", "CCONJ", "SCONJ", "ADP", "AUX", "PART"}

_GENDERED = re.compile(
    r"\b((?i:" + "|".join((*_SWAPPED, *_AMBIGUOUS)) + r")|" + "|".join(_TITLES) + r")(?!\w)"
)


@functools.cache
def _tagger():
    """spaCy's small English pipeline with only the tagger and its tag-to-POS mapping."""
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def _stands_alone(token) -> bool:
    """Whether no noun phrase continues after the token: it ends the text or a break follows."""
    following = token.doc[token.i + 1 :]
    return len(following) == 0 or following[0].pos_ in _PHRASE_BREAKS


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
    reads it. "his" becomes "her" before a noun and "hers" when it stands alone: at the end of
    the text, or before a word that does not continue a noun phrase.
    """
    tokens = {}
    if any(match.group(1).lower() in _AMBIGUOUS for match in _GENDERED.finditer(text)):
        tokens = {token.idx: token for token in _tagger()(text)}

    def replacement(match: re.Match) -> str:
        word = match.group(1)
        if word in _TITLES:
            return _TITLES[word]
        if word.lower() == "her":
            return _with_case_of(word, "his" if tokens[match.start()].tag_ == "PRP$" else "him")
        if word.lower() == "his":
            return _with_case_of(word, "hers" if _stands_alone(tokens[match.start()]) else "her")
        return _with_case_of(word, _SWAPPED[word.lower()])

    return _GENDERED.sub(replacement, text)


# Residual cues: words that still tell the gender of the person after `swap_gender`. The list is
# fixed in code before any swapped bio is read, and `swap_gender` does not touch these words.
RESIDUAL_NOUNS = (
    # female
    "mother", "mothers", "mom", "moms", "mum", "mama", "wife", "wives", "actress", "actresses",
    "businesswoman", "businesswomen", "daughter", "daughters", "sister", "sisters", "girl",
    "girls", "girlfriend", "woman", "women", "lady", "ladies", "queen", "queens", "chairwoman",
    "chairwomen", "spokeswoman", "congresswoman", "policewoman", "saleswoman", "waitress",
    "stewardess", "hostess", "heroine", "grandmother", "grandma", "aunt", "niece", "bride",
    "widow", "madam", "madame", "princess", "female", "females", "matriarch", "gal",
    # male
    "father", "fathers", "dad", "dads", "husband", "husbands", "actor", "actors", "businessman",
    "businessmen", "son", "sons", "brother", "brothers", "boy", "boys", "boyfriend", "man", "men",
    "gentleman", "gentlemen", "king", "kings", "chairman", "chairmen", "spokesman", "congressman",
    "policeman", "salesman", "waiter", "steward", "grandfather", "grandpa", "uncle", "nephew",
    "groom", "widower", "sir", "prince", "male", "males", "patriarch", "guy", "guys",
)  # fmt: skip

_RESIDUAL_NOUN = re.compile(r"\b(?:" + "|".join(RESIDUAL_NOUNS) + r")\b", re.IGNORECASE)


def residual_cues(text: str) -> list[str]:
    """The residual cues in the text, lower-cased, each once, in alphabetical order."""
    return sorted({match.group().lower() for match in _RESIDUAL_NOUN.finditer(text)})
