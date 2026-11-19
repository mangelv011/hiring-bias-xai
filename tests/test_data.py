import pytest

from hiring_bias_xai import data


def test_occupations_are_the_28_names_in_code_order():
    assert data.OCCUPATIONS == (
        "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
        "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
        "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
        "photographer", "physician", "poet", "professor", "psychologist", "rapper",
        "software_engineer", "surgeon", "teacher", "yoga_teacher",
    )  # fmt: skip


GENDERED_WORDS = [
    "he", "she", "her", "his", "him", "hers", "himself", "herself", "mr", "mrs", "ms",
]  # fmt: skip


@pytest.mark.parametrize("word", GENDERED_WORDS)
@pytest.mark.parametrize("style", [str.lower, str.capitalize, str.upper])
def test_scrub_removes_each_gendered_word_in_any_case(word, style):
    assert data.scrub(f"a {style(word)} b") == "a b"


def test_scrub_strips_the_ends():
    assert data.scrub("She is a nurse. ") == "is a nurse."



@pytest.mark.parametrize(
    "word",
    ["the", "shell", "heritage", "hierarchy", "there", "sheriff", "history", "ashes", "this"],
)
def test_scrub_keeps_words_that_contain_a_pronoun(word):
    text = f"the {word} was here"
    assert data.scrub(text) == text


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("she's a nurse", "a nurse"),
        ("He'd say so", "say so"),
        ("she'll go", "go"),
        ("his' book", "book"),
        ("the nurse and her's", "the nurse and"),
    ],
)
def test_scrub_removes_attached_pronoun_suffixes(text, expected, apostrophe):
    assert data.scrub(text.replace("'", apostrophe)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr. Smith is a dentist", "Smith is a dentist"),
        ("Mrs. Lee, a nurse", "Lee, a nurse"),
        ("ms. Ray", "Ray"),
        ("Dr. Mr. X", "Dr. X"),
    ],
)
def test_scrub_removes_the_period_after_a_title(text, expected):
    assert data.scrub(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Worked hard (she said) daily.", "Worked hard ( said) daily."),
        ("Nurse, she.", "Nurse, ."),
        ("Ask him, her, or his-own", "Ask , , or -own"),
        ("a  b   she   c", "a b c"),
    ],
)
def test_scrub_handles_punctuation_next_to_the_word_and_collapses_spaces(text, expected):
    assert data.scrub(text) == expected
