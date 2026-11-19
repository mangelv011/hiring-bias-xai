import pytest

from hiring_bias_xai import data


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("She is a nurse and he is a doctor.", "is a nurse and is a doctor."),
        ("He and SHE and Him and hErself", "and and and"),
        ("Her bio and his book. Hers is long, himself too.", "bio and book. is long, too."),
        (
            "The shell of the heritage; hierarchy there. Sheriff history.",
            "The shell of the heritage; hierarchy there. Sheriff history.",
        ),
        ("This is her's and his' and she's here", "This is and and here"),
        ('They said (she) left, "he" stayed.', 'They said () left, "" stayed.'),
        ("Mr. Smith, Mrs Jones and Ms. Lee", "Smith, Jones and Lee"),
        ("Her’s and he’d", "and"),
        ("Ask him. Then she left.", "Ask . Then left."),
        ("line one\nshe line two", "line one\nline two"),
    ],
)
def test_scrub_removes_gendered_words_only(text, expected):
    assert data.scrub(text) == expected
