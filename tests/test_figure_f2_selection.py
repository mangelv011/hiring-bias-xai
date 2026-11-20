import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f2.py"
MALE_DOMINATED = ["surgeon", "composer"]
FILLER = [f"word{i}" for i in range(20)]


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f2():
    return _load_script()


def _case(f2, bio_id, true, predicted, text, values, gender="female", pad=True):
    """A case whose words are `text` followed by filler words with value 0, so it has 15 units."""
    words = text.split()
    values = list(values)
    if pad:
        extra = max(0, 15 - len(words))
        words += FILLER[:extra]
        values += [0.0] * extra
    return f2.Case(bio_id, words, true, predicted, gender, np.array(values, dtype=float))


def test_the_push_adds_the_values_of_the_gendered_words_and_counts_each_occurrence(f2):
    case = _case(
        f2, "test-1", "surgeon", "nurse", "She and her team she", [1.0, 5.0, 2.0, 7.0, 0.5]
    )
    assert f2.gendered_push(case) == pytest.approx(3.5)


def test_a_unit_that_only_contains_a_gendered_word_does_not_count(f2):
    case = _case(
        f2, "test-1", "surgeon", "nurse", "shoot...she sheriff She, x", [4.0, 3.0, 1.0, 9.0]
    )
    assert f2.gendered_push(case) == pytest.approx(1.0)


def test_a_negative_push_is_kept_with_its_sign(f2):
    case = _case(f2, "test-1", "surgeon", "nurse", "she x", [-2.0, 3.0])
    assert f2.gendered_push(case) == pytest.approx(-2.0)


def _pick(f2, cases, gender="female", occupations=MALE_DOMINATED):
    return f2.pick(cases, gender, occupations)


def test_the_pick_is_the_misclassified_bio_of_that_gender_and_group_with_the_largest_push(f2):
    cases = [
        _case(f2, "test-1", "surgeon", "nurse", "she x", [1.0, 5.0]),  # 1.0
        _case(f2, "test-2", "composer", "nurse", "her x", [3.0, 9.0]),  # 3.0
        _case(f2, "test-3", "surgeon", "dentist", "she and her", [1.0, 0.0, 2.5]),  # 3.5
        _case(f2, "test-4", "surgeon", "surgeon", "she and her", [9.0, 0.0, 9.0]),  # right call
        _case(f2, "test-5", "surgeon", "nurse", "he x", [9.0, 9.0], gender="male"),  # a man
        _case(f2, "test-6", "nurse", "dentist", "she x", [9.0, 9.0]),  # not in the group
        _case(f2, "test-7", "surgeon", "nurse", "she x", [-9.0, 9.0]),  # pushes away
    ]
    assert _pick(f2, cases).bio_id == "test-3"
    assert _pick(f2, cases, gender="male", occupations=["nurse"]) is None


def test_the_men_in_the_other_group_are_picked_by_the_same_score(f2):
    cases = [
        _case(f2, "test-1", "nurse", "surgeon", "he x", [1.0, 5.0], gender="male"),
        _case(f2, "test-2", "nurse", "surgeon", "his and he", [1.0, 0.0, 2.0], gender="male"),
        _case(f2, "test-3", "surgeon", "nurse", "he x", [9.0, 5.0], gender="male"),
    ]
    assert _pick(f2, cases, "male", ["nurse", "model"]).bio_id == "test-2"


def test_a_tie_goes_to_the_bio_that_comes_first(f2):
    cases = [
        _case(f2, "test-7", "surgeon", "nurse", "she x", [2.0, 0.0]),
        _case(f2, "test-3", "surgeon", "nurse", "she y", [2.0, 0.0]),
    ]
    assert _pick(f2, cases).bio_id == "test-7"
    assert _pick(f2, cases[::-1]).bio_id == "test-3"


def test_with_no_candidate_pushing_towards_the_prediction_there_is_no_pick(f2):
    cases = [
        _case(f2, "test-1", "surgeon", "nurse", "she x", [-1.0, 5.0]),
        _case(f2, "test-2", "surgeon", "nurse", "x y", [3.0, 5.0]),  # no gendered word
        _case(f2, "test-3", "surgeon", "nurse", "she x", [0.0, 5.0]),  # a push of zero
        _case(f2, "test-4", "surgeon", "surgeon", "she x", [3.0, 5.0]),
    ]
    assert _pick(f2, cases) is None
    assert _pick(f2, []) is None


def _with_units(f2, bio_id, n_units, push):
    words = ["she"] + [f"w{i}" for i in range(n_units - 1)]
    return f2.Case(
        bio_id, words, "surgeon", "nurse", "female", np.array([push] + [0.0] * (n_units - 1))
    )


def test_a_bio_with_fewer_than_15_units_is_never_picked(f2):
    assert _pick(f2, [_with_units(f2, "test-1", 14, 9.0)]) is None
    picked = _pick(f2, [_with_units(f2, "test-1", 14, 9.0), _with_units(f2, "test-2", 15, 1.0)])
    assert picked.bio_id == "test-2"


def test_a_bio_with_six_or_more_digits_in_a_row_is_never_picked_as_it_may_hold_an_identifier(
    f2,
):
    cases = [
        _case(f2, "test-1", "surgeon", "nurse", "she has 123456 and", [9.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-2", "surgeon", "nurse", "she has 1285904128 and", [9.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-3", "surgeon", "nurse", "she has 12345 and", [1.0, 0.0, 0.0, 0.0]),
    ]
    assert _pick(f2, cases).bio_id == "test-3"


def _saved_bios(tmp_path, monkeypatch, texts, occupations, genders):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    model = Hashed()
    ids = [f"test-{i}" for i in range(len(texts))]
    subset = pd.DataFrame({"id": ids, "text": texts, "occupation": occupations, "gender": genders})
    for bio_id, text, occupation in zip(ids, texts, occupations, strict=True):
        (result,) = attribution.shapley(model, [text], [occupation], 2, "delete", seed=1)
        attribution.save(attribution.path("deberta", bio_id), result)
    return model, subset, model.predict(list(texts))


def test_cases_carry_gender_and_the_values_for_the_predicted_occupation_from_stored_logits(
    f2, tmp_path, monkeypatch
):
    texts = ["she is a surgeon", "he was a nurse here"]
    _, subset, predicted = _saved_bios(
        tmp_path, monkeypatch, texts, ["nurse", "surgeon"], ["female", "male"]
    )
    cases, skipped = f2.load_cases(subset, predicted)
    assert skipped == 0
    assert [case.bio_id for case in cases] == ["test-0", "test-1"]
    assert [case.gender for case in cases] == ["female", "male"]
    for case, text, label, true in zip(cases, texts, predicted, ["nurse", "surgeon"], strict=True):
        assert (case.words, case.true, case.predicted) == (text.split(), true, label)
        stored = attribution.load(attribution.path("deberta", case.bio_id))
        np.testing.assert_allclose(case.values, stored.values_for(data.OCCUPATIONS.index(label)))


def test_a_bio_whose_stored_full_text_gives_another_prediction_than_the_saved_one_is_skipped(
    f2, tmp_path, monkeypatch
):
    _, subset, predicted = _saved_bios(
        tmp_path,
        monkeypatch,
        ["she is a surgeon", "he was a nurse here"],
        ["nurse", "surgeon"],
        ["female", "male"],
    )
    other = next(label for label in data.OCCUPATIONS if label != predicted[0])
    cases, skipped = f2.load_cases(subset, [other, predicted[1]])
    assert skipped == 1
    assert [case.bio_id for case in cases] == ["test-1"]


def test_the_sentence_wraps_when_the_next_word_would_pass_the_line_and_not_before(f2):
    # Lines hold 5 characters. "ab" and "cd" fill a line exactly (columns 0 and 3); "ef" starts a
    # new one; a word longer than a line is cut into pieces.
    assert f2._layout(["ab", "cd", "ef"], 5) == [(0, 0, 0, "ab"), (0, 3, 1, "cd"), (1, 0, 2, "ef")]
    assert f2._layout(["abcdefghijkl", "x"], 5) == [
        (0, 0, 0, "abcde"),
        (1, 0, 0, "fghij"),
        (2, 0, 0, "kl"),
        (2, 3, 1, "x"),
    ]


def test_digits_in_different_units_do_not_make_a_run(f2):
    # "123" and "456" next to each other are two numbers, not a run of six digits.
    case = f2.Case(
        "test-1",
        ["she", "123", "456"] + [f"w{i}" for i in range(12)],
        "surgeon",
        "nurse",
        "female",
        np.array([2.0] + [0.0] * 14),
    )
    assert _pick(f2, [case]) is case
