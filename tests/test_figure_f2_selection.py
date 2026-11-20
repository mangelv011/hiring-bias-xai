import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f2():
    return _load_script()


@pytest.mark.parametrize("word", ["She", "her,", "he", "his.", "She's", "Ms.", "HIM", "(herself)"])
def test_gendered_words_are_those_the_scrubber_removes(f2, word):
    assert f2.is_gendered(word)


@pytest.mark.parametrize("word", ["the", "shell", "heritage", "there", "nurse", "Smith,", ""])
def test_other_words_are_not_gendered(f2, word):
    assert not f2.is_gendered(word)


def _case(f2, bio_id, true, predicted, words, values):
    return f2.Case(bio_id, words.split(), true, predicted, np.array(values, dtype=float))


def test_the_error_case_is_the_misclassified_bio_with_the_largest_gendered_push_to_the_prediction(
    f2,
):
    cases = [
        _case(f2, "test-1", "nurse", "surgeon", "she is a surgeon", [1.0, 0.0, 0.0, 5.0]),  # 1.0
        _case(f2, "test-2", "nurse", "surgeon", "he is a surgeon", [3.0, 0.0, 0.0, 9.0]),  # 3.0
        _case(f2, "test-3", "nurse", "surgeon", "she and her team", [1.0, 0.0, 2.5, 0.0]),  # 3.5
        _case(f2, "test-4", "nurse", "nurse", "she and her team", [9.0, 0.0, 9.0, 0.0]),  # correct
        _case(f2, "test-5", "nurse", "surgeon", "she and her team", [-9.0, 0.0, -9.0, 0.0]),
    ]
    assert f2.pick_error(cases).bio_id == "test-3"


def test_sums_over_gendered_words_count_each_occurrence_with_its_own_value(f2):
    cases = [
        _case(f2, "test-1", "a", "b", "she her she", [1.0, 1.0, 1.0]),  # 3.0
        _case(f2, "test-2", "a", "b", "she x x", [2.9, 9.0, 9.0]),  # 2.9
    ]
    assert f2.pick_error(cases).bio_id == "test-1"


def test_a_tie_goes_to_the_bio_that_comes_first(f2):
    cases = [
        _case(f2, "test-7", "a", "b", "she x", [2.0, 0.0]),
        _case(f2, "test-3", "a", "b", "she y", [2.0, 0.0]),
    ]
    assert f2.pick_error(cases).bio_id == "test-7"
    assert f2.pick_error(cases[::-1]).bio_id == "test-3"


def test_no_error_with_gendered_words_pushing_towards_the_prediction_is_an_error_of_the_rule(f2):
    cases = [
        _case(f2, "test-1", "a", "b", "she x", [-1.0, 5.0]),
        _case(f2, "test-2", "a", "b", "x y", [3.0, 5.0]),
        _case(f2, "test-3", "a", "a", "she x", [3.0, 5.0]),
    ]
    with pytest.raises(ValueError, match="gendered"):
        f2.pick_error(cases)


def test_the_correct_case_is_the_right_call_with_the_largest_share_of_gendered_values(f2):
    # Shares of the sum of |value| carried by gendered words: 2/4, 3/5, 1/5, and a miss.
    cases = [
        _case(f2, "test-1", "a", "a", "she x y", [2.0, -1.0, -1.0]),
        _case(f2, "test-2", "a", "a", "her x y", [-3.0, 1.0, 1.0]),
        _case(f2, "test-3", "a", "a", "his x y", [1.0, 2.0, 2.0]),
        _case(f2, "test-4", "a", "b", "she x y", [9.0, 0.0, 0.0]),
        _case(f2, "test-5", "a", "a", "x y z", [1.0, 1.0, 1.0]),
    ]
    assert f2.pick_correct(cases).bio_id == "test-2"


def test_a_bio_whose_values_are_all_zero_has_no_share(f2):
    cases = [
        _case(f2, "test-1", "a", "a", "she x", [0.0, 0.0]),
        _case(f2, "test-2", "a", "a", "she x", [1.0, 3.0]),
    ]
    assert f2.pick_correct(cases).bio_id == "test-2"


def test_a_bio_with_a_long_number_is_never_picked_because_it_may_hold_an_identifier(f2):
    cases = [
        _case(f2, "test-1", "a", "b", "she has licence 1285904128 and", [9.0, 0.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-2", "a", "b", "she was published in 2013", [1.0, 0.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-3", "a", "a", "she has licence 1285904128 and", [9.0, 0.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-4", "a", "a", "she is here", [1.0, 1.0, 1.0]),
    ]
    assert f2.pick_error(cases).bio_id == "test-2"
    assert f2.pick_correct(cases).bio_id == "test-4"


def _saved_bios(tmp_path, monkeypatch, texts, occupations):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    model = Hashed()
    ids = [f"test-{i}" for i in range(len(texts))]
    subset = pd.DataFrame({"id": ids, "text": texts, "occupation": occupations})
    for bio_id, text, occupation in zip(ids, texts, occupations, strict=True):
        (result,) = attribution.shapley(model, [text], [occupation], 2, "delete", seed=1)
        attribution.save(attribution.path("deberta", bio_id), result)
    predicted = model.predict(list(texts))
    return model, subset, predicted


def test_cases_carry_the_values_for_the_predicted_occupation_from_the_stored_logits(
    f2, tmp_path, monkeypatch
):
    texts = ["she is a surgeon", "he was a nurse here"]
    model, subset, predicted = _saved_bios(tmp_path, monkeypatch, texts, ["nurse", "surgeon"])
    cases, skipped = f2.load_cases(subset, predicted)
    assert skipped == 0
    assert [case.bio_id for case in cases] == ["test-0", "test-1"]
    for case, text, label, true in zip(cases, texts, predicted, ["nurse", "surgeon"], strict=True):
        assert (case.words, case.true, case.predicted) == (text.split(), true, label)
        stored = attribution.load(attribution.path("deberta", case.bio_id))
        np.testing.assert_allclose(case.values, stored.values_for(data.OCCUPATIONS.index(label)))


def test_a_bio_whose_stored_full_text_gives_another_prediction_than_the_saved_one_is_skipped(
    f2, tmp_path, monkeypatch
):
    model, subset, predicted = _saved_bios(
        tmp_path, monkeypatch, ["she is a surgeon", "he was a nurse here"], ["nurse", "surgeon"]
    )
    other = next(label for label in data.OCCUPATIONS if label != predicted[0])
    cases, skipped = f2.load_cases(subset, [other, predicted[1]])
    assert skipped == 1
    assert [case.bio_id for case in cases] == ["test-1"]


def test_a_number_of_six_digits_marks_an_identifier_and_one_of_five_does_not(f2):
    cases = [
        _case(f2, "test-1", "a", "b", "she has 123456 and", [9.0, 0.0, 0.0, 0.0]),
        _case(f2, "test-2", "a", "b", "she has 12345 and", [1.0, 0.0, 0.0, 0.0]),
    ]
    assert f2.pick_error(cases).bio_id == "test-2"


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
