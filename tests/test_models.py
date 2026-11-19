import inspect
import random
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from hiring_bias_xai import data
from hiring_bias_xai.models import BaselineModel

# Occupation names and their columns in data.OCCUPATIONS: nurse 13, surgeon 25.
NURSE, SURGEON = "nurse", "surgeon"
TEXTS = [
    "ward patient care nurse",
    "patient care medication ward",
    "nurse care medication",
    "operating theatre scalpel surgeon",
    "scalpel surgery operating",
    "surgeon surgery theatre",
]
OCCUPATIONS = [NURSE] * 3 + [SURGEON] * 3
WORD_POOL = [f"word{i}" for i in range(20)]


def _fitted(**kwargs):
    return BaselineModel(**{"C": 10.0, "min_df": 1} | kwargs).fit(TEXTS, OCCUPATIONS)


def test_labels_are_the_28_occupations_in_dataset_order():
    assert list(_fitted().labels) == list(data.OCCUPATIONS)


def test_predict_proba_gives_one_probability_row_per_text_over_all_28_occupations():
    proba = _fitted().predict_proba(["a patient", "a scalpel", "something else"])
    assert proba.shape == (3, 28)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_probability_lands_in_the_column_of_the_learned_occupation_and_none_elsewhere():
    proba = _fitted().predict_proba(["the patient ward", "operating scalpel"])
    assert proba.argmax(axis=1).tolist() == [13, 25]
    assert proba[0, 13] > 0.5
    assert proba[1, 25] > 0.5
    others = np.delete(proba, [13, 25], axis=1)
    assert not others.any()


def test_word_pairs_are_features():
    # Both occupations use the words "alpha" and "beta" equally often; only their order differs,
    # so a model without two-word features cannot tell them apart and stays at 1/2.
    texts = ["alpha beta"] * 3 + ["beta alpha"] * 3
    model = BaselineModel(C=10.0, min_df=1).fit(texts, OCCUPATIONS)
    proba = model.predict_proba(["alpha beta", "beta alpha"])
    assert proba[0, 13] > 0.7
    assert proba[1, 25] > 0.7


def test_min_df_drops_words_found_in_fewer_bios():
    # "quokka" is in one surgeon bio only. With min_df=1 it is a feature and points to surgeon;
    # with min_df=2 it is unknown, the text is empty and the two classes (3 bios each) stay
    # near 1/2.
    texts = [TEXTS[0], TEXTS[1], TEXTS[2], TEXTS[3] + " quokka", TEXTS[4], TEXTS[5]]

    def surgeon_probability(min_df):
        model = BaselineModel(C=10.0, min_df=min_df).fit(texts, OCCUPATIONS)
        return model.predict_proba(["quokka"])[0, 25]

    assert surgeon_probability(1) > 0.65
    assert surgeon_probability(2) == pytest.approx(0.5, abs=0.1)


def test_a_smaller_C_regularises_more_so_predictions_are_less_confident():
    def confidence(C):
        return _fitted(C=C).predict_proba(["operating scalpel"])[0, 25]

    assert confidence(0.01) < confidence(100.0)


def test_a_saved_model_loads_back_from_the_artifacts_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    model = _fitted(C=3.0, min_df=1)
    model.save()
    assert (tmp_path / "models" / "baseline.joblib").is_file()
    loaded = BaselineModel.load()
    assert (loaded.C, loaded.min_df) == (3.0, 1)
    texts = ["the patient ward", "operating scalpel"]
    np.testing.assert_array_equal(loaded.predict_proba(texts), model.predict_proba(texts))


def test_load_takes_no_path_so_only_the_project_file_in_the_artifacts_directory_is_read():
    # joblib uses pickle, which can run code. `load` has no path argument: callers cannot point
    # it at a file from another source.
    assert not inspect.signature(BaselineModel.load).parameters


def test_fit_runs_until_the_solver_converges():
    # 3,000 word salads from 20 words with random labels cannot be separated, so lbfgs needs
    # more than 100 iterations at this C. A fit that stops early is an under-trained model.
    rng = random.Random(0)
    texts = [" ".join(rng.choices(WORD_POOL, k=10)) for _ in range(3000)]
    labels = rng.choices(data.OCCUPATIONS, k=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        BaselineModel(C=1e3, min_df=1).fit(texts, labels)


def test_fit_works_in_single_precision():
    # The solver is memory-bound: float32 features and weights run 1.6 times faster with six
    # fits at once on the 60,000-bio training subsample.
    assert _fitted().pipeline[-1].coef_.dtype == np.float32


def test_logits_are_float64_decision_scores_whose_softmax_is_predict_proba():
    model = _fitted()
    texts = ["the patient ward", "operating scalpel", "something else"]
    logits = model.logits(texts)
    assert logits.shape == (3, 28)
    assert logits.dtype == np.float64
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = shifted / shifted.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(expected, model.predict_proba(texts))


def test_logits_of_occupations_the_model_never_saw_are_minus_infinity():
    logits = _fitted().logits(["the patient ward"])
    seen = [13, 25]
    assert np.isfinite(logits[0, seen]).all()
    assert np.isneginf(np.delete(logits, seen, axis=1)).all()


def _three_occupations():
    texts = TEXTS + ["dental drill tooth", "drill tooth dentist", "dentist tooth"]
    return BaselineModel(C=10.0, min_df=1).fit(texts, OCCUPATIONS + ["dentist"] * 3)


def test_logits_follow_dataset_order_when_the_model_saw_three_occupations():
    model = _three_occupations()
    assert model.logits(["dental drill tooth"])[0].argmax() == data.OCCUPATIONS.index("dentist")
    assert model.logits(["operating scalpel"])[0].argmax() == 25


def test_predict_returns_the_occupation_name_with_the_highest_probability():
    assert list(_fitted().predict(["the patient ward", "operating scalpel"])) == [NURSE, SURGEON]


def test_class_log_odds_is_the_class_logit_minus_the_log_sum_of_exponentials_of_the_others():
    model = _fitted()
    texts = ["the patient ward", "operating scalpel"]
    z = model.logits(texts)
    expected = z[:, 13] - z[:, 25]  # the 26 unseen columns are -inf and add nothing
    np.testing.assert_allclose(model.class_log_odds(texts, NURSE), expected)
    np.testing.assert_allclose(model.class_log_odds(texts, SURGEON), -expected)


def test_class_log_odds_is_the_log_of_the_odds_of_the_probabilities():
    # With a third occupation the others sum over two columns.
    model = _three_occupations()
    proba = model.predict_proba(["patient scalpel drill"])[0, 13]
    np.testing.assert_allclose(
        model.class_log_odds(["patient scalpel drill"], NURSE), np.log(proba / (1 - proba))
    )
