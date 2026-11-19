import numpy as np
import pytest

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
