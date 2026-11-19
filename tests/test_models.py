import numpy as np

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
