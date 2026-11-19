
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
