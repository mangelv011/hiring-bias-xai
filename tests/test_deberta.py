import numpy as np

from hiring_bias_xai import data
from hiring_bias_xai.models import DebertaModel

TEXTS = ["alpha beta", "gamma delta epsilon", "zeta", "eta theta alpha beta gamma"]


def _model(tiny_network, tiny_tokenizer, **kwargs):
    return DebertaModel(tiny_network(), tiny_tokenizer(kwargs.pop("max_length", 64)), **kwargs)


def test_labels_are_the_28_occupations_in_dataset_order(tiny_network, tiny_tokenizer):
    assert list(_model(tiny_network, tiny_tokenizer).labels) == list(data.OCCUPATIONS)


def test_predict_proba_gives_one_probability_row_per_text_over_all_28_occupations(
    tiny_network, tiny_tokenizer
):
    proba = _model(tiny_network, tiny_tokenizer).predict_proba(TEXTS)
    assert proba.shape == (4, 28)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_column_i_is_the_probability_of_occupation_i(tiny_network, tiny_tokenizer):
    # Occupation names and their columns in data.OCCUPATIONS: nurse 13, surgeon 25. A large
    # bias on one output unit makes that column the argmax whatever the text.
    for name in ("nurse", "surgeon"):
        model = _model(tiny_network, tiny_tokenizer)
        model.network.classifier.bias.data[data.OCCUPATIONS.index(name)] = 100.0
        picked = model.predict_proba(TEXTS).argmax(axis=1)
        assert [model.labels[i] for i in picked] == [name] * len(TEXTS)
