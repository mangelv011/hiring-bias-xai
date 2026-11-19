import inspect

import numpy as np
import pytest
import torch

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


def test_predict_proba_is_deterministic_even_if_the_network_was_left_in_training_mode(
    tiny_network, tiny_tokenizer
):
    model = _model(tiny_network, tiny_tokenizer)
    model.network.train()
    np.testing.assert_array_equal(model.predict_proba(TEXTS), model.predict_proba(TEXTS))


def test_batches_keep_input_order_and_padding_does_not_change_a_text_probabilities(
    tiny_network, tiny_tokenizer
):
    model = _model(tiny_network, tiny_tokenizer, batch_size=3)
    alone = np.stack([model.predict_proba([text])[0] for text in TEXTS])
    np.testing.assert_allclose(model.predict_proba(TEXTS), alone, atol=1e-5)


def test_texts_are_cut_to_the_tokenizer_max_length_including_the_two_special_tokens(
    tiny_network, tiny_tokenizer
):
    # max_length 5 = [CLS], three words, [SEP].
    model = _model(tiny_network, tiny_tokenizer, max_length=5)
    cut = model.predict_proba(["alpha beta gamma delta epsilon zeta"])
    np.testing.assert_allclose(cut, model.predict_proba(["alpha beta gamma"]), atol=1e-6)


def test_representations_hold_the_cls_state_of_every_layer_and_the_final_vector(
    tiny_network, tiny_tokenizer
):
    # The tiny network has 2 transformer layers and hidden size 32.
    layers, final = _model(tiny_network, tiny_tokenizer).representations(TEXTS)
    assert layers.shape == (4, 3, 32)
    assert final.shape == (4, 32)


def test_the_classifier_reads_the_final_vector(tiny_network, tiny_tokenizer):
    model = _model(tiny_network, tiny_tokenizer)
    _, final = model.representations(TEXTS)
    with torch.no_grad():
        logits = model.network.classifier(torch.from_numpy(final))
    np.testing.assert_allclose(
        torch.softmax(logits, dim=-1).numpy(), model.predict_proba(TEXTS), atol=1e-5
    )


def test_the_final_vector_is_the_pooled_cls_state_of_the_last_layer(tiny_network, tiny_tokenizer):
    model = _model(tiny_network, tiny_tokenizer)
    layers, final = model.representations(TEXTS)
    model.network.eval()
    with torch.no_grad():
        pooled = model.network.pooler(torch.from_numpy(layers[:, -1])[:, None, :])
    np.testing.assert_allclose(pooled.numpy(), final, atol=1e-5)


def test_a_saved_model_loads_back_from_the_artifacts_directory(
    monkeypatch, tmp_path, tiny_network, tiny_tokenizer
):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    model = _model(tiny_network, tiny_tokenizer, max_length=5)
    model.save()
    folder = tmp_path / "models" / "deberta"
    assert (folder / "config.json").is_file()
    loaded = DebertaModel.load()
    texts = ["alpha beta gamma delta epsilon", "zeta eta"]
    np.testing.assert_array_equal(loaded.predict_proba(texts), model.predict_proba(texts))


def test_load_takes_no_path_so_only_the_project_folder_in_the_artifacts_directory_is_read():
    assert list(inspect.signature(DebertaModel.load).parameters) == ["batch_size"]


def test_the_checkpoint_is_safetensors_and_pickle_weights_are_refused(
    monkeypatch, tmp_path, tiny_network, tiny_tokenizer
):
    # Pickle files can run code when loaded: the checkpoint has no pickle weights and a folder
    # that holds only pickle weights does not load.
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    model = _model(tiny_network, tiny_tokenizer)
    model.save()
    folder = tmp_path / "models" / "deberta"
    assert (folder / "model.safetensors").is_file()
    assert not list(folder.glob("*.bin")) + list(folder.glob("*.pt"))
    (folder / "model.safetensors").replace(tmp_path / "moved.safetensors")
    torch.save(model.network.state_dict(), folder / "pytorch_model.bin")
    with pytest.raises(OSError):
        DebertaModel.load()
