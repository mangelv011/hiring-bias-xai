import inspect
import re

import numpy as np
import pytest
import torch

from hiring_bias_xai import data, models
from hiring_bias_xai.models import DebertaModel

TEXTS = ["alpha beta", "gamma delta epsilon", "zeta", "eta theta alpha beta gamma"]


def _model(tiny_network, tiny_tokenizer, **kwargs):
    return DebertaModel(
        tiny_network(spread=3.0), tiny_tokenizer(kwargs.pop("max_length", 64)), **kwargs
    )


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
    loaded = DebertaModel.load(batch_size=7)
    assert loaded.batch_size == 7
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


def test_pretrained_downloads_the_pinned_revision_into_the_artifacts_directory_with_28_labels(
    monkeypatch, tmp_path, tiny_network, tiny_tokenizer
):
    # The download is the trust boundary: weights and tokenizer come from one pinned commit of
    # microsoft/deberta-v3-small, are cached under artifacts_dir() and never run repository code.
    import transformers

    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    calls = {}

    def fake(kind, result):
        def from_pretrained(name, **kwargs):
            calls[kind] = (name, kwargs)
            return result

        return from_pretrained

    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        fake("model", tiny_network()),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake("tokenizer", tiny_tokenizer(64))
    )
    model = DebertaModel.pretrained(max_length=256, batch_size=7)
    assert isinstance(model, DebertaModel)
    assert model.batch_size == 7
    assert re.fullmatch(r"[0-9a-f]{40}", models.DEBERTA_REVISION)
    for kind in ("model", "tokenizer"):
        name, kwargs = calls[kind]
        assert name == "microsoft/deberta-v3-small"
        assert kwargs["revision"] == models.DEBERTA_REVISION
        assert kwargs["cache_dir"] == str(tmp_path / "hf")
        assert not kwargs.get("trust_remote_code")
    assert calls["model"][1]["num_labels"] == 28
    assert calls["model"][1]["label2id"] == {name: i for i, name in enumerate(data.OCCUPATIONS)}
    # The published checkpoint is float16 and transformers keeps that unless told otherwise;
    # training needs float32 master weights (float16 ones turn into NaN at the first step).
    assert calls["model"][1]["dtype"] == torch.float32
    assert calls["model"][1]["id2label"] == dict(enumerate(data.OCCUPATIONS))
    assert calls["tokenizer"][1]["model_max_length"] == 256


def test_logits_are_float32_with_one_row_per_text(tiny_network, tiny_tokenizer):
    logits = _model(tiny_network, tiny_tokenizer).logits(TEXTS)
    assert logits.dtype == np.float32
    assert logits.shape == (4, 28)


def test_predict_returns_the_occupation_name_with_the_largest_logit(tiny_network, tiny_tokenizer):
    model = _model(tiny_network, tiny_tokenizer)
    model.network.classifier.bias.data[data.OCCUPATIONS.index("surgeon")] = 100.0
    assert list(model.predict(TEXTS)) == ["surgeon"] * len(TEXTS)


def test_class_log_odds_is_the_class_logit_minus_the_log_sum_of_exponentials_of_the_others(
    tiny_network, tiny_tokenizer
):
    model = _model(tiny_network, tiny_tokenizer)
    z = model.logits(TEXTS).astype(np.float64)
    for index in (0, 13, 27):
        others = np.delete(z, index, axis=1)
        expected = z[:, index] - np.log(np.exp(others).sum(axis=1))
        np.testing.assert_allclose(model.class_log_odds(TEXTS, data.OCCUPATIONS[index]), expected)


def test_class_log_odds_does_not_saturate_where_probabilities_do(tiny_network, tiny_tokenizer):
    # A bias of 200 on nurse makes the float32 probability of nurse exactly 1, so log-odds from
    # probabilities would be infinite. From logits they stay finite.
    model = _model(tiny_network, tiny_tokenizer)
    model.network.classifier.bias.data[data.OCCUPATIONS.index("nurse")] = 200.0
    assert (model.predict_proba(TEXTS)[:, 13] == 1.0).all()
    odds = model.class_log_odds(TEXTS, "nurse")
    assert np.isfinite(odds).all()
    assert (odds > 150).all()


def _batch_widths(model):
    """Record the padded length of every batch the network sees."""
    widths = []
    model.network.register_forward_pre_hook(
        lambda _module, _args, kwargs: widths.append(kwargs["input_ids"].shape[1]),
        with_kwargs=True,
    )
    return widths


def test_texts_are_batched_by_token_length_so_short_ones_are_not_padded_to_long_ones(
    tiny_network, tiny_tokenizer
):
    # Token lengths with [CLS] and [SEP]: 4, 5, 3, 6. In input order batches of 2 would be padded
    # to 5 and 6; sorted they are padded to 4 and 6.
    model = _model(tiny_network, tiny_tokenizer, batch_size=2)
    widths = _batch_widths(model)
    model.logits(["alpha beta", "gamma delta epsilon", "zeta", "eta theta alpha beta"])
    assert widths == [4, 6]


def test_logits_come_back_in_input_order_and_equal_those_of_each_text_alone(
    tiny_network, tiny_tokenizer
):
    model = _model(tiny_network, tiny_tokenizer, batch_size=2)
    texts = ["eta theta alpha beta", "zeta", "gamma delta epsilon", "alpha beta", "zeta"]
    alone = np.concatenate([model.logits([text]) for text in texts])
    np.testing.assert_allclose(model.logits(texts), alone, atol=1e-5)
    assert not np.allclose(alone[0], alone[1], atol=1e-2)  # order matters in this fixture


def test_representations_come_back_in_input_order_too(tiny_network, tiny_tokenizer):
    model = _model(tiny_network, tiny_tokenizer, batch_size=2)
    texts = ["eta theta alpha beta", "zeta", "gamma delta epsilon", "alpha beta"]
    layers, final = model.representations(texts)
    for i, text in enumerate(texts):
        alone_layers, alone_final = model.representations([text])
        np.testing.assert_allclose(layers[i], alone_layers[0], atol=1e-4)
        np.testing.assert_allclose(final[i], alone_final[0], atol=1e-4)


def test_batches_yield_padded_inputs_sorted_by_length_with_the_positions_of_their_texts(
    tiny_network, tiny_tokenizer
):
    # Token lengths of TEXTS are 4, 5, 3 and 7 (two special tokens each); batch size 2 sorts
    # them as [zeta, alpha beta] and [gamma delta epsilon, the long one].
    model = _model(tiny_network, tiny_tokenizer, batch_size=2)
    batches = list(model.batches(TEXTS))
    assert [list(positions) for positions, _ in batches] == [[2, 0], [1, 3]]
    assert [tuple(inputs["input_ids"].shape) for _, inputs in batches] == [(2, 4), (2, 7)]
    assert not model.network.training


def test_mean_pooled_representations_average_the_states_of_the_non_padding_tokens(
    tiny_network, tiny_tokenizer
):
    # One text at a time there is no padding, so the plain mean over its tokens (the two special
    # tokens included) is the reference; batch size 3 pads the shorter texts.
    model = _model(tiny_network, tiny_tokenizer, batch_size=3)
    layers, _ = model.representations(TEXTS, pooling="mean")
    assert layers.shape == (4, 3, 32)
    for row, text in zip(layers, TEXTS, strict=True):
        with torch.no_grad():
            states = model.network.deberta(
                **model.tokenizer([text], return_tensors="pt"), output_hidden_states=True
            ).hidden_states
        expected = torch.stack([state[0].mean(dim=0) for state in states]).numpy()
        np.testing.assert_allclose(row, expected, atol=1e-5)


def test_representations_pool_at_the_cls_token_unless_told_otherwise(tiny_network, tiny_tokenizer):
    model = _model(tiny_network, tiny_tokenizer)
    cls = model.representations(TEXTS)[0]
    np.testing.assert_array_equal(cls, model.representations(TEXTS, pooling="cls")[0])
    assert not np.allclose(cls, model.representations(TEXTS, pooling="mean")[0])


def test_representations_refuse_an_unknown_pooling(tiny_network, tiny_tokenizer):
    with pytest.raises(ValueError, match="pooling"):
        _model(tiny_network, tiny_tokenizer).representations(TEXTS, pooling="max")
