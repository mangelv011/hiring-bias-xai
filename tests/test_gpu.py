import numpy as np
import pytest
import torch


@pytest.mark.gpu
def test_cuda_is_available():
    assert torch.cuda.is_available()


@pytest.mark.gpu
def test_a_network_on_the_gpu_gives_numpy_probabilities_and_representations(
    tiny_network, tiny_tokenizer
):
    from hiring_bias_xai.models import DebertaModel

    model = DebertaModel(tiny_network().to("cuda"), tiny_tokenizer(64))
    texts = ["alpha beta", "gamma delta"]
    assert model.predict_proba(texts).shape == (2, 28)
    layers, final = model.representations(texts)
    assert (layers.shape, final.shape) == ((2, 3, 32), (2, 32))


@pytest.mark.gpu
def test_forward_passes_of_training_and_inference_run_in_bfloat16_on_the_gpu(
    tiny_network, tiny_tokenizer
):
    # Mixed precision: matrix products in bf16, weights in fp32.
    from hiring_bias_xai.models import DebertaModel
    from hiring_bias_xai.training import train

    model = DebertaModel(tiny_network().to("cuda"), tiny_tokenizer(64))
    seen = []
    model.network.classifier.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
    texts = ["alpha beta", "gamma delta"] * 4
    labels = ["nurse", "surgeon"] * 4
    train(
        model, texts, labels, texts, labels, learning_rate=1e-3, epochs=1, batch_size=4,
        eval_every=1,
    )  # fmt: skip
    training_dtypes, seen[:] = set(seen), []
    model.logits(texts)
    assert training_dtypes == {torch.bfloat16}
    assert set(seen) == {torch.bfloat16}
    assert next(model.network.parameters()).dtype == torch.float32


@pytest.mark.gpu
@pytest.mark.slow
def test_the_trained_checkpoint_predicts_ten_real_bios_twice_with_identical_outputs():
    # Needs the checkpoint that scripts/train_deberta.py saves and the real dataset.
    from hiring_bias_xai import data
    from hiring_bias_xai.models import DebertaModel

    texts = list(data.load_split("test")["text"][:10])
    model = DebertaModel.load()
    model.network.to("cuda")
    first, second = model.predict_proba(texts), model.predict_proba(texts)
    assert first.shape == (10, 28)
    assert next(model.network.parameters()).dtype == torch.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.gpu
def test_representations_run_in_bfloat16_on_the_gpu_and_come_back_as_float32(
    tiny_network, tiny_tokenizer
):
    from hiring_bias_xai.models import DebertaModel

    model = DebertaModel(tiny_network().to("cuda"), tiny_tokenizer(64))
    seen = []
    model.network.pooler.dense.register_forward_hook(lambda _m, _i, out: seen.append(out.dtype))
    layers, final = model.representations(["alpha beta", "gamma delta"])
    assert set(seen) == {torch.bfloat16}
    assert layers.dtype == final.dtype == np.float32
