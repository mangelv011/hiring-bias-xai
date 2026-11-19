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
