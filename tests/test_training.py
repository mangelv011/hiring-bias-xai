import pytest

from hiring_bias_xai.models import DebertaModel
from hiring_bias_xai.training import train

# Two separable occupations: nurse bios use alpha and beta, surgeon bios use gamma and delta.
NURSE, SURGEON = "nurse", "surgeon"


def _bios(n_per_class):
    texts = ["alpha beta alpha", "beta alpha beta"] * (n_per_class // 2)
    texts += ["gamma delta gamma", "delta gamma delta"] * (n_per_class // 2)
    return texts, [NURSE] * n_per_class + [SURGEON] * n_per_class


@pytest.fixture
def model(tiny_network, tiny_tokenizer):
    return DebertaModel(tiny_network(), tiny_tokenizer(64))


def test_training_on_a_separable_task_brings_the_dev_loss_far_below_chance(model):
    # A random 28-way classifier has a loss near ln 28 = 3.33; two separable classes can go
    # far below it.
    train_texts, train_labels = _bios(16)
    dev_texts, dev_labels = _bios(4)
    rows = train(
        model,
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        learning_rate=1e-2,
        epochs=3,
        batch_size=8,
        eval_every=2,
    )
    assert rows[0]["dev_loss"] > rows[-1]["dev_loss"]
    assert rows[-1]["dev_loss"] < 1.0
