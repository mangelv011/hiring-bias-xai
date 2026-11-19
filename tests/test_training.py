import pytest
import torch

from hiring_bias_xai.models import DebertaModel
from hiring_bias_xai.training import train

# Two separable occupations: nurse bios say alpha, surgeon bios say gamma.
NURSE, SURGEON = "nurse", "surgeon"


def _bios(n_per_class):
    texts = ["alpha alpha alpha"] * n_per_class + ["gamma gamma gamma"] * n_per_class
    return texts, [NURSE] * n_per_class + [SURGEON] * n_per_class


@pytest.fixture
def model(tiny_network, tiny_tokenizer):
    return DebertaModel(tiny_network(), tiny_tokenizer(64))


def test_training_on_a_separable_task_brings_the_dev_loss_far_below_chance(model):
    # A random 28-way classifier has a loss near ln 28 = 3.33 and a macro-F1 below 1; a model
    # that has learned two separable classes gets every dev bio right.
    train_texts, train_labels = _bios(16)
    dev_texts, dev_labels = _bios(4)
    rows = train(
        model,
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        learning_rate=3e-3,
        epochs=30,
        batch_size=4,
        eval_every=10,
    )
    assert rows[0]["dev_loss"] > rows[-1]["dev_loss"]
    assert rows[-1]["dev_loss"] < 0.1
    assert rows[-1]["dev_macro_f1"] == 1.0


def test_the_learning_rate_warms_up_over_six_percent_of_the_steps_then_decays_linearly_to_zero(
    model,
):
    # 40 bios, batches of 4 and 5 epochs make 50 steps; six percent of 50 is 3 warm-up steps.
    # The rate after step k is base * k / 3 while warming up, then base * (50 - k) / (50 - 3).
    texts, labels = _bios(20)
    rows = train(
        model,
        texts,
        labels,
        texts[:4],
        labels[:4],
        learning_rate=1e-3,
        epochs=5,
        batch_size=4,
        eval_every=1,
    )
    rate = {row["step"]: row["learning_rate"] for row in rows}
    assert len(rows) == 50
    expected = {1: 1e-3 / 3, 2: 2e-3 / 3, 3: 1e-3, 26: 1e-3 * 24 / 47, 50: 0.0}
    for step, value in expected.items():
        assert rate[step] == pytest.approx(value)


def test_each_evaluation_gets_a_row_and_the_last_step_is_always_evaluated(model):
    # 32 bios in batches of 8 give 4 steps per epoch, 8 in two epochs. Evaluating every 3 steps
    # gives steps 3 and 6, plus the final step 8.
    texts, labels = _bios(16)
    rows = train(
        model,
        texts,
        labels,
        texts[:4],
        labels[:4],
        learning_rate=1e-3,
        epochs=2,
        batch_size=8,
        eval_every=3,
    )
    assert [(row["step"], row["epoch"]) for row in rows] == [(3, 1), (6, 2), (8, 2)]
    assert set(rows[0]) == {
        "step",
        "epoch",
        "train_loss",
        "learning_rate",
        "dev_loss",
        "dev_macro_f1",
    }


def _plateau_run(model, patience):
    """Learn the task, then train 10 more epochs on a dev set whose F1 cannot rise.

    One nurse dev bio carries the surgeon label, so the F1 of a model that has learned the
    task is flat (below 1) while its dev loss keeps moving.
    """
    texts, labels = _bios(16)
    dev_texts, dev_labels = _bios(4)
    dev_labels[0] = SURGEON
    args = (model, texts, labels, dev_texts, dev_labels)
    train(*args, learning_rate=3e-3, epochs=30, batch_size=4, eval_every=240)
    rows = train(
        *args, learning_rate=1e-4, epochs=10, batch_size=4, eval_every=10, patience=patience
    )
    return rows, dev_texts, dev_labels


def test_training_stops_after_patience_evaluations_without_a_better_dev_macro_f1(model):
    rows, _, _ = _plateau_run(model, patience=3)
    scores = [row["dev_macro_f1"] for row in rows]
    assert scores[0] > 0.8  # the task was learned before the plateau
    assert len(rows) == scores.index(max(scores)) + 1 + 3
    assert len(rows) < 8


def test_training_leaves_the_model_at_the_evaluation_with_the_best_dev_macro_f1(model):
    rows, dev_texts, dev_labels = _plateau_run(model, patience=3)
    scores = [row["dev_macro_f1"] for row in rows]
    best = rows[scores.index(max(scores))]
    assert rows[-1]["dev_loss"] != pytest.approx(best["dev_loss"])  # the last step is not the best
    logits = torch.from_numpy(model.logits(dev_texts))
    targets = torch.tensor([model.labels.index(label) for label in dev_labels])
    restored = torch.nn.functional.cross_entropy(logits, targets)
    assert float(restored) == pytest.approx(best["dev_loss"], rel=1e-4)


def test_gradient_accumulation_gives_the_same_updates_as_the_batch_it_adds_up_to(
    tiny_network, tiny_tokenizer
):
    # Two micro-batches of 4 bios are one batch of 8: the same weights after every step. The
    # networks have no dropout, which would draw different random numbers in the two runs.
    texts, labels = _bios(16)

    def run(**sizes):
        model = DebertaModel(tiny_network(dropout=0.0), tiny_tokenizer(64))
        torch.manual_seed(1)
        return train(
            model, texts, labels, texts[::4], labels[::4],
            learning_rate=1e-3, epochs=2, eval_every=1, **sizes,
        )  # fmt: skip

    whole = run(batch_size=8)
    split = run(batch_size=4, accumulation=2)
    assert [row["step"] for row in split] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [row["dev_loss"] for row in split] == pytest.approx(
        [row["dev_loss"] for row in whole], rel=1e-4
    )
