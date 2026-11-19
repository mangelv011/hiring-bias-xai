import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_deberta.py"

# Two easy occupations: every nurse bio says "alpha" and every surgeon bio "gamma".
WORDS = {"nurse": "alpha alpha alpha", "surgeon": "gamma gamma gamma"}


def _split(name, per_cell):
    cells = [(o, g) for o in WORDS for g in ("female", "male") for _ in range(per_cell)]
    rows = [(f"{name}-{i}", WORDS[o], o, g) for i, (o, g) in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture(scope="module")
def run(tmp_path_factory, tiny_network, tiny_tokenizer):
    """Run the script's main() once on synthetic splits and a tiny network; give the paths."""
    root = tmp_path_factory.mktemp("deberta")
    splits = {"train": _split("train", 8), "dev": _split("dev", 2), "test": _split("test", 2)}

    def pretrained(cls, max_length, batch_size=32):
        return cls(tiny_network(), tiny_tokenizer(max_length), batch_size)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data, "load_split", lambda split, variant="original": splits[split])
        patch.setattr(DebertaModel, "pretrained", classmethod(pretrained))
        patch.setenv("HBX_ARTIFACTS", str(root / "artifacts"))
        spec = importlib.util.spec_from_file_location("train_deberta", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        patch.setattr(module, "LEARNING_RATES", (3e-3, 1e-5))
        patch.setattr(module, "EPOCHS", 30)
        patch.setattr(module, "BATCH_SIZE", 4)
        patch.setattr(module, "ACCUMULATION", 1)
        patch.setattr(module, "EVAL_EVERY", 40)
        patch.setattr(module, "PATIENCE", 100)
        patch.setattr(module, "MAX_LENGTH", 64)
        module.main(root / "results")
        yield root


def _metrics(run):
    return json.loads((run / "results" / "metrics_deberta.json").read_text())


def test_the_learning_rate_with_the_best_dev_macro_f1_is_chosen(run):
    # 1e-5 barely moves the toy network in 240 steps; 3e-3 learns the task. The better rate
    # comes first, so a script that kept the last model would fail the next test.
    assert _metrics(run)["hyperparameters"]["learning_rate"] == 3e-3


def test_the_saved_checkpoint_is_the_one_of_the_chosen_learning_rate(run, monkeypatch):
    monkeypatch.setenv("HBX_ARTIFACTS", str(run / "artifacts"))
    probabilities = DebertaModel.load().predict_proba(list(WORDS.values()))
    assert [data.OCCUPATIONS[i] for i in probabilities.argmax(axis=1)] == list(WORDS)


def test_the_training_log_has_one_row_per_evaluation_of_every_run(run):
    # 32 train bios in batches of 4 make 8 steps per epoch, 240 in 30 epochs; an evaluation
    # every 40 steps gives 6 rows per run, and the runs follow LEARNING_RATES.
    log = pd.read_csv(run / "results" / "training_log.csv")
    assert list(log.columns) == [
        "base_learning_rate",
        "step",
        "epoch",
        "train_loss",
        "dev_loss",
        "dev_macro_f1",
        "learning_rate",
    ]
    assert log["base_learning_rate"].tolist() == [3e-3] * 6 + [1e-5] * 6
    assert log["step"].tolist() == [40, 80, 120, 160, 200, 240] * 2
