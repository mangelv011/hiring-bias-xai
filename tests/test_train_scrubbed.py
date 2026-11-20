import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from hiring_bias_xai import data, training
from hiring_bias_xai.models import DebertaModel, _deberta_folder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_scrubbed.py"
MAIN_METRICS = Path(__file__).resolve().parents[1] / "results" / "metrics_deberta.json"

# On the scrubbed bios a nurse says alpha and a surgeon says gamma. The original bios say the
# opposite, so a script that trained on them would learn the reverse mapping.
SCRUBBED = {"nurse": "alpha alpha alpha", "surgeon": "gamma gamma gamma"}
ORIGINAL = {"nurse": "gamma gamma gamma", "surgeon": "alpha alpha alpha"}
TOY_CONSTANTS = {
    "LEARNING_RATE": 3e-3,
    "EPOCHS": 30,
    "BATCH_SIZE": 4,
    "ACCUMULATION": 1,
    "EVAL_EVERY": 40,
    "MAX_LENGTH": 64,
    "EVAL_BATCH_SIZE": 8,
    "DEVICE": "cpu",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("train_scrubbed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(name, per_cell, words):
    cells = [o for o in words for _ in range(2 * per_cell)]
    rows = [(f"{name}-{i}", words[o], o, "female") for i, o in enumerate(cells)]
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture
def toy(monkeypatch, tmp_path, tiny_network, tiny_tokenizer):
    """toy(**constants): the script's main() on synthetic splits and a tiny network.

    Returns the directory that holds `results/` and `artifacts/`; `toy.requested` lists the
    (split, variant) pairs asked of `data.load_split` and `toy.draws` the random draw made right
    when each model was built.
    """
    root = tmp_path
    requested, draws = [], []

    def load_split(split, variant="original"):
        requested.append((split, variant))
        words = SCRUBBED if variant == "scrubbed" else ORIGINAL
        return _split(split, {"train": 8, "dev": 2}[split], words)

    def pretrained(cls, max_length, batch_size=32):
        draws.append(torch.rand(1).item())
        return cls(tiny_network(), tiny_tokenizer(max_length), batch_size)

    def run(**constants):
        module = _load_script()
        for name, value in (TOY_CONSTANTS | constants).items():
            monkeypatch.setattr(module, name, value)
        module.main(root / "results")
        return root

    run.requested, run.draws = requested, draws
    monkeypatch.setattr(data, "load_split", load_split)
    monkeypatch.setattr(DebertaModel, "pretrained", classmethod(pretrained))
    monkeypatch.setenv("HBX_ARTIFACTS", str(root / "artifacts"))
    return run


def test_the_recipe_is_the_main_one_with_the_learning_rate_the_main_run_chose():
    # Epochs, batches, evaluation interval and length come from `training`. The rate is the one
    # the main run kept after trying both, read from its metrics.
    module = _load_script()
    chosen = json.loads(MAIN_METRICS.read_text())["hyperparameters"]["learning_rate"]
    assert module.LEARNING_RATE == chosen == 3e-5
    for name in ("EPOCHS", "BATCH_SIZE", "ACCUMULATION", "EVAL_EVERY", "MAX_LENGTH"):
        assert getattr(module, name) == getattr(training, name)
    assert module.EVAL_BATCH_SIZE == training.EVAL_BATCH_SIZE


def test_the_model_trains_and_is_chosen_on_the_scrubbed_train_and_dev_bios(toy):
    toy()
    assert sorted(toy.requested) == [("dev", "scrubbed"), ("train", "scrubbed")]


def test_the_saved_model_learned_the_scrubbed_text(toy):
    root = toy()
    assert sorted(p.name for p in (root / "artifacts" / "models").iterdir()) == ["deberta-scrubbed"]
    model = DebertaModel.load(name="deberta-scrubbed")
    assert model.predict(list(SCRUBBED.values())) == list(SCRUBBED)


def test_a_main_checkpoint_that_already_exists_is_not_touched(toy, tiny_network, tiny_tokenizer):
    DebertaModel(tiny_network(seed=3), tiny_tokenizer(64)).save()
    folder = _deberta_folder("deberta")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    toy(EPOCHS=1)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert _deberta_folder("deberta-scrubbed").is_dir()


def test_the_training_log_has_the_columns_of_the_main_log_and_one_row_per_evaluation(toy):
    log = pd.read_csv(toy() / "results" / "training_log_scrubbed.csv")
    assert list(log.columns) == [
        "base_learning_rate",
        "step",
        "epoch",
        "train_loss",
        "dev_loss",
        "dev_macro_f1",
        "learning_rate",
    ]
    assert log["base_learning_rate"].tolist() == [3e-3] * 6
    assert log["step"].tolist() == [40, 80, 120, 160, 200, 240]
    assert log["dev_macro_f1"].iloc[-1] == 1.0


def test_only_the_scrubbed_log_is_written_and_with_unix_line_endings(toy):
    results = toy(EPOCHS=1) / "results"
    assert [p.name for p in results.iterdir()] == ["training_log_scrubbed.csv"]
    assert b"\r" not in (results / "training_log_scrubbed.csv").read_bytes()


def test_the_script_prints_one_progress_line_per_evaluation_and_the_training_time(toy, capsys):
    toy(EPOCHS=1)
    lines = capsys.readouterr().out.splitlines()
    progress = [line for line in lines if "dev_macro_f1" in line]
    assert len(progress) == 1
    assert "lr=0.003" in progress[0] and "step=8" in progress[0] and "epoch=1" in progress[0]
    assert sum(line.startswith("training seconds: ") for line in lines) == 1


def test_the_head_is_initialised_after_seeding_the_global_generators(toy):
    toy(EPOCHS=1)
    assert toy.draws == [torch.rand(1, generator=torch.Generator().manual_seed(42)).item()]
