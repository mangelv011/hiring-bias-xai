"""Fine-tune DeBERTa-v3-small, pick the learning rate on dev, and evaluate once on test."""

import json
from pathlib import Path

import torch

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.models import DebertaModel
from hiring_bias_xai.training import seed_everything, train

LEARNING_RATES = (5e-5, 3e-5)
EPOCHS = 3
BATCH_SIZE = 8
ACCUMULATION = 4
EVAL_EVERY = 375
PATIENCE = 3
MAX_LENGTH = 256


def _write_json(path: Path, content: dict) -> None:
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def _fit(learning_rate: float, train_frame, dev_frame):
    seed_everything()
    model = DebertaModel.pretrained(MAX_LENGTH)
    model.network.to("cuda" if torch.cuda.is_available() else "cpu")
    rows = train(
        model,
        list(train_frame["text"]),
        list(train_frame["occupation"]),
        list(dev_frame["text"]),
        list(dev_frame["occupation"]),
        learning_rate=learning_rate,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        accumulation=ACCUMULATION,
        eval_every=EVAL_EVERY,
        patience=PATIENCE,
    )
    return model, rows


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    train_frame, dev_frame = data.load_split("train"), data.load_split("dev")
    best_f1, best_learning_rate = -1.0, None
    for learning_rate in LEARNING_RATES:
        model, rows = _fit(learning_rate, train_frame, dev_frame)
        dev_f1 = max(row["dev_macro_f1"] for row in rows)
        if dev_f1 > best_f1:
            best_f1, best_learning_rate = dev_f1, learning_rate
            model.save()
    _write_json(
        results_dir / "metrics_deberta.json",
        {"hyperparameters": {"learning_rate": best_learning_rate}},
    )


if __name__ == "__main__":
    main()
