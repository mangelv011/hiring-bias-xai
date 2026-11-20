"""Fine-tune DeBERTa-v3-small on scrubbed bios with the recipe of the main model.

Only the text changes: `data.scrub` is applied to the train and dev bios. The learning rate is the
one the main run chose, the epochs, batches and seed are those of `training`, and the weights of
the best dev macro-F1 are restored. The model is saved as the checkpoint `deberta-scrubbed`, so
the main model is not touched, and its log goes to `results/training_log_scrubbed.csv`. The run
takes about 1.7 hours on the project's GPU, so launch it on its own.
"""

from pathlib import Path

import pandas as pd
import torch

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.training import (
    ACCUMULATION,
    BATCH_SIZE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    EVAL_EVERY,
    MAX_LENGTH,
    fit_deberta,
    progress_printer,
)

LEARNING_RATE = 3e-5
CHECKPOINT = "deberta-scrubbed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    train_frame = data.load_split("train", "scrubbed")
    dev_frame = data.load_split("dev", "scrubbed")
    model, rows, seconds = fit_deberta(
        LEARNING_RATE,
        train_frame,
        dev_frame,
        device=DEVICE,
        max_length=MAX_LENGTH,
        eval_batch_size=EVAL_BATCH_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        accumulation=ACCUMULATION,
        eval_every=EVAL_EVERY,
        on_evaluation=progress_printer(f"lr={LEARNING_RATE:g}"),
    )
    model.save(CHECKPOINT)
    columns = ["base_learning_rate", "step", "epoch", "train_loss", "dev_loss", "dev_macro_f1"]
    pd.DataFrame([{"base_learning_rate": LEARNING_RATE} | row for row in rows])[
        [*columns, "learning_rate"]
    ].to_csv(results_dir / "training_log_scrubbed.csv", index=False, lineterminator="\n")
    print(f"training seconds: {seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
