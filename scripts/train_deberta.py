"""Fine-tune DeBERTa-v3-small, pick the learning rate on dev, and evaluate once on test."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from hiring_bias_xai import data, fairness
from hiring_bias_xai.config import RESULTS_DIR, SEED, artifacts_dir
from hiring_bias_xai.models import DebertaModel
from hiring_bias_xai.training import WARMUP_RATIO, seed_everything, train

LEARNING_RATES = (5e-5, 3e-5)
EPOCHS = 3
BATCH_SIZE = 8
ACCUMULATION = 4
EVAL_EVERY = 375
PATIENCE = 3
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _write_json(path: Path, content: dict) -> None:
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def _predict(model: DebertaModel, texts) -> list[str]:
    return [model.labels[i] for i in model.logits(list(texts)).argmax(axis=1)]


def _share_women(frame) -> dict:
    return (frame["gender"] == "female").groupby(frame["occupation"]).mean().to_dict()


def _length_report(texts) -> dict:
    """Token lengths of the bios before any cutting, and the share `MAX_LENGTH` would cut."""
    tokenizer = DebertaModel.pretrained(MAX_LENGTH).tokenizer
    lengths = np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])
    percentiles = {f"p{q:g}": float(np.percentile(lengths, q)) for q in (50, 90, 95, 99, 99.9)}
    return {
        "value": MAX_LENGTH,
        "token_length_percentiles": percentiles | {"max": int(lengths.max())},
        "share_above": {str(n): float(np.mean(lengths > n)) for n in (128, 256, 512)},
        "share_truncated": float(np.mean(lengths > MAX_LENGTH)),
    }


def _fit(learning_rate: float, train_frame, dev_frame):
    seed_everything()
    model = DebertaModel.pretrained(MAX_LENGTH)
    model.network.to(DEVICE)
    started = time.perf_counter()
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
    return model, rows, time.perf_counter() - started


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    train_frame, dev_frame = data.load_split("train"), data.load_split("dev")
    max_length_report = _length_report(train_frame["text"])
    on_gpu = DEVICE == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    runs, log, best_f1 = [], [], -1.0
    for learning_rate in LEARNING_RATES:
        model, rows, seconds = _fit(learning_rate, train_frame, dev_frame)
        log += [{"base_learning_rate": learning_rate} | row for row in rows]
        best = max(rows, key=lambda row: row["dev_macro_f1"])  # the first of equal rows
        runs.append(
            {
                "learning_rate": learning_rate,
                "best_dev_macro_f1": best["dev_macro_f1"],
                "best_step": best["step"],
                "best_epoch": best["epoch"],
                "evaluations": len(rows),
                "training_seconds": round(seconds, 1),
            }
        )
        if best["dev_macro_f1"] > best_f1:
            best_f1, chosen = best["dev_macro_f1"], runs[-1]
            model.save()
    peak_memory = round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_gpu else None
    model = DebertaModel.load()
    model.network.to(DEVICE)
    test_frame = data.load_split("test")
    predicted = _predict(model, test_frame["text"])
    predictions = artifacts_dir() / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_frame["id"], "predicted_occupation": predicted}).to_csv(
        predictions / "deberta_test.csv", index=False
    )
    report = fairness.evaluate(
        test_frame["occupation"], predicted, test_frame["gender"], _share_women(train_frame)
    )
    train_f1 = float(
        f1_score(train_frame["occupation"], _predict(model, train_frame["text"]), average="macro")
    )
    columns = ["base_learning_rate", "step", "epoch", "train_loss", "dev_loss", "dev_macro_f1"]
    pd.DataFrame(log)[[*columns, "learning_rate"]].to_csv(
        results_dir / "training_log.csv", index=False
    )
    _write_json(
        results_dir / "metrics_deberta.json",
        {
            "hyperparameters": {
                "learning_rate": chosen["learning_rate"],
                "best_step": chosen["best_step"],
                "max_epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": ACCUMULATION,
                "effective_batch_size": BATCH_SIZE * ACCUMULATION,
                "warmup_ratio": WARMUP_RATIO,
                "evaluation_every_steps": EVAL_EVERY,
                "patience_in_evaluations": PATIENCE,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "precision": "bfloat16 autocast on CUDA, float32 otherwise",
            },
            "peak_gpu_memory_mb": peak_memory,
            "runs": runs,
            "max_length": max_length_report,
            "overfitting_check": {
                "train_macro_f1": train_f1,
                "test_macro_f1": report["macro_f1"]["value"],
                "gap": train_f1 - report["macro_f1"]["value"],
            },
            "test": report,
            "training_seconds": round(sum(run["training_seconds"] for run in runs), 1),
        },
    )


if __name__ == "__main__":
    main()
