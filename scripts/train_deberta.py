"""Fine-tune DeBERTa-v3-small, pick the learning rate on dev, and evaluate once on test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from hiring_bias_xai import data, fairness, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel
from hiring_bias_xai.training import (
    ACCUMULATION,
    BATCH_SIZE,
    EPOCHS,
    EVAL_BATCH_SIZE,
    EVAL_EVERY,
    MAX_LENGTH,
    WARMUP_RATIO,
    fit_deberta,
    progress_printer,
)

LEARNING_RATES = (5e-5, 3e-5)
BASELINE_METRICS = RESULTS_DIR / "metrics_baseline.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _length_report(texts) -> dict:
    """Token lengths of the bios before any cutting, and the share `MAX_LENGTH` would cut."""
    tokenizer = DebertaModel.pretrained(MAX_LENGTH, EVAL_BATCH_SIZE).tokenizer
    lengths = np.array([len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]])
    percentiles = {f"p{q:g}": float(np.percentile(lengths, q)) for q in (50, 90, 95, 99, 99.9)}
    return {
        "value": MAX_LENGTH,
        "token_length_percentiles": percentiles | {"max": int(lengths.max())},
        "share_above": {str(n): float(np.mean(lengths > n)) for n in (128, 256, 512)},
        "share_truncated": float(np.mean(lengths > MAX_LENGTH)),
    }


def _comparison(deberta: dict, baseline: dict) -> dict:
    """Both models' value and interval for the two headline metrics, and if the intervals meet."""
    comparison = {}
    for name in ("macro_f1", "rms_tpr_gap"):
        a, b = baseline[name], deberta[name]
        comparison[name] = {
            "baseline": a,
            "deberta": b,
            "cis_overlap": bool(a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]),
        }
    return comparison


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    train_frame, dev_frame = data.load_split("train"), data.load_split("dev")
    max_length_report = _length_report(train_frame["text"])
    on_gpu = DEVICE == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    runs, log, best_f1 = [], [], -1.0
    for learning_rate in LEARNING_RATES:
        model, rows, seconds = fit_deberta(
            learning_rate,
            train_frame,
            dev_frame,
            device=DEVICE,
            max_length=MAX_LENGTH,
            eval_batch_size=EVAL_BATCH_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            accumulation=ACCUMULATION,
            eval_every=EVAL_EVERY,
            on_evaluation=progress_printer(f"lr={learning_rate:g}"),
        )
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
        del model  # frees the GPU before the next run, so the peak is that of one run
    peak_memory = round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_gpu else None
    model = DebertaModel.load(EVAL_BATCH_SIZE)
    model.network.to(DEVICE)
    test_frame = data.load_split("test")
    predicted = model.predict(list(test_frame["text"]))
    results.save_predictions("deberta", test_frame, predicted)
    report = fairness.evaluate(
        test_frame["occupation"], predicted, test_frame["gender"], data.share_women(train_frame)
    )
    train_f1 = float(
        f1_score(
            train_frame["occupation"], model.predict(list(train_frame["text"])), average="macro"
        )
    )
    columns = ["base_learning_rate", "step", "epoch", "train_loss", "dev_loss", "dev_macro_f1"]
    pd.DataFrame(log)[[*columns, "learning_rate"]].to_csv(
        results_dir / "training_log.csv", index=False, lineterminator="\n"
    )
    results.write_json(
        results_dir / "metrics_deberta.json",
        {
            "comparison_with_baseline": _comparison(
                report, json.loads(BASELINE_METRICS.read_text())["test"]
            ),
            "hyperparameters": {
                "learning_rate": chosen["learning_rate"],
                "best_step": chosen["best_step"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": ACCUMULATION,
                "effective_batch_size": BATCH_SIZE * ACCUMULATION,
                "warmup_ratio": WARMUP_RATIO,
                "evaluation_batch_size": EVAL_BATCH_SIZE,
                "evaluation_every_steps": EVAL_EVERY,
                "early_stopping": False,
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
