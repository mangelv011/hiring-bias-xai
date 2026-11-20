"""Does removing the cues close the gap? Five conditions of model and test text, scored alike.

Conditions: (a) the original model on the raw test bios, (b) the original model on scrubbed test
bios, (c) the scrubbed model on raw bios, (d) the scrubbed model on scrubbed bios, (e) the original
model on raw bios with the proxy words of `results/proxy_words.json` deleted. Both models must
exist: the scrubbed one comes from `scripts/train_scrubbed.py`.

    python scripts/mitigation.py predict a    # GPU, about 2 minutes; one condition per call
    python scripts/mitigation.py report       # CPU: writes results/mitigation.json

`predict` saves the predicted occupation of every test bio to `HBX_ARTIFACTS/predictions` and
skips a condition whose predictions are complete, so a killed run restarts where it stopped.
`report` scores every condition on the test bios outside the showcase subset (the proxy words were
chosen on the showcase subset), and (a) to (d) also on the full test split.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from hiring_bias_xai import data, fairness, gap_statistics, proxy_words, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel

CONDITIONS = {
    "a": {"model": "deberta", "text": "original"},
    "b": {"model": "deberta", "text": "scrubbed"},
    "c": {"model": "deberta-scrubbed", "text": "original"},
    "d": {"model": "deberta-scrubbed", "text": "scrubbed"},
    "e": {"model": "deberta", "text": "proxy words deleted"},
}
EVAL_BATCH_SIZE = 64
N_BOOT = 1000
N_PERM = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAIN_PREDICTIONS = "deberta"
LIMITATION = (
    "One training seed per model. A difference between the original and the scrubbed model mixes "
    "the effect of the text with the run-to-run variation of training, which I did not measure. "
    "The intervals cover the sampling of test bios only."
)


def _name(condition: str) -> str:
    return f"mitigation_{condition}"


def _is_complete(name: str, test: pd.DataFrame) -> bool:
    try:
        results.load_predictions(name, test)
    except (FileNotFoundError, ValueError):
        return False
    return True


def _texts(condition: str, test: pd.DataFrame, results_dir: Path) -> list[str]:
    variant = CONDITIONS[condition]["text"]
    if variant == "scrubbed":
        return list(data.load_split("test", "scrubbed")["text"])
    if variant == "original":
        return list(test["text"])
    words = set(json.loads((results_dir / "proxy_words.json").read_text())["words"])
    return [proxy_words.mask_words(text, words) for text in test["text"]]


def _predict(condition: str, results_dir: Path) -> None:
    test = data.load_split("test")
    if _is_complete(_name(condition), test):
        print(f"condition {condition}: predictions complete, skipped", flush=True)
        return
    started = time.perf_counter()
    model = DebertaModel.load(EVAL_BATCH_SIZE, name=CONDITIONS[condition]["model"])
    model.network.to(DEVICE)
    predicted = model.predict(_texts(condition, test, results_dir))
    results.save_predictions(_name(condition), test, predicted)
    seconds = time.perf_counter() - started
    print(f"condition {condition}: {len(test)} bios in {seconds:.0f} s", flush=True)


def _block(frame: pd.DataFrame, predicted: pd.Series, share_women: dict, occupations: list) -> dict:
    """Metrics of one set of bios: macro-F1, accuracy, RMS TPR gap with its null, showcase gaps."""
    report = fairness.evaluate(
        frame["occupation"], predicted, frame["gender"], share_women, n_boot=N_BOOT
    )
    null = gap_statistics.rms_permutation_test(
        frame["occupation"], predicted, frame["gender"], n_perm=N_PERM, seed=SEED
    )
    null = {
        "mean": null["null_mean"],
        "p95": null["null_p95"],
        "p_value": null["p_value"],
        "n_permutations": null["n_permutations"],
    }
    return {
        "n_bios": len(frame),
        "macro_f1": report["macro_f1"],
        "accuracy": report["accuracy"],
        "rms_tpr_gap": report["rms_tpr_gap"] | {"null": null},
        "tpr_gap": {o: report["per_occupation"][o]["tpr_gap"] for o in occupations},
    }


def _share_of_bios(units: list[list[str]], has) -> float:
    return sum(any(has(unit) for unit in bio) for bio in units) / len(units)


def _report(results_dir: Path) -> None:
    test = data.load_split("test")
    missing = [c for c in CONDITIONS if not _is_complete(_name(c), test)]
    if missing:
        raise FileNotFoundError(f"no complete predictions for conditions: {', '.join(missing)}")
    share_women = data.share_women(data.load_split("train"))
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    occupations = [
        row["occupation"]
        for side in ("male_dominated", "female_dominated")
        for row in showcase[side]
    ]
    showcase_ids = set(pd.read_csv(results_dir / "showcase_subset_ids.csv")["id"])
    outside = ~test["id"].isin(showcase_ids)
    rest = test[outside]
    words = json.loads((results_dir / "proxy_words.json").read_text())["words"]
    predictions = {c: results.load_predictions(_name(c), test) for c in CONDITIONS}
    conditions = {}
    for condition, description in CONDITIONS.items():
        block = _block(rest, predictions[condition][outside], share_women, occupations)
        if condition != "a":
            block["change_from_a"] = fairness.paired_difference(
                rest["occupation"],
                predictions["a"][outside],
                predictions[condition][outside],
                rest["gender"],
                share_women,
                n_boot=N_BOOT,
            )
        conditions[condition] = description | {"outside_showcase": block}
        if condition != "e":
            conditions[condition]["full_test"] = _block(
                test, predictions[condition], share_women, occupations
            )
    units = [text.split() for text in rest["text"]]
    main_run = results.load_predictions(MAIN_PREDICTIONS, test)
    results.write_json(
        results_dir / "mitigation.json",
        {
            "settings": {
                "n_bootstrap": N_BOOT,
                "n_permutations": N_PERM,
                "seed": SEED,
                "scoring_batch_size": EVAL_BATCH_SIZE,
            },
            "evaluation_set": {
                "test_bios": len(test),
                "showcase_bios": int((~outside).sum()),
                "bios_scored": len(rest),
            },
            "proxy_words": words,
            "bios_changed": {
                "gendered_word_share": _share_of_bios(units, data.is_gendered_word),
                "proxy_word_share": _share_of_bios(units, lambda w: data.word_core(w) in words),
            },
            "scoring_check": {
                "agreement_with_main_run": float((predictions["a"] == main_run).mean())
            },
            "conditions": conditions,
            "limitation": LIMITATION,
        },
    )


def main(results_dir: Path = RESULTS_DIR, argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["predict"] and len(argv) == 2:
        if argv[1] not in CONDITIONS:
            raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}; got {argv[1]!r}")
        _predict(argv[1], results_dir)
    elif argv == ["report"]:
        _report(results_dir)
    else:
        raise ValueError("usage: mitigation.py predict <condition> | mitigation.py report")


if __name__ == "__main__":
    main()
