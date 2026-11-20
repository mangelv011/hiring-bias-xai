"""Two pilots before the Shapley run: which removal to use, and how many permutations it needs.

Both use the main model. The removal pilot drops a share of the words of dev bios at random with
each removal and compares the macro-F1 drop. The cost pilot runs Shapley on showcase-subset bios
with several permutation counts and two seeds each, and measures time and agreement between seeds.
`results/shapley_pilot.json` holds both with their rules and the choices `shapley_run.py` reads.
"""

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

from hiring_bias_xai import attribution, data, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Measured on the RTX 3050 (4 GB) with 500 dev bios, inference mode and bfloat16 autocast: peak
# allocated memory 1,190 MB at batch 64, below the 2,500 MB ceiling; throughput is flat from 32 up.
BATCH_SIZE = 64

REMOVAL_BIOS = 200
REMOVAL_DRAWS = 5
DROPPED_SHARE = 0.3
REMOVAL_RULE = (
    "Drop 30% of the words of each of 200 random dev bios, the same words for both removals, "
    "5 times. The removal with the smaller mean macro-F1 drop (macro-F1 on the unchanged bios "
    "minus macro-F1 on the changed ones) is the primary removal; the other is used in #9. A "
    "tie goes to delete."
)

COST_BIOS = 50
PERMUTATIONS = (5, 10, 20)
SEEDS = (0, 1)
SPEARMAN_THRESHOLD = 0.9
COST_RULE = (
    "50 random showcase-subset bios, the primary removal, the permutation counts 5, 10 and 20 "
    "with seeds 0 and 1 each. Seed agreement of a bio is the Spearman correlation between the "
    "values of its two seeds (bios with fewer than 2 words or constant values are left out). "
    "The permutation count is the smallest whose median Spearman is at least 0.9; if none "
    "reaches it, the largest is used and `threshold_met` is false."
)
FULL_RUN_LIMIT_SECONDS = 3 * 3600
FULL_RUN_RULE = (
    "Projected time = seconds per bio of the chosen count x the number of showcase-subset bios, "
    "main model. If it exceeds 3 hours, fall back to leave-one-out occlusion."
)


def _load_main(batch_size: int):
    model = DebertaModel.load(batch_size)
    model.network.to(DEVICE)
    return model


def _macro_f1(labels, predicted) -> float:
    return float(f1_score(labels, predicted, average="macro"))


def removal_pilot(model, dev) -> dict:
    texts, labels = list(dev["text"]), list(dev["occupation"])
    unperturbed = _macro_f1(labels, model.predict(texts))
    rng = np.random.default_rng(SEED)
    drops = {removal: [] for removal in attribution.REMOVALS}
    for _ in range(REMOVAL_DRAWS):
        words = [text.split() for text in texts]
        masks = []
        for bio in words:
            mask = np.zeros(len(bio), dtype=bool)
            mask[rng.choice(len(bio), size=round(DROPPED_SHARE * len(bio)), replace=False)] = True
            masks.append(mask)
        for removal in attribution.REMOVALS:
            changed = [
                attribution.remove_words(bio, mask, removal)
                for bio, mask in zip(words, masks, strict=True)
            ]
            drops[removal].append(unperturbed - _macro_f1(labels, model.predict(changed)))
    mean_drop = {removal: float(np.mean(values)) for removal, values in drops.items()}
    primary = min(attribution.REMOVALS, key=mean_drop.get)  # a tie goes to the first: delete
    return {
        "rule": REMOVAL_RULE,
        "n_bios": len(texts),
        "dropped_share": DROPPED_SHARE,
        "n_draws": REMOVAL_DRAWS,
        "macro_f1_unperturbed": unperturbed,
        "macro_f1_drops": drops,
        "macro_f1_drop_mean": mean_drop,
        "primary_removal": primary,
        "secondary_removal": next(r for r in attribution.REMOVALS if r != primary),
    }


def choose_permutations(median_spearman: dict, threshold: float) -> tuple[int, bool]:
    """The smallest count whose median Spearman reaches `threshold`, and whether one did.

    If none does, the largest count and False. A missing or NaN median does not reach it.
    """
    for count in sorted(median_spearman):
        value = median_spearman[count]
        if value is not None and value >= threshold:
            return count, True
    return max(median_spearman), False


def _spearman(one: np.ndarray, other: np.ndarray) -> float:
    if len(one) < 2 or np.ptp(one) == 0 or np.ptp(other) == 0:
        return float("nan")
    return float(spearmanr(one, other).statistic)


def cost_pilot(model, subset, removal: str, n_full_run: int) -> dict:
    texts, targets = list(subset["text"]), list(subset["occupation"])
    model.logits(texts[:8])  # the first calls are slower: keep that out of the timings
    by_count, medians = {}, {}
    for count in PERMUTATIONS:
        runs = [attribution.shapley(model, texts, targets, count, removal, seed) for seed in SEEDS]
        rhos = [
            _spearman(one.values, other.values) for one, other in zip(runs[0], runs[1], strict=True)
        ]
        defined = [rho for rho in rhos if not np.isnan(rho)]
        medians[count] = float(np.median(defined)) if defined else None
        by_count[count] = {
            "seconds_per_bio": sum(a.seconds for run in runs for a in run)
            / (len(runs) * len(texts)),
            "mean_words": float(np.mean([len(a.words) for a in runs[0]])),
            "mean_evaluations": float(np.mean([len(a.logits) for run in runs for a in run])),
            "median_spearman": medians[count],
            "n_bios_with_spearman": len(defined),
        }
    chosen, met = choose_permutations(medians, SPEARMAN_THRESHOLD)
    projected = by_count[chosen]["seconds_per_bio"] * n_full_run
    return {
        "rule": COST_RULE,
        "n_bios": len(texts),
        "seeds": list(SEEDS),
        "removal": removal,
        "threshold": SPEARMAN_THRESHOLD,
        "by_permutations": by_count,
        "chosen_permutations": chosen,
        "threshold_met": met,
        "full_run": {
            "rule": FULL_RUN_RULE,
            "n_bios": n_full_run,
            "projected_seconds": projected,
            "projected_hours": projected / 3600,
            "limit_hours": FULL_RUN_LIMIT_SECONDS / 3600,
            "method": "shapley" if projected <= FULL_RUN_LIMIT_SECONDS else "leave_one_out",
        },
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    started = time.perf_counter()
    model = _load_main(BATCH_SIZE)
    on_gpu = DEVICE == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    dev = data.load_split("dev").sample(REMOVAL_BIOS, random_state=SEED)
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    removal = removal_pilot(model, dev)
    cost = cost_pilot(
        model,
        subset.sample(COST_BIOS, random_state=SEED).reset_index(drop=True),
        removal["primary_removal"],
        len(subset),
    )
    method = cost["full_run"]["method"]
    chosen = {"method": method, "removal": removal["primary_removal"]}
    if method == "shapley":
        chosen["n_permutations"] = cost["chosen_permutations"]
    results.write_json(
        results_dir / "shapley_pilot.json",
        {
            "removal_pilot": removal,
            "cost_pilot": cost,
            "chosen": chosen,
            "inference_batch_size": BATCH_SIZE,
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if on_gpu
            else None,
            "runtime_seconds": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
