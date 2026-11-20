"""Activation patching on the main model: does the gender swap travel through the layers?

Clean is the original showcase bio, corrupted its `counterfactual.swap_gender`. Eligible bios
are those whose swap moves the true-class log-odds by more than the 95th percentile of the move
under the removal control (`data.scrub`) over all bios, and whose two token sequences have the
same length. For each layer output, the clean states are patched into the corrupted run at the
positions where the tokens differ and at all positions (`patching.patch_pairs`); the restored
share of the true-class log-odds is summarised in `results/patching.csv` and the counts in
`results/patching.json`. Forward passes only; needs the saved main model and the GPU, with
nothing else running.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hiring_bias_xai import counterfactual, data, patching, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.models import DebertaModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
N_BOOT = patching.N_BOOT
QUANTILE = 0.95
MAX_ALL_POSITIONS_ERROR = 1e-2  # the hook check: patching every position must give share 1


def _load() -> DebertaModel:
    model = DebertaModel.load(BATCH_SIZE)
    model.network.to(DEVICE)
    return model


def main(results_dir: Path = RESULTS_DIR) -> None:
    started = time.perf_counter()
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    model = _load()
    texts = list(subset["text"])
    swapped = [counterfactual.swap_gender(text) for text in texts]
    removed = [data.scrub(text) for text in texts]
    target = np.array([model.labels.index(name) for name in subset["occupation"]])
    original, after_swap, after_removal = (
        patching.true_class_log_odds(model.logits(items), target)
        for items in (texts, swapped, removed)
    )
    eligible, bar = patching.eligible(after_swap - original, after_removal - original, QUANTILE)
    same = patching.equal_length(model.tokenizer, texts, swapped)
    keep = np.flatnonzero(eligible & same)
    out = patching.patch_pairs(
        model,
        [texts[i] for i in keep],
        [swapped[i] for i in keep],
        target[keep],
    )
    rows, worst = [], 0.0
    for layer in range(out["patched"].shape[1]):
        for k, position_set in enumerate(patching.POSITION_SETS):
            shares = patching.restored_share(
                out["clean"], out["corrupted"], out["patched"][:, layer, k]
            )
            if position_set == "all":
                worst = max(worst, float(np.abs(shares - 1).max()))
            rows.append(
                {"layer": layer, "position_set": position_set}
                | patching.summarise(shares, N_BOOT, SEED)
            )
    if worst > MAX_ALL_POSITIONS_ERROR:
        raise RuntimeError(f"patching all positions left a share {worst:.3g} away from 1")
    pd.DataFrame(rows).to_csv(results_dir / "patching.csv", index=False)
    results.write_json(
        results_dir / "patching.json",
        {
            "settings": {
                "n_boot": N_BOOT,
                "seed": SEED,
                "quantile": QUANTILE,
                "batch_size": BATCH_SIZE,
            },
            "bar": bar,
            "n_bios": len(texts),
            "n_eligible": int(eligible.sum()),
            "n_unequal_length": int((~same).sum()),
            "n_unequal_length_eligible": int((eligible & ~same).sum()),
            "n_patched": len(keep),
            "max_all_positions_error": worst,
            "seconds": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
