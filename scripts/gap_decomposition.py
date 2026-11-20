"""Gap decomposition: which words make up the women-minus-men score gap of each showcase occupation.

Reads the saved Shapley values of both screeners on the showcase subset (target = true occupation)
and splits, per screener and occupation, the gap between the mean score of women's and men's bios
into one contribution per word (`gap_decomposition.decompose`). Writes
`results/gap_decomposition.csv` and `results/gap_explained.json`.
"""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai import attribution, data, gap_decomposition, results
from hiring_bias_xai.config import RESULTS_DIR, SEED

SCREENERS = ("deberta", "baseline")
N_BOOT = gap_decomposition.N_BOOT


def _bios(screener: str, subset: pd.DataFrame) -> list[gap_decomposition.Bio]:
    return [
        gap_decomposition.bio_from_attribution(
            attribution.load(attribution.path(screener, row.id)), row.gender
        )
        for row in subset.itertuples()
    ]


def main(results_dir: Path = RESULTS_DIR) -> None:
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    occupations = [
        row["occupation"]
        for side in ("male_dominated", "female_dominated")
        for row in showcase[side]
    ]
    tables, summaries = [], {}
    for screener in SCREENERS:
        summaries[screener] = {}
        for occupation in occupations:
            bios = _bios(screener, subset[subset["occupation"] == occupation])
            decomposition = gap_decomposition.decompose(bios, n_boot=N_BOOT)
            table = decomposition.table
            table.insert(0, "occupation", occupation)
            table.insert(0, "screener", screener)
            tables.append(table)
            summaries[screener][occupation] = decomposition.summary
            if decomposition.summary["gap_unstable"]:
                gap = decomposition.summary["decomposed_gap"]
                print(
                    f"{screener} {occupation}: the total gap {gap['value']:+.3f} has an interval "
                    f"[{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}] that includes 0, so its "
                    "shares are unstable"
                )
    pd.concat(tables).to_csv(results_dir / "gap_decomposition.csv", index=False)
    results.write_json(
        results_dir / "gap_explained.json",
        {
            "settings": {
                "n_boot": N_BOOT,
                "min_bios": gap_decomposition.MIN_BIOS,
                "top": gap_decomposition.TOP,
                "seed": SEED,
            },
            "max_decomposition_error": max(
                s["decomposition_error"]
                for by_occupation in summaries.values()
                for s in by_occupation.values()
            ),
            "screeners": summaries,
        },
    )


if __name__ == "__main__":
    main()
