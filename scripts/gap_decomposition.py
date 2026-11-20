"""Gap decomposition: which words make up the women-minus-men score gap of each showcase occupation.

Reads the saved Shapley values of both screeners on the showcase subset (target = true occupation)
and splits, per screener and occupation, the gap between the mean score of women's and men's bios
into one contribution per word (`gap_decomposition.decompose`). Writes
`results/gap_decomposition.csv` and `results/gap_explained.json`.
"""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai import attribution, data, gap_decomposition
from hiring_bias_xai.config import RESULTS_DIR

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
        row["occupation"] for side in ("male_dominated", "female_dominated") for row in showcase[side]
    ]
    tables = []
    for screener in SCREENERS:
        for occupation in occupations:
            bios = _bios(screener, subset[subset["occupation"] == occupation])
            decomposition = gap_decomposition.decompose(bios, n_boot=N_BOOT)
            table = decomposition.table
            table.insert(0, "occupation", occupation)
            table.insert(0, "screener", screener)
            tables.append(table)
    pd.concat(tables).to_csv(results_dir / "gap_decomposition.csv", index=False)


if __name__ == "__main__":
    main()
