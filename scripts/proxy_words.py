"""The proxy-word list: non-gendered words that push each showcase gap the way pronouns do.

Applies the rule of ADR 0007 (`proxy_words.RULE`) to the main model's rows of
`results/gap_decomposition.csv` and the gendered sums of `results/gap_explained.json`, and writes
the words per occupation and their union to `results/proxy_words.json`.
"""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai import proxy_words, results
from hiring_bias_xai.config import RESULTS_DIR

SCREENER = "deberta"
COLUMNS = [
    "screener",
    "occupation",
    "word",
    "contribution",
    "contribution_ci_low",
    "contribution_ci_high",
    "gendered",
    "freq_female",
    "freq_male",
]


def main(results_dir: Path = RESULTS_DIR) -> None:
    # Words such as "nan" and "null" must stay words, not become missing values.
    table = pd.read_csv(
        results_dir / "gap_decomposition.csv",
        usecols=COLUMNS,
        dtype={"word": str},
        keep_default_na=False,
    )
    table = table[table["screener"] == SCREENER]
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    occupations = [
        row["occupation"]
        for side in ("male_dominated", "female_dominated")
        for row in showcase[side]
    ]
    explained = json.loads((results_dir / "gap_explained.json").read_text())["screeners"][SCREENER]
    sums = {o: explained[o]["gendered_sum"]["value"] for o in occupations}
    chosen = {
        o: proxy_words.proxy_rows(table[table["occupation"] == o], sums[o]) for o in occupations
    }
    results.write_json(
        results_dir / "proxy_words.json",
        {
            "rule": proxy_words.RULE,
            "screener": SCREENER,
            "top_per_occupation": proxy_words.TOP,
            "usage": {
                "ratio": proxy_words.RATIO,
                "smoothing": proxy_words.SMOOTHING,
                "min_share": proxy_words.MIN_SHARE,
            },
            "gendered_sums": sums,
            "occupations": {
                o: [
                    {
                        "word": row.word,
                        "contribution": row.contribution,
                        "ci_low": row.contribution_ci_low,
                        "ci_high": row.contribution_ci_high,
                        "freq_female": row.freq_female,
                        "freq_male": row.freq_male,
                    }
                    for row in rows.itertuples()
                ]
                for o, rows in chosen.items()
            },
            "occupations_without_words": sorted(o for o, rows in chosen.items() if rows.empty),
            "words": sorted({word for rows in chosen.values() for word in rows["word"]}),
        },
    )


if __name__ == "__main__":
    main()
