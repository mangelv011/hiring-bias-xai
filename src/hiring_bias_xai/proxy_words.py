"""Proxy words: the non-gendered words that push a showcase occupation's score gap like pronouns."""

import numpy as np
import pandas as pd

TOP = 5
RULE = (
    "For each showcase occupation, in the rows of the main model of results/gap_decomposition.csv: "
    "keep the words that are not gendered words, whose 95% interval of the gap contribution "
    "excludes 0, and whose contribution has the sign of the gendered sum of that occupation; "
    "take the 5 with the largest absolute contribution (ties in alphabetical order). The mask "
    "list is the union over the occupations."
)


def proxy_rows(table: pd.DataFrame, gendered_sum: float, top: int = TOP) -> pd.DataFrame:
    """The rows of `table` (one occupation of the main model) that are proxy words.

    `table` has the columns of `results/gap_decomposition.csv`. A proxy word is not gendered, has
    a gap contribution whose 95% interval excludes 0, and has the sign of `gendered_sum`. The `top`
    of them with the largest absolute contribution are returned, largest first, ties in
    alphabetical order.
    """
    excludes_zero = (table["contribution_ci_low"] > 0) | (table["contribution_ci_high"] < 0)
    pushes_the_same_way = np.sign(table["contribution"]) == np.sign(gendered_sum)
    kept = table[~table["gendered"] & excludes_zero & pushes_the_same_way]
    size = kept["contribution"].abs()
    return (
        kept.assign(size=size)
        .sort_values(["size", "word"], ascending=[False, True])
        .head(top)
        .drop(columns="size")
    )
