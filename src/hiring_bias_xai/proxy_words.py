"""Proxy words: the non-gendered words that push a showcase occupation's score gap like pronouns."""

import numpy as np
import pandas as pd


def proxy_rows(table: pd.DataFrame, gendered_sum: float, top: int = 5) -> pd.DataFrame:
    """The rows of `table` (one occupation of the main model) that are proxy words.

    `table` has the columns of `results/gap_decomposition.csv`; a proxy word is not gendered and
    has a gap contribution whose 95% interval excludes 0 and whose sign is that of `gendered_sum`.
    """
    excludes_zero = (table["contribution_ci_low"] > 0) | (table["contribution_ci_high"] < 0)
    pushes_the_same_way = np.sign(table["contribution"]) == np.sign(gendered_sum)
    return table[~table["gendered"] & excludes_zero & pushes_the_same_way]
