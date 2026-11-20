"""Proxy words: the non-gendered words that push a showcase occupation's score gap like pronouns."""

import numpy as np
import pandas as pd

TOP = 5
RATIO = 3
SMOOTHING = 0.01
MIN_SHARE = 0.05
RULE = (
    "For each showcase occupation, in the rows of the main model of results/gap_decomposition.csv: "
    "keep the words that are not gendered words; whose 95% interval of the gap contribution "
    "excludes 0 and whose contribution has the sign of the gendered sum of that occupation; "
    "that the gender the gap favours (women if the sum is positive, men if negative) uses at "
    "least 3 times as often as the other gender, as (share of bios + 0.01) over (share of the "
    "other + 0.01); and that at least 5% of the bios of that gender contain. Take the 5 with "
    "the largest absolute contribution (ties in alphabetical order). The mask list is the union "
    "over the occupations; an occupation may add no word."
)


def proxy_rows(table: pd.DataFrame, gendered_sum: float, top: int = TOP) -> pd.DataFrame:
    """The rows of `table` (one occupation of the main model) that are proxy words.

    `table` has the columns of `results/gap_decomposition.csv`. A proxy word is not gendered; its
    gap contribution has a 95% interval that excludes 0 and the sign of `gendered_sum`; it is
    used much more by the gender that sum favours (women if positive) than by the other, and by
    at least `MIN_SHARE` of that gender's bios. The `top` of them with the largest absolute
    contribution are returned, largest first, ties in alphabetical order.
    """
    excludes_zero = (table["contribution_ci_low"] > 0) | (table["contribution_ci_high"] < 0)
    pushes_the_same_way = np.sign(table["contribution"]) == np.sign(gendered_sum)
    women_favoured = gendered_sum > 0
    favoured = table["freq_female" if women_favoured else "freq_male"]
    other = table["freq_male" if women_favoured else "freq_female"]
    used_unequally = (favoured + SMOOTHING >= RATIO * (other + SMOOTHING)) & (favoured >= MIN_SHARE)
    kept = table[~table["gendered"] & excludes_zero & pushes_the_same_way & used_unequally]
    size = kept["contribution"].abs()
    return (
        kept.assign(size=size)
        .sort_values(["size", "word"], ascending=[False, True])
        .head(top)
        .drop(columns="size")
    )
