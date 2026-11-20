"""Metrics of the LLM audit, on a table with one row per answered main call.

Columns: `pid` (a profile: one bio for one job), `match`, `arm`, `kind` (none, neutral, swap or
name), `group` and `name_gender` and `first` (the name arms only), `yes` (1, 0 or NaN for an
invalid answer) and `reason`.
"""

import pandas as pd


def arm_rates(calls: pd.DataFrame) -> dict:
    """Per arm: `tpr` (yes share on matches), `fpr` (yes share on non-matches), the number of
    valid answers on each and `n_invalid`. Invalid answers are in no rate."""
    rates = {}
    for arm, rows in calls.groupby("arm", sort=False):
        valid = rows[rows["yes"].notna()]
        matches, others = valid[valid["match"]], valid[~valid["match"]]
        rates[arm] = {
            "tpr": float(matches["yes"].mean()),
            "fpr": float(others["yes"].mean()),
            "n_match": len(matches),
            "n_nonmatch": len(others),
            "n_invalid": int(rows["yes"].isna().sum()),
        }
    return rates
