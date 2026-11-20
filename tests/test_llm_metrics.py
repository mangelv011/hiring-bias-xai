import numpy as np
import pandas as pd

from hiring_bias_xai import llm_metrics

COLUMNS = ["pid", "match", "arm", "group", "name_gender", "first", "yes", "reason"]


def _calls(rows):
    """Calls from (pid, match, arm, group, name_gender, first, yes, reason); `kind` is derived."""
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["kind"] = frame["arm"].str.split("|").str[0]
    return frame


def test_tpr_is_the_yes_share_on_matches_and_fpr_on_non_matches_and_invalid_answers_are_apart():
    rows = [
        ("p1", True, "none", None, None, None, 1.0, ""),
        ("p2", True, "none", None, None, None, 0.0, ""),
        ("p3", True, "none", None, None, None, np.nan, ""),
        ("p4", False, "none", None, None, None, 1.0, ""),
        ("p5", False, "none", None, None, None, 0.0, ""),
        ("p6", False, "none", None, None, None, 0.0, ""),
        ("p1", True, "swap", None, None, None, 1.0, ""),
        ("p4", False, "swap", None, None, None, 0.0, ""),
    ]
    rates = llm_metrics.arm_rates(_calls(rows))
    assert rates["none"] == {
        "tpr": 0.5,
        "fpr": 1 / 3,
        "n_match": 2,
        "n_nonmatch": 3,
        "n_invalid": 1,
    }
    assert rates["swap"]["tpr"] == 1.0 and rates["swap"]["fpr"] == 0.0
