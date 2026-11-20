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


def _scenario(yes, n_match=4, n_nonmatch=2, groups=("A", "United Kingdom")):
    """Calls for every arm. `yes(arm, pid_number)` gives 1 or 0."""
    arms = ["none", "neutral", "swap"] + [
        f"name|{g}|{s}" for g in groups for s in ("female", "male")
    ]
    rows = []
    for number in range(n_match + n_nonmatch):
        for arm in arms:
            kind, _, rest = arm.partition("|")
            group, _, gender = rest.partition("|")
            first = {"female": "Ana", "male": "Juan"}.get(gender)
            rows.append(
                (
                    f"p{number}",
                    number < n_match,
                    arm,
                    group or None,
                    gender or None,
                    first,
                    float(yes(arm, number)),
                    "",
                )
            )
    return _calls(rows)


def _one_male_a_name_says_no(arm, number):
    return 0 if (arm == "name|A|male" and number == 0) else 1


def test_the_contrasts_are_differences_of_yes_shares_on_matches_paired_by_bio_and_group():
    stats = llm_metrics.contrasts(_scenario(_one_male_a_name_says_no), n_boot=200)["statistics"]
    assert stats["primary"]["value"] == 0.125  # (1 + 0 + 0 + 0 + 4 x 0) / 8 pairs
    assert stats["diff_tpr|name|A|female"]["value"] == 0.0
    assert stats["diff_tpr|name|A|male"]["value"] == -0.25
    assert stats["vs_uk|A"]["value"] == -0.125  # (4 + 3) / 8 against 8 / 8
    assert stats["tpr|none"]["value"] == 1.0
    assert stats["diff_fpr|swap"]["value"] == 0.0
    for stat in stats.values():
        assert stat["ci_low"] <= stat["value"] <= stat["ci_high"] or np.isnan(stat["value"])


def test_when_every_answer_is_the_same_the_intervals_have_no_width_and_p_is_one():
    stats = llm_metrics.contrasts(_scenario(lambda arm, n: 1), n_boot=100)["statistics"]
    for name in ("primary", "vs_uk|A", "diff_tpr|swap", "diff_tpr|neutral"):
        assert stats[name]["value"] == stats[name]["ci_low"] == stats[name]["ci_high"] == 0.0
        assert stats[name]["p"] == 1.0


def test_a_sure_effect_gets_a_small_p_and_holm_adjusts_only_the_family_of_other_contrasts():
    def yes(arm, number):
        return 0 if arm == "name|A|male" else 1

    report = llm_metrics.contrasts(_scenario(yes, n_match=40, n_nonmatch=0), n_boot=200)
    stats = report["statistics"]
    assert stats["primary"]["value"] == 0.5
    assert stats["primary"]["p"] == 2 / 201  # no resample reaches zero
    family = ["vs_uk|A", "diff_tpr|swap", "diff_tpr|neutral"]
    assert sorted(report["holm"]) == sorted(family)
    assert all(stats[name]["p"] <= report["holm"][name] <= 1 for name in family)
    assert report["holm"]["vs_uk|A"] == 6 / 201  # the smallest p is multiplied by the family size
