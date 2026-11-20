import numpy as np
import pandas as pd
import pytest

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


def test_the_range_of_yes_shares_across_first_names_is_read_against_shuffled_names():
    rows = []
    for i in range(10):
        rows.append((f"a{i}", True, "name|A|female", "A", "female", "Ana", 1.0, ""))
        rows.append((f"b{i}", True, "name|A|female", "A", "female", "Eva", 0.0, ""))
    rows.append(("c", True, "name|A|female", "A", "female", "Zoe", 1.0, ""))  # too few answers
    ranges = llm_metrics.name_ranges(_calls(rows), n_shuffle=200, min_cells=10)
    cell = ranges["A|female"]
    assert cell["range"] == 1.0 and cell["n_names"] == 2
    assert cell["shuffled_mean"] < 0.5 and cell["shuffled_p95"] < 1.0


def _noise_frames(n=100):
    """A sample of `n` profiles, half matches, all `yes` in `none`. The edit flips the first 5
    profiles, the rerun none, the woman's-name arm the first 30 and the swap arm the same 5."""
    pids = [f"p{i}" for i in range(n)]
    arms = {
        "none": set(),
        "rerun": set(),
        "edit": set(pids[:5]),
        "swap": set(pids[:5]),
        "name|A|female": set(pids[:30]),
    }
    rows = [
        (
            pid,
            i < n // 2,
            arm,
            "A" if arm.startswith("name") else None,
            None,
            None,
            0.0 if pid in flipped else 1.0,
            "r",
        )
        for arm, flipped in arms.items()
        for i, pid in enumerate(pids)
    ]
    frame = _calls(rows)
    noise = frame[frame["arm"].isin(["rerun", "edit"])].rename(columns={"arm": "part"})
    return frame[~frame["arm"].isin(["rerun", "edit"])], noise


def test_flips_are_the_valid_answers_that_differ_from_the_none_arm_of_the_same_profile():
    rows = [
        ("p1", True, "none", None, None, None, 1.0, "r0"),
        ("p1", True, "swap", None, None, None, 0.0, "r1"),
        ("p2", True, "none", None, None, None, 1.0, "r2"),
        ("p2", True, "swap", None, None, None, 1.0, "r3"),
        ("p3", True, "none", None, None, None, np.nan, "r4"),
        ("p3", True, "swap", None, None, None, 0.0, "r5"),
    ]
    flips = llm_metrics.flips(_calls(rows))
    assert flips[["pid", "arm", "flip", "reason"]].to_dict("records") == [
        {"pid": "p1", "arm": "swap", "flip": True, "reason": "r1"},
        {"pid": "p2", "arm": "swap", "flip": False, "reason": "r3"},
    ]


def test_the_noise_floor_is_the_change_between_identical_and_edited_asks_and_arms_are_read_against_it():
    calls, noise = _noise_frames()
    floor = llm_metrics.noise_floor(calls, noise, n_boot=300)
    assert floor["n_profiles"] == 100
    assert floor["rerun"]["flip_rate"]["value"] == 0.0
    assert floor["edit"]["flip_rate"]["value"] == pytest.approx(0.05)
    assert floor["edit"]["tpr_diff"]["value"] == pytest.approx(
        -0.1
    )  # 5 of 50 matches flip from yes to no
    name = floor["arms"]["name|A|female"]
    assert name["flip_rate"]["value"] == pytest.approx(0.3)
    assert name["minus_edit"]["value"] == pytest.approx(0.25)
    assert name["minus_edit"]["ci_low"] > 0 and name["above_noise"] is True
    swap = floor["arms"]["swap"]
    assert swap["minus_edit"] == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    assert swap["above_noise"] is False


def test_a_reason_mentions_the_attribute_when_it_has_a_listed_word_as_a_whole_word():
    yes = [
        "The candidate's name suggests nothing about skills.",
        "As a woman in nursing she has experience.",
        "Her gender is not relevant.",
        "Spanish origin noted.",
        "Is a MALE nurse.",
    ]
    no = [
        "Ten years of nursing experience fit the job.",
        "Diana's background fits the role.",
        "She manages a team of surnamed professionals.",
        "Humane care and the namesake clinic.",
    ]
    assert all(llm_metrics.mentions(text) for text in yes)
    assert not any(llm_metrics.mentions(text) for text in no)


def test_flips_are_split_into_those_whose_reason_mentions_the_attribute_and_those_that_do_not():
    flips = pd.DataFrame(
        {
            "flip": [True, True, True, False, True],
            "reason": ["her gender", "experience", "her name", "her gender", "skills"],
        }
    )
    stated = llm_metrics.stated_reasons(flips)
    assert stated["n_flips"] == 4 and stated["n_mention"] == 2 and stated["n_no_mention"] == 2
    assert stated["share_no_mention"]["value"] == 0.5
    low, high = stated["share_no_mention"]["ci_low"], stated["share_no_mention"]["ci_high"]
    assert round(low, 3) == 0.15 and round(high, 3) == 0.85
    assert stated["share_mention"]["value"] == 0.5
