import pandas as pd

from hiring_bias_xai import llm_profiles

GROUPS = {
    "Spain": {
        "first_names": {"female": ["Maria", "Carmen"], "male": ["Antonio", "Manuel"]},
        "surnames": ["Garcia", "Lopez"],
    },
    "Romania": {
        "first_names": {"female": ["Elena", "Ana"], "male": ["Ion", "Mihai"]},
        "surnames": ["Popescu", "Radu"],
    },
}


def test_a_name_is_drawn_from_the_group_and_sex_and_stays_the_same_for_the_same_bio_and_arm():
    first, surname = llm_profiles.draw_name(GROUPS, "Romania", "female", "test-1")
    assert first in GROUPS["Romania"]["first_names"]["female"]
    assert surname in GROUPS["Romania"]["surnames"]
    assert llm_profiles.draw_name(GROUPS, "Romania", "female", "test-1") == (first, surname)
    drawn = {llm_profiles.draw_name(GROUPS, "Spain", "male", f"test-{i}") for i in range(60)}
    assert len(drawn) == 4  # 2 first names x 2 surnames, all of them turn up


def test_the_none_and_neutral_arms_leave_the_bio_alone_and_the_neutral_one_adds_an_id():
    bio = "She is a nurse. Her work is hard."
    none = llm_profiles.candidate(bio, "female", "test-1234", "none", GROUPS)
    assert none == (None, bio)
    neutral = llm_profiles.candidate(bio, "female", "test-1234", "neutral", GROUPS)
    assert neutral == ("Candidate ID: 1234", bio)


def test_the_swap_arm_swaps_the_pronouns_and_adds_no_header():
    header, text = llm_profiles.candidate("She is a nurse.", "female", "test-1", "swap", GROUPS)
    assert (header, text) == (None, "He is a nurse.")


def test_a_name_arm_sets_the_pronouns_to_the_gender_of_the_name():
    bio = "She is a nurse. Her work is hard."
    header, text = llm_profiles.candidate(bio, "female", "test-9", "name|Spain|female", GROUPS)
    assert header.startswith("Name: ") and header.split()[1] in ("Maria", "Carmen")
    assert text == bio
    header, text = llm_profiles.candidate(bio, "female", "test-9", "name|Spain|male", GROUPS)
    assert header.split()[1] in ("Antonio", "Manuel")
    assert text == "He is a nurse. His work is hard."
    header, text = llm_profiles.candidate(
        "He is a nurse.", "male", "test-9", "name|Romania|female", GROUPS
    )
    assert header.split()[1] in ("Elena", "Ana") and text == "She is a nurse."


def test_the_hardest_confusion_of_each_occupation_is_the_other_one_it_is_most_often_taken_for():
    true = ["a"] * 6 + ["b"] * 4 + ["c"] * 3 + ["z"] * 5
    predicted = ["a", "b", "b", "c", "z", "z"] + ["b", "b", "b", "a"] + ["a", "a", "c"] + ["a"] * 5
    hardest = llm_profiles.hardest_confusions(true, predicted, ["a", "b", "c"])
    assert hardest == {"a": "b", "b": "a", "c": "a"}  # z is outside the list and is ignored


def _subset():
    groups = [(o, g) for o in "abc" for g in ("female", "male") for _ in range(10)]
    return pd.DataFrame(
        [(f"test-{i}", f"bio {o} {g}", o, g) for i, (o, g) in enumerate(groups)],
        columns=["id", "text", "occupation", "gender"],
    )


def test_profiles_hold_matches_hard_and_other_non_matches_half_women_and_half_men():
    hardest = {"a": "b", "b": "a", "c": "a"}
    profiles = llm_profiles.sample_profiles(_subset(), hardest, n_match=4, n_nonmatch=4)
    assert set(profiles.columns) >= {"job", "id", "text", "occupation", "gender", "match", "hard"}
    for job, rows in profiles.groupby("job"):
        assert len(rows) == 8 and rows["id"].is_unique
        matches = rows[rows["match"]]
        assert (matches["occupation"] == job).all()
        assert matches["gender"].value_counts().to_dict() == {"female": 2, "male": 2}
        hard = rows[~rows["match"] & rows["hard"]]
        assert (hard["occupation"] == hardest[job]).all()
        assert hard["gender"].value_counts().to_dict() == {"female": 1, "male": 1}
        other = rows[~rows["match"] & ~rows["hard"]]
        assert not other["occupation"].isin([job, hardest[job]]).any()
        assert other["gender"].value_counts().to_dict() == {"female": 1, "male": 1}
    again = llm_profiles.sample_profiles(_subset(), hardest, n_match=4, n_nonmatch=4)
    pd.testing.assert_frame_equal(profiles, again)
    other_seed = llm_profiles.sample_profiles(_subset(), hardest, 4, 4, seed=7)
    assert not profiles["id"].equals(other_seed["id"])


def test_merged_groups_pool_their_names_and_dropped_groups_are_left_out():
    groups = GROUPS | {
        "Morocco": {"first_names": {"female": ["Fatima"], "male": ["Ali"]}, "surnames": ["Benali"]},
        "Hispanic America": {
            "first_names": {"female": ["Maria", "Luz"], "male": ["Juan"]},
            "surnames": ["Vera"],
        },
    }
    decision = {"groups": [["Spain", "Hispanic America"], ["Romania"]], "dropped": ["Morocco"]}
    pools = llm_profiles.analysis_groups(groups, decision)
    assert list(pools) == ["Spain + Hispanic America", "Romania"]
    assert pools["Spain + Hispanic America"] == {
        "first_names": {
            "female": ["Maria", "Carmen", "Luz"],
            "male": ["Antonio", "Manuel", "Juan"],
        },
        "surnames": ["Garcia", "Lopez", "Vera"],
    }
    assert pools["Romania"] == groups["Romania"]
    assert llm_profiles.arm_labels(pools) == [
        "none",
        "neutral",
        "swap",
        "name|Spain + Hispanic America|female",
        "name|Spain + Hispanic America|male",
        "name|Romania|female",
        "name|Romania|male",
    ]
