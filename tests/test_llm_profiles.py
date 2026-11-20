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
