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
