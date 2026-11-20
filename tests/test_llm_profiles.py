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
