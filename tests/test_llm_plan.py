import pandas as pd

from hiring_bias_xai import llm_plan

POOLS = {
    "A": {
        "first_names": {"female": ["Maria"], "male": ["Juan"]},
        "surnames": ["Garcia"],
    },
    "United Kingdom": {
        "first_names": {"female": ["Susan"], "male": ["John"]},
        "surnames": ["Smith"],
    },
}


def _profiles():
    rows = [
        ("nurse", "test-1", "She is a nurse. Her work is good.", "nurse", "female", True, False),
        ("nurse", "test-2", "He is a surgeon. He cuts.", "surgeon", "male", False, True),
        ("surgeon", "test-1", "She is a nurse. Her work is good.", "nurse", "female", False, True),
    ]
    columns = ["job", "id", "text", "occupation", "gender", "match", "hard"]
    return pd.DataFrame(rows, columns=columns)


def test_the_plan_holds_every_arm_the_paraphrases_and_the_noise_asks_of_each_profile():
    plan = llm_plan.build(_profiles(), POOLS, noise_pids=["nurse|test-1", "surgeon|test-1"])
    parts = plan["part"].value_counts().to_dict()
    assert parts == {"main": 3 * 7, "paraphrase": 3 * 2, "noise": 2 * 2}
    one = plan[(plan["pid"] == "nurse|test-2") & (plan["part"] == "main")]
    assert sorted(one["arm"]) == sorted(
        ["none", "neutral", "swap"] + [f"name|{g}|{s}" for g in POOLS for s in ("female", "male")]
    )
    assert set(one["version"]) == {"v1"} and set(one["run"]) == {0}


def test_plan_rows_carry_the_attributes_the_analysis_needs():
    plan = llm_plan.build(_profiles(), POOLS, noise_pids=[])
    row = plan[(plan["pid"] == "nurse|test-1") & (plan["arm"] == "name|A|male")].iloc[0]
    assert (row["job"], row["bio_id"], row["match"], row["hard"]) == (
        "nurse",
        "test-1",
        True,
        False,
    )
    assert (row["kind"], row["group"], row["name_gender"], row["first"]) == (
        "name",
        "A",
        "male",
        "Juan",
    )
    assert "Name: Juan Garcia\nHe is a nurse. His work is good." in row["messages"][0]["content"]
    none = plan[(plan["pid"] == "nurse|test-1") & (plan["arm"] == "none")].iloc[0]
    assert none["kind"] == "none" and pd.isna(none["group"]) and pd.isna(none["first"])


def test_the_noise_asks_repeat_the_none_request_with_a_run_index_and_with_a_whitespace_edit():
    plan = llm_plan.build(_profiles(), POOLS, noise_pids=["nurse|test-1"])
    noise = plan[plan["part"] == "noise"].set_index("noise")
    base = plan[
        (plan["part"] == "main") & (plan["pid"] == "nurse|test-1") & (plan["arm"] == "none")
    ].iloc[0]
    assert noise.loc["rerun", "messages"] == base["messages"] and noise.loc["rerun", "run"] == 1
    assert noise.loc["edit", "run"] == 0
    edited = noise.loc["edit", "messages"][0]["content"]
    assert (
        "nurse.  Her work is good.  " not in edited and "is a nurse.  Her work is good." in edited
    )
    assert edited != base["messages"][0]["content"]


def test_the_paraphrase_rows_use_the_none_arm_with_the_other_two_versions():
    plan = llm_plan.build(_profiles(), POOLS, noise_pids=[])
    para = plan[plan["part"] == "paraphrase"]
    assert set(para["arm"]) == {"none"} and set(para["version"]) == {"v2", "v3"}
