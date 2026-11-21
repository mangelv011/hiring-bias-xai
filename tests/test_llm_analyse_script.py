import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeLLMServer
from test_llm_run_script import SIZES, _load_script, setup  # noqa: F401  (fixture)

from hiring_bias_xai import data, llm_audit
from hiring_bias_xai.config import SEED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_analyse.py"


def _load_analysis():
    spec = importlib.util.spec_from_file_location("llm_analyse", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _answer(payload):
    """Says no to a man's name header, and says "no" with a reason about experience; says yes
    to everyone else, with a reason that mentions the gender for the neutral header."""
    text = payload["messages"][0]["content"]
    if "Name: Juan" in text or "Name: Luis" in text:
        return {"content": '{"reason": "not enough experience", "decision": "no"}'}
    return {"content": '{"reason": "fits the job", "decision": "yes"}'}


@pytest.fixture
def analysed(setup, tmp_path):  # noqa: F811
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(setup / "llm" / "main.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client, max_minutes=60, **SIZES)
        _load_analysis().main(setup, client=client, n_boot=50)
    return setup, json.loads((setup / "llm_audit.json").read_text())


def test_the_report_holds_the_arms_the_primary_contrast_and_the_run_cost(analysed):
    _, report = analysed
    assert report["arms"]["none"]["tpr"] == 1.0
    assert report["arms"]["name|Spain + Romania|male"]["tpr"] == 0.0
    assert report["arms"]["name|Spain + Romania|female"]["tpr"] == 1.0
    stats = report["statistics"]
    assert stats["primary"]["value"] == 1.0
    assert stats["diff_tpr|name|Spain + Romania|male"]["value"] == -1.0
    cost = report["run_cost"]
    assert cost["n_calls"] == 174 and cost["seconds_total"] > 0
    assert cost["n_invalid"] == 0 and cost["thinking_leaks"] == 0
    assert report["model"] == "m"
    assert report["profiles"]["n_profiles"] == 24
    assert report["n_noise_profiles"] == 3


def test_flips_are_counted_against_the_noise_floor_and_split_by_their_reason(analysed):
    results_dir, report = analysed
    male = report["noise_floor"]["arms"]["name|Spain + Romania|male"]
    assert male["flip_rate"]["value"] == 1.0 and male["above_noise"] is True
    assert report["noise_floor"]["arms"]["name|Spain + Romania|female"]["above_noise"] is False
    assert report["noise_floor"]["edit"]["flip_rate"]["value"] == 0.0
    stated = report["rationalization"]["beyond_noise"]
    assert stated["n_flips"] == 24 and stated["n_no_mention"] == 24
    assert stated["share_no_mention"]["value"] == 1.0
    assert report["rationalization"]["name_arms_above_noise"] == 1


def test_the_hand_check_sample_is_written_for_labelling_and_scored_once_labelled(analysed):
    results_dir, report = analysed
    path = results_dir / "llm" / "rationalization_hand_check.csv"
    sample = pd.read_csv(path)
    assert len(sample) == 24 and sample["hand_label"].isna().all()
    assert report["rationalization"]["hand_check"] is None
    sample["hand_label"] = [0] * 20 + [1] * 4
    sample.to_csv(path, index=False)
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(results_dir / "llm" / "main.jsonl", server.base_url, "m")
        _load_analysis().main(results_dir, client=client, n_boot=50)
    check = json.loads((results_dir / "llm_audit.json").read_text())
    summary = check["rationalization"]["hand_check"]
    assert summary["n"] == 24 and summary["n_rule_agrees"] == 20
    assert summary["agreement"]["value"] == pytest.approx(20 / 24)


def test_the_report_splits_the_none_arm_false_positives_by_hardness(analysed):
    _, report = analysed
    hardness = report["fpr_by_hardness"]["none"]
    assert hardness["hard"]["n"] == 6 and hardness["other"]["n"] == 6
    assert hardness["hard"]["value"] == 1.0


WORK = re.compile(r"work is (\d+)")


def _analyse_with(results_dir, loud):
    """Run the plan and the analysis against a server that answers by the prompt.

    Always: the female-name arm says no to the first bio outside the noise sample (a few flips,
    too few to be beyond the noise). With `loud`: the other paraphrase says no to everyone, the
    third one gives no JSON for even bio numbers, the neutral arm says no with a reasoning text,
    and the pronoun swap says no with a reason that names a man.
    """
    state = {}
    yes = {"content": '{"reason": "fits the job", "decision": "yes"}'}

    def answer(payload):
        text = payload["messages"][0]["content"]
        if "target" not in state:
            profiles = pd.read_csv(results_dir / "llm" / "profiles.csv")
            state["target"] = WORK.search(profiles[~profiles["noise_sample"]].iloc[0]["text"])[1]
        number = WORK.search(text)[1]
        if loud and "Your task is to shortlist" in text:
            return {"content": '{"reason": "weak fit", "decision": "no"}'}
        if loud and "A recruiter needs help" in text:
            return {"content": "maybe"} if int(number) % 2 == 0 else yes
        if loud and "Candidate ID" in text:
            return {
                "content": '{"reason": "weak fit", "decision": "no"}',
                "reasoning_content": "hm",
            }
        if loud and "Name: " not in text and "He is a" in text:
            return {"content": '{"reason": "the candidate is a man", "decision": "no"}'}
        if ("Name: Maria" in text or "Name: Ana" in text) and number == state["target"]:
            return {"content": '{"reason": "weak fit", "decision": "no"}'}
        return yes

    with FakeLLMServer(answer) as server:
        client = llm_audit.Client(results_dir / "llm" / "main.jsonl", server.base_url, "m")
        _load_script().main(results_dir, client=client, max_minutes=60, **SIZES)
        _load_analysis().main(results_dir, client=client, n_boot=50)
    profiles = pd.read_csv(results_dir / "llm" / "profiles.csv")
    numbers = profiles["text"].str.extract(WORK.pattern)[0]
    return {
        "report": json.loads((results_dir / "llm_audit.json").read_text()),
        "n_target": int((numbers == state["target"]).sum()),
        "n_even": int((numbers.astype(int) % 2 == 0).sum()),
    }


@pytest.fixture
def loud(setup):  # noqa: F811
    return setup, _analyse_with(setup, loud=True)


@pytest.fixture
def quiet(setup):  # noqa: F811
    return setup, _analyse_with(setup, loud=False)


def test_flips_of_the_swap_and_name_arms_are_the_ones_beyond_the_noise_and_all_of_them(loud):
    _, run = loud
    rationalization = run["report"]["rationalization"]
    # the swap arm says no to all 24 profiles; the woman's name arm flips on the target bio only
    assert rationalization["beyond_noise"]["n_flips"] == 24
    assert rationalization["all_name_and_swap_flips"]["n_flips"] == 24 + run["n_target"]
    assert rationalization["name_arms_above_noise"] == 0  # the neutral and the swap arm only
    assert rationalization["beyond_noise"]["n_mention"] == 24  # "the candidate is a man"
    assert rationalization["beyond_noise"]["share_no_mention"]["value"] == 0.0
    assert rationalization["hand_check_pool"] == "beyond_noise"


def test_the_hand_check_sample_is_drawn_from_the_flips_beyond_the_noise_and_scored_by_label(loud):
    results_dir, _ = loud
    path = results_dir / "llm" / "rationalization_hand_check.csv"
    sample = pd.read_csv(path)
    assert len(sample) == 24 and set(sample["arm"]) == {"swap"} and sample["rule_mention"].all()
    sample["hand_label"] = [1] * 6 + [0] * 4 + [None] * 14  # 10 reasons labelled
    sample.to_csv(path, index=False)
    with FakeLLMServer(lambda payload: pytest.fail("no call is asked again")) as server:
        client = llm_audit.Client(results_dir / "llm" / "main.jsonl", server.base_url, "m")
        _load_analysis().main(results_dir, client=client, n_boot=50)
    check = json.loads((results_dir / "llm_audit.json").read_text())["rationalization"]
    summary = check["hand_check"]
    assert (summary["n"], summary["n_rule_agrees"]) == (10, 6)
    assert summary["n_hand_mention_rule_misses"] == 0
    assert summary["n_rule_mention_hand_not"] == 4
    assert summary["share_no_mention_by_hand"]["value"] == pytest.approx(0.4)


def test_the_run_cost_counts_thinking_leaks_and_invalid_answers_and_the_settings_are_kept(loud):
    _, run = loud
    report = run["report"]
    assert report["run_cost"]["thinking_leaks"] == 24  # the neutral arm
    assert report["run_cost"]["n_invalid"] == run["n_even"]  # the third paraphrase
    assert report["settings"] == llm_audit.SETTINGS


def test_prompt_sensitivity_gives_the_rates_of_each_paraphrase_and_its_flips_against_v1(loud):
    _, run = loud
    v2, v3 = run["report"]["prompt_sensitivity"]["v2"], run["report"]["prompt_sensitivity"]["v3"]
    assert (v2["tpr"], v2["fpr"], v2["n_invalid"], v2["flip_rate_against_v1"]) == (0.0, 0.0, 0, 1.0)
    assert (v2["n_match"], v2["n_nonmatch"]) == (12, 12)
    assert (v3["tpr"], v3["fpr"], v3["n_invalid"]) == (1.0, 1.0, run["n_even"])
    assert v3["flip_rate_against_v1"] == 0.0  # the invalid answers are left out


def test_the_none_arm_false_positives_are_those_of_the_none_arm_and_not_of_another(loud):
    _, run = loud
    hardness = run["report"]["fpr_by_hardness"]["none"]
    assert (hardness["hard"]["value"], hardness["other"]["value"]) == (1.0, 1.0)


def test_with_no_arm_beyond_the_noise_the_flips_of_all_name_and_swap_arms_are_read(quiet):
    results_dir, run = quiet
    rationalization = run["report"]["rationalization"]
    assert rationalization["beyond_noise"] is None
    assert rationalization["name_arms_above_noise"] == 0
    assert rationalization["hand_check_pool"] == "all_name_and_swap_flips"
    assert rationalization["all_name_and_swap_flips"]["n_flips"] == run["n_target"]
    sample = pd.read_csv(results_dir / "llm" / "rationalization_hand_check.csv")
    assert len(sample) == run["n_target"]


def test_the_hand_check_sample_holds_at_most_50_flips_drawn_with_the_seed_and_no_other_row(
    tmp_path,
):
    flips = pd.DataFrame(
        {
            "pid": [f"p{i}" for i in range(70)],
            "arm": "swap",
            "yes_none": 1.0,
            "yes": 0.0,
            "reason": "r",
            "flip": [i < 60 for i in range(70)],
        }
    )
    assert _load_analysis()._hand_check(flips, tmp_path / "sample.csv") is None
    sample = pd.read_csv(tmp_path / "sample.csv")
    expected = flips[flips["flip"]].sample(50, random_state=SEED).sort_index()
    assert list(sample["pid"]) == list(expected["pid"])


def test_a_run_that_is_not_complete_is_an_error_and_not_an_analysis_of_what_is_there(setup):  # noqa: F811
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(setup / "llm" / "main.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client, max_minutes=0, **SIZES)
        with pytest.raises(ValueError, match="incomplete"):
            _load_analysis().main(setup, client=client, n_boot=5)
    assert not (setup / "llm_audit.json").exists()


def test_the_cost_counts_each_stored_call_once_even_when_the_plan_asks_it_twice(setup, monkeypatch):  # noqa: F811
    original = data.load_showcase_subset

    def subset(path):  # no full stop followed by a space, so the whitespace edit changes nothing
        frame = original(path)
        return frame.assign(text=frame["text"].str.replace(". ", ", ", regex=False))

    monkeypatch.setattr(data, "load_showcase_subset", subset)
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(setup / "llm" / "main.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client, max_minutes=60, **SIZES)
        _load_analysis().main(setup, client=client, n_boot=20)
    cost = json.loads((setup / "llm_audit.json").read_text())["run_cost"]
    log = [json.loads(line) for line in (setup / "llm" / "main.jsonl").read_text().splitlines()]
    assert len(log) == 171  # 174 in the plan, and the 3 edits of the noise sample repeat none
    assert cost["n_calls"] == 171
    assert cost["seconds_total"] == pytest.approx(sum(call["seconds"] for call in log))
    assert cost["seconds_mean"] == pytest.approx(sum(call["seconds"] for call in log) / 171)


def test_a_rerun_before_any_reason_is_labelled_scores_nothing_and_keeps_the_sample(loud):
    results_dir, _ = loud
    path = results_dir / "llm" / "rationalization_hand_check.csv"
    before = path.read_text()
    with FakeLLMServer(lambda payload: pytest.fail("no call is asked again")) as server:
        client = llm_audit.Client(results_dir / "llm" / "main.jsonl", server.base_url, "m")
        _load_analysis().main(results_dir, client=client, n_boot=20)
    report = json.loads((results_dir / "llm_audit.json").read_text())
    assert report["rationalization"]["hand_check"] is None
    assert path.read_text() == before
