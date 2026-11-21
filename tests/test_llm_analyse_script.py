import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeLLMServer
from test_llm_run_script import SIZES, _load_script, setup  # noqa: F401  (fixture)

from hiring_bias_xai import llm_audit

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
        _load_analysis().main(setup, client=client, n_boot=50, **SIZES)
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
        _load_analysis().main(results_dir, client=client, n_boot=50, **SIZES)
    check = json.loads((results_dir / "llm_audit.json").read_text())
    summary = check["rationalization"]["hand_check"]
    assert summary["n"] == 24 and summary["n_rule_agrees"] == 20
    assert summary["agreement"]["value"] == pytest.approx(20 / 24)


def test_the_report_splits_the_none_arm_false_positives_by_hardness(analysed):
    _, report = analysed
    hardness = report["fpr_by_hardness"]["none"]
    assert hardness["hard"]["n"] == 6 and hardness["other"]["n"] == 6
    assert hardness["hard"]["value"] == 1.0
