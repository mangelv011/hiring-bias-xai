import importlib.util
import json
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import FakeLLMServer

from hiring_bias_xai import data, llm_audit, results
from hiring_bias_xai.config import SEED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_pilot.py"
JOBS = ["dietitian", "nurse", "model", "surgeon", "software_engineer", "composer"]
POOL = {
    "first_names": {"female": ["Maria", "Ana"], "male": ["Juan", "Luis"]},
    "surnames": ["Garcia", "Lopez"],
}


def _load_script():
    spec = importlib.util.spec_from_file_location("llm_pilot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "llm").mkdir(parents=True)
    results.write_json(
        results_dir / "showcase_occupations.json",
        {
            "female_dominated": [{"occupation": o} for o in JOBS[:3]],
            "male_dominated": [{"occupation": o} for o in JOBS[3:]],
        },
    )
    results.write_json(
        results_dir / "llm" / "names.json", {"groups": {"Spain": POOL, "Romania": POOL}}
    )
    results.write_json(
        results_dir / "llm" / "name_check.json",
        {"merge": {"groups": [["Spain", "Romania"]], "dropped": []}},
    )
    rows = [
        (f"test-{i}", f"She is a {o}. Her work is {i}.", o, g)
        for i, (o, g) in enumerate(
            (o, g) for o in JOBS for g in ("female", "male") for _ in range(6)
        )
    ]
    subset = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: subset)
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    # the main model takes every bio for the next occupation in the list
    nxt = {o: JOBS[(i + 1) % 6] for i, o in enumerate(JOBS)}
    monkeypatch.setattr(results, "load_predictions", lambda name, test: test["occupation"].map(nxt))
    return results_dir


def _answer():
    """Valid answers, except the first neutral-arm request, which gets plain text."""
    seen = []

    def answer(payload):
        content = payload["messages"][0]["content"]
        if "Candidate ID: " in content and not seen:
            seen.append(content)
            return {"content": "maybe"}
        return {"content": '{"reason": "fits", "decision": "yes"}'}

    return answer


def test_the_pilot_asks_about_30_profiles_per_arm_type_and_reports_time_and_invalid_answers(setup):
    with FakeLLMServer(_answer()) as server:
        client = llm_audit.Client(setup / "llm" / "pilot.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client)
    record = json.loads((setup / "llm" / "pilot.json").read_text())
    types = record["arm_types"]
    assert set(types) == {"none", "neutral", "swap", "name"}
    assert {t["n_calls"] for t in types.values()} == {30}
    assert len(server.requests) == 120
    assert types["neutral"]["n_invalid"] == 1 and types["none"]["n_invalid"] == 0
    assert types["name"]["thinking_leaks"] == 0
    assert all(
        t["seconds_mean"] > 0 and t["seconds_max"] >= t["seconds_median"] for t in types.values()
    )
    assert record["n_calls"] == 120 and record["seconds_total"] > 0 and record["model"] == "m"
    sent = [r["payload"]["messages"][0]["content"] for r in server.requests]
    assert sum("Name: " in text for text in sent) == 30
    assert sum("Candidate ID: " in text for text in sent) == 30


def _run_pilot(setup, answer):
    with FakeLLMServer(answer) as server:
        client = llm_audit.Client(setup / "llm" / "pilot.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client)
    log = [json.loads(line) for line in (setup / "llm" / "pilot.jsonl").read_text().splitlines()]
    return log, json.loads((setup / "llm" / "pilot.json").read_text())


def test_the_times_are_the_mean_median_max_and_total_of_the_stored_calls(setup):
    log, record = _run_pilot(setup, _answer())
    assert len(log) == 120  # asked profile by profile, the four arm types in turn
    for position, kind in enumerate(("none", "neutral", "swap", "name")):
        seconds = [call["seconds"] for call in log[position::4]]
        entry = record["arm_types"][kind]
        assert entry["seconds_mean"] == pytest.approx(statistics.mean(seconds))
        assert entry["seconds_median"] == pytest.approx(statistics.median(seconds))
        assert entry["seconds_max"] == pytest.approx(max(seconds))
    assert record["seconds_total"] == pytest.approx(sum(call["seconds"] for call in log))


def test_the_pilot_counts_thinking_leaks_by_arm_type(setup):
    def answer(payload):
        reply = {"content": '{"reason": "fits", "decision": "yes"}'}
        if "Candidate ID: " in payload["messages"][0]["content"]:
            reply["reasoning_content"] = "hm"
        return reply

    _, record = _run_pilot(setup, answer)
    leaks = {kind: entry["thinking_leaks"] for kind, entry in record["arm_types"].items()}
    assert leaks == {"none": 0, "neutral": 30, "swap": 0, "name": 0}


def test_each_job_gets_2_qualified_and_3_other_bios_and_the_name_arm_is_drawn_with_the_seed(setup):
    log, _ = _run_pilot(setup, _answer())
    none = [call["messages"][0]["content"] for call in log[0::4]]
    for job in JOBS:
        asked = [text for text in none if llm_audit.job_ad(job) in text]
        occupations = [re.search(r"She is a (\w+)\.", text)[1] for text in asked]
        assert len(asked) == 5 and occupations.count(job) == 2
    arms = np.random.default_rng(SEED).choice(
        ["name|Spain + Romania|female", "name|Spain + Romania|male"], 30
    )
    named = [call["messages"][0]["content"] for call in log[3::4]]
    women = [re.search(r"Name: (\w+)", text)[1] in ("Maria", "Ana") for text in named]
    assert women == [arm.endswith("female") for arm in arms]
