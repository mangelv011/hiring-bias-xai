import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import FakeLLMServer

from hiring_bias_xai import (
    counterfactual,
    data,
    first_names,
    llm_audit,
    llm_plan,
    llm_profiles,
    results,
)
from hiring_bias_xai.config import SEED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_run.py"
JOBS = ["dietitian", "nurse", "model", "surgeon", "software_engineer", "composer"]
POOL = {
    "first_names": {"female": ["Maria", "Ana"], "male": ["Juan", "Luis"]},
    "surnames": ["Garcia", "Lopez"],
}
SIZES = {"n_match": 2, "n_nonmatch": 2, "n_noise": 3}


def _load_script():
    spec = importlib.util.spec_from_file_location("llm_run", SCRIPT)
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
    groups = [(o, g) for o in JOBS for g in ("female", "male") for _ in range(8)]
    rows = [
        (
            f"test-{i}",
            f"She is a {o}. Her work is {i}." + (" Maria knows." if i % 4 == 0 else ""),
            o,
            g,
        )
        for i, (o, g) in enumerate(groups)
    ]
    subset = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: subset)
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    nxt = {o: JOBS[(i + 1) % 6] for i, o in enumerate(JOBS)}
    monkeypatch.setattr(results, "load_predictions", lambda name, test: test["occupation"].map(nxt))
    monkeypatch.setattr(first_names, "load", lambda: frozenset({"Maria"}))
    monkeypatch.setattr(
        counterfactual, "first_names_in", lambda text, names: ["Maria"] if "Maria" in text else []
    )
    return results_dir


def _answer(payload):
    return {"content": '{"reason": "fits", "decision": "yes"}'}


def _run(results_dir, server, **kwargs):
    client = llm_audit.Client(results_dir / "llm" / "main.jsonl", server.base_url, "m")
    return _load_script().main(results_dir, client=client, **SIZES, **kwargs)


def test_profiles_come_only_from_bios_without_a_first_name_and_are_saved_for_later_chunks(setup):
    with FakeLLMServer(_answer) as server:
        _run(setup, server, max_minutes=0)
    profiles = pd.read_csv(setup / "llm" / "profiles.csv")
    assert len(profiles) == 6 * 4
    assert not profiles["text"].str.contains("Maria").any()
    assert profiles["noise_sample"].sum() == 3
    info = json.loads((setup / "llm" / "profiles.json").read_text())
    assert info["n_showcase_bios"] == 6 * 16 and info["n_with_first_name"] == 24
    assert info["n_without_first_name"] == 72


def test_the_run_stops_at_the_time_limit_and_resumes_from_the_log(setup):
    ticks = itertools.count(0, 60)
    with FakeLLMServer(_answer) as server:
        first = _run(setup, server, max_minutes=2.5, clock=lambda: next(ticks))
        assert (first["asked"], first["total"]) == (2, 174)
        assert len(server.requests) == 2
        second = _run(setup, server, max_minutes=60)
        assert second["asked"] == 172 and second["done"] == 174
        third = _run(setup, server, max_minutes=60)
        assert third["asked"] == 0
    assert len(server.requests) == 174
    assert len((setup / "llm" / "main.jsonl").read_text().splitlines()) == 174


def test_the_sizes_are_the_ones_fixed_in_the_adr():
    module = _load_script()
    assert (module.N_MATCH, module.N_NONMATCH, module.N_NOISE) == (70, 30, 200)


def test_the_saved_profiles_keep_the_job_order_the_hardest_confusions_and_a_seeded_noise_sample(
    setup,
):
    with FakeLLMServer(_answer) as server:
        _run(setup, server, max_minutes=0)
    profiles = pd.read_csv(setup / "llm" / "profiles.csv")
    assert list(dict.fromkeys(profiles["job"])) == JOBS
    info = json.loads((setup / "llm" / "profiles.json").read_text())
    assert (info["n_match"], info["n_nonmatch"], info["n_profiles"]) == (2, 2, 24)
    assert info["hardest"] == {job: JOBS[(i + 1) % 6] for i, job in enumerate(JOBS)}
    chosen = np.random.default_rng(SEED).choice(24, size=3, replace=False)
    assert list(profiles.index[profiles["noise_sample"]]) == sorted(chosen)


def test_a_later_chunk_reads_the_saved_profiles_and_does_not_sample_again(setup, monkeypatch):
    with FakeLLMServer(_answer) as server:
        first = _run(setup, server, max_minutes=0)
        monkeypatch.setattr(
            data,
            "load_showcase_subset",
            lambda path: pytest.fail("the profiles were sampled again"),
        )
        second = _run(setup, server, max_minutes=60)
    assert first["total"] == second["total"] == 174 and second["done"] == 174


def test_the_run_stops_when_the_time_limit_is_reached_exactly(setup):
    ticks = itertools.count(0, 60)
    with FakeLLMServer(_answer) as server:
        state = _run(setup, server, max_minutes=2, clock=lambda: next(ticks))
    assert state["asked"] == 1  # asked at 60 s, stopped at 120 s


def test_an_identical_rerun_is_asked_although_the_same_request_was_answered_with_run_0(setup):
    with FakeLLMServer(_answer) as server:
        _run(setup, server, max_minutes=0)
        profiles = pd.read_csv(setup / "llm" / "profiles.csv")
        names = json.loads((setup / "llm" / "names.json").read_text())["groups"]
        merge = json.loads((setup / "llm" / "name_check.json").read_text())["merge"]
        pools = llm_profiles.analysis_groups(names, merge)
        noise = list((profiles["job"] + "|" + profiles["id"])[profiles["noise_sample"]])
        plan = llm_plan.build(profiles, pools, noise)
        (none,) = plan[
            (plan["pid"] == noise[0]) & (plan["arm"] == "none") & (plan["part"] == "main")
        ]["messages"]
        client = llm_audit.Client(setup / "llm" / "main.jsonl", server.base_url, "m")
        client.ask(none)  # the none arm of a noise profile, answered with run index 0
        again = _run(setup, server, max_minutes=0)
        assert (again["asked"], again["done"]) == (0, 1)  # the rerun is not answered yet
        rest = _run(setup, server, max_minutes=60)
    assert rest["asked"] == 173 and rest["done"] == 174


def test_the_profile_counts_of_the_sizes_asked_are_saved_with_the_profiles(setup):
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(setup / "llm" / "main.jsonl", server.base_url, "m")
        _load_script().main(setup, client=client, max_minutes=0, n_match=3, n_nonmatch=2, n_noise=4)
    info = json.loads((setup / "llm" / "profiles.json").read_text())
    assert (info["n_match"], info["n_nonmatch"], info["n_profiles"]) == (3, 2, 30)
    assert pd.read_csv(setup / "llm" / "profiles.csv")["noise_sample"].sum() == 4
