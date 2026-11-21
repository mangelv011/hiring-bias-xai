import importlib.util
import json
from pathlib import Path

import pytest
from fakes import FakeLLMServer

from hiring_bias_xai import llm_audit, results

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_name_check.py"
GROUPS = {
    "Spain": (["Antonio", "Maria"], ["Garcia", "Lopez"]),
    "Morocco": (["Mohamed", "Fatima"], ["Benali", "Tahiri"]),
    "Romania": (["Ion", "Elena"], ["Popescu", "Radu"]),
    "Hispanic America": (["Santiago", "Valentina"], ["Zambrano", "Vera"]),
    "United Kingdom": (["John", "Susan"], ["Smith", "Jones"]),
}
FEMALE = {"Maria", "Fatima", "Elena", "Valentina", "Susan"}


def _load_script():
    spec = importlib.util.spec_from_file_location("llm_name_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names_json(results_dir):
    groups = {
        group: {
            "first_names": {
                "male": [n for n in firsts if n not in FEMALE],
                "female": [n for n in firsts if n in FEMALE],
            },
            "surnames": surnames,
        }
        for group, (firsts, surnames) in GROUPS.items()
    }
    (results_dir / "llm").mkdir(parents=True)
    results.write_json(results_dir / "llm" / "names.json", {"groups": groups})


def _guess(payload):
    """Morocco's names are taken for British ones; one reply is not JSON; the rest are right."""
    text = payload["messages"][0]["content"]
    name = text.split("Name: ")[1].split("\n")[0]
    first, surname = name.split()
    origin = next(g for g, (_, surnames) in GROUPS.items() if surname in surnames)
    if origin == "Morocco":
        origin = "United Kingdom"
    gender = "female" if first in FEMALE else "male"
    if name == "Ion Radu":
        return {"content": "I think Romanian"}
    return {"content": json.dumps({"origin": origin, "gender": gender})}


def _run(tmp_path):
    results_dir = tmp_path / "results"
    _names_json(results_dir)
    with FakeLLMServer(_guess) as server:
        client = llm_audit.Client(results_dir / "llm" / "name_check.jsonl", server.base_url, "m")
        _load_script().main(results_dir, client=client, surnames_per_first=2)
    return results_dir, server, json.loads((results_dir / "llm" / "name_check.json").read_text())


def test_every_full_name_is_asked_once_and_the_confusion_matrices_count_the_answers(tmp_path):
    results_dir, server, record = _run(tmp_path)
    assert len(server.requests) == 20  # 5 groups x 2 first names x 2 surnames
    origin = record["origin_confusion"]
    assert origin["Spain"] == {"Spain": 4}
    assert origin["Morocco"] == {"United Kingdom": 4}
    assert origin["Romania"] == {"Romania": 3, "invalid": 1}
    assert origin["United Kingdom"] == {"United Kingdom": 4}
    assert record["gender_confusion"] == {
        "female": {"female": 10},
        "male": {"male": 9, "invalid": 1},
    }


def test_the_record_holds_the_merge_decision_the_cost_and_the_model(tmp_path):
    results_dir, _, record = _run(tmp_path)
    assert record["merge"]["dropped"] == ["Morocco"]
    assert record["merge"]["groups"] == [
        ["Spain"],
        ["Romania"],
        ["Hispanic America"],
        ["United Kingdom"],
    ]
    assert record["n_calls"] == 20 and record["n_invalid"] == 1
    assert record["thinking_leaks"] == 0
    assert record["seconds"]["total"] > 0 and record["seconds"]["mean"] > 0
    assert record["model"] == "m"
    assert record["surnames_per_first_name"] == 2
    assert len((results_dir / "llm" / "name_check.jsonl").read_text().splitlines()) == 20


def _run_default(tmp_path, answer, surnames=None):
    """Run with the default number of surnames per first name; `surnames` replaces the lists."""
    results_dir = tmp_path / "results"
    _names_json(results_dir)
    if surnames:
        path = results_dir / "llm" / "names.json"
        names = json.loads(path.read_text(encoding="utf-8"))
        for content in names["groups"].values():
            content["surnames"] = surnames
        results.write_json(path, names)
    with FakeLLMServer(answer) as server:
        client = llm_audit.Client(results_dir / "llm" / "name_check.jsonl", server.base_url, "m")
        _load_script().main(results_dir, client=client)
    return server, json.loads((results_dir / "llm" / "name_check.json").read_text())


def _asked(server):
    return [
        r["payload"]["messages"][0]["content"].split("Name: ")[1].split("\n")[0]
        for r in server.requests
    ]


def _unclear(payload):
    return {"content": '{"origin": "unclear", "gender": "unclear"}'}


def test_by_default_each_first_name_is_asked_with_three_surnames_or_all_it_has(tmp_path):
    server, record = _run_default(tmp_path, _guess)
    assert len(server.requests) == 20 and record["n_calls"] == 20  # 2 surnames in each group
    assert record["surnames_per_first_name"] == 3
    server, record = _run_default(tmp_path / "more", _unclear, surnames=["S1", "S2", "S3", "S4"])
    assert len(server.requests) == 30 and record["n_calls"] == 30  # 5 groups x 2 x 3


def test_the_surnames_of_each_first_name_are_drawn_with_the_global_seed(tmp_path):
    server, _ = _run_default(tmp_path, _unclear, surnames=["S1", "S2", "S3", "S4"])
    assert _asked(server)[:6] == [
        "Maria S4", "Maria S1", "Maria S3", "Antonio S1", "Antonio S3", "Antonio S2",
    ]  # fmt: skip


def test_names_of_one_group_taken_for_another_are_merged_by_the_rule_of_the_adr(tmp_path):
    def answer(payload):
        reply = _guess(payload)
        if any(surname in payload["messages"][0]["content"] for surname in GROUPS["Spain"][1]):
            guess = json.loads(reply["content"])
            reply = {"content": json.dumps(guess | {"origin": "Romania"})}
        return reply

    _, record = _run_default(tmp_path, answer)
    assert record["origin_confusion"]["Spain"] == {"Romania": 4}
    assert record["merge"] == {
        "groups": [["Spain", "Romania"], ["Hispanic America"], ["United Kingdom"]],
        "dropped": ["Morocco"],
    }


def test_the_mean_time_is_the_total_over_the_calls_and_thinking_leaks_are_counted(tmp_path):
    def answer(payload):
        reply = _guess(payload)
        if any(surname in payload["messages"][0]["content"] for surname in GROUPS["Spain"][1]):
            reply["reasoning_content"] = "hm"
        return reply

    _, record = _run_default(tmp_path, answer)
    assert record["thinking_leaks"] == 4  # the 4 names of Spain
    assert record["seconds"]["mean"] == pytest.approx(record["seconds"]["total"] / 20)
