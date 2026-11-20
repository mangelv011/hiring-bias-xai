import json

from fakes import FakeLLMServer

from hiring_bias_xai import llm_audit
from hiring_bias_xai.config import SEED

MESSAGES = [{"role": "user", "content": "hello"}]


def test_the_request_key_depends_on_model_messages_and_settings_and_nothing_else():
    key = llm_audit.request_key("m", MESSAGES, {"temperature": 0, "seed": 1})
    assert key == llm_audit.request_key("m", list(MESSAGES), {"seed": 1, "temperature": 0})
    assert key != llm_audit.request_key("other", MESSAGES, {"temperature": 0, "seed": 1})
    assert key != llm_audit.request_key(
        "m", [{"role": "user", "content": "hello "}], {"temperature": 0, "seed": 1}
    )
    assert key != llm_audit.request_key("m", MESSAGES, {"temperature": 0, "seed": 2})
    assert len(key) == 64


def test_a_strict_answer_parses_to_its_reason_and_decision():
    answer = '{"reason": "ten years of nursing", "decision": "yes"}'
    assert llm_audit.parse_decision(answer) == {"reason": "ten years of nursing", "decision": "yes"}
    assert llm_audit.parse_decision('{"reason": "", "decision": "no"}')["decision"] == "no"


def test_anything_but_a_strict_answer_is_invalid_and_is_never_coerced():
    invalid = [
        "yes",
        "",
        '```json\n{"reason": "r", "decision": "yes"}\n```',
        'Sure. {"reason": "r", "decision": "yes"}',
        '{"decision": "yes"}',
        '{"reason": "r"}',
        '{"reason": "r", "decision": "Yes"}',
        '{"reason": "r", "decision": true}',
        '{"reason": 3, "decision": "yes"}',
        '{"reason": "r", "decision": "yes", "extra": 1}',
        '["reason", "decision"]',
        '{"reason": "r", "decision": "maybe"}',
        None,
    ]
    assert [llm_audit.parse_decision(text) for text in invalid] == [None] * len(invalid)


def _answer(payload):
    return {"role": "assistant", "content": '{"reason": "r", "decision": "yes"}'}


def test_asking_sends_the_fixed_settings_and_stores_the_request_and_the_response(tmp_path):
    log = tmp_path / "llm" / "calls.jsonl"
    with FakeLLMServer(_answer) as server:
        client = llm_audit.Client(log, base_url=server.base_url, model="m")
        record = client.ask(MESSAGES)
    (request,) = server.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["payload"]["model"] == "m"
    assert request["payload"]["messages"] == MESSAGES
    assert request["payload"]["temperature"] == 0
    assert request["payload"]["seed"] == SEED
    assert request["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert record["content"] == '{"reason": "r", "decision": "yes"}'
    assert record["key"] == llm_audit.request_key("m", MESSAGES, llm_audit.SETTINGS)
    assert record["seconds"] > 0
    assert json.loads(log.read_text()) == record


def test_a_rerun_skips_the_requests_already_answered(tmp_path):
    log = tmp_path / "calls.jsonl"
    with FakeLLMServer(_answer) as server:
        first = llm_audit.Client(log, base_url=server.base_url, model="m").ask(MESSAGES)
        again = llm_audit.Client(log, base_url=server.base_url, model="m")
        assert again.ask(MESSAGES) == first
        again.ask([{"role": "user", "content": "other"}])
    assert len(server.requests) == 2
    assert len(log.read_text().splitlines()) == 2


def test_base_url_and_model_come_from_the_environment_with_the_local_server_as_default(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("HBX_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("HBX_LLM_MODEL", raising=False)
    default = llm_audit.Client(tmp_path / "a.jsonl")
    assert (default.base_url, default.model) == ("http://127.0.0.1:11400/v1", "pc-equilibrado")
    monkeypatch.setenv("HBX_LLM_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("HBX_LLM_MODEL", "other")
    from_env = llm_audit.Client(tmp_path / "a.jsonl")
    assert (from_env.base_url, from_env.model) == ("http://example.test/v1", "other")
    explicit = llm_audit.Client(tmp_path / "a.jsonl", base_url="http://x/v1", model="m")
    assert (explicit.base_url, explicit.model) == ("http://x/v1", "m")
