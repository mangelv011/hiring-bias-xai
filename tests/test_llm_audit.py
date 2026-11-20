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


def test_reasoning_text_in_a_reply_is_kept_and_flagged_as_a_thinking_leak(tmp_path):
    replies = iter(
        [
            {"content": '{"reason": "r", "decision": "no"}'},
            {"content": "x", "reasoning_content": "Let me think"},
            {"content": "<think>hmm</think>" + '{"reason": "r", "decision": "no"}'},
        ]
    )
    with FakeLLMServer(lambda payload: next(replies)) as server:
        client = llm_audit.Client(tmp_path / "c.jsonl", base_url=server.base_url, model="m")
        records = [client.ask([{"role": "user", "content": str(i)}]) for i in range(3)]
    assert [record["reasoning"] for record in records] == ["", "Let me think", ""]
    assert [llm_audit.thinking_leaked(record) for record in records] == [False, True, True]


SHOWCASE = ["dietitian", "nurse", "model", "surgeon", "software_engineer", "composer"]


def test_the_name_check_prompt_holds_the_name_and_the_answer_choices():
    (message,) = llm_audit.name_check_messages("Ana Garcia")
    assert message["role"] == "user"
    assert "Ana Garcia" in message["content"]
    for choice in ("Spain", "Morocco", "Romania", "Hispanic America", "United Kingdom", "unclear"):
        assert choice in message["content"]
    assert "{" in message["content"] and "{name}" not in message["content"]


def test_the_screening_prompt_holds_the_job_ad_the_header_and_the_bio_whatever_the_bio_says():
    bio = "She is a nurse {with} braces and {candidate} in the text."
    plain = llm_audit.screening_messages("nurse", bio)[0]["content"]
    with_header = llm_audit.screening_messages("nurse", bio, header="Name: Ana Garcia")[0][
        "content"
    ]
    ad = llm_audit.job_ad("nurse")
    assert ad in plain and bio in plain
    assert "Name: Ana Garcia\n" + bio in with_header
    assert plain.count(bio) == 1 and "Name: Ana Garcia" not in plain
    assert "{job_ad}" not in plain and "{candidate}" not in with_header.replace(bio, "")


def test_every_showcase_occupation_has_a_job_ad_and_the_paraphrases_differ():
    ads = [llm_audit.job_ad(occupation) for occupation in SHOWCASE]
    assert len(set(ads)) == 6 and all(ad.strip() for ad in ads)
    prompts = [
        llm_audit.screening_messages("nurse", "bio", version=version)[0]["content"]
        for version in llm_audit.SCREENING_VERSIONS
    ]
    assert len(llm_audit.SCREENING_VERSIONS) == 3
    assert len(set(prompts)) == 3
    assert all("reason" in prompt and "decision" in prompt for prompt in prompts)
    assert all(prompt.index('"reason"') < prompt.index('"decision"') for prompt in prompts)


def test_a_strict_name_guess_parses_and_anything_else_is_invalid():
    assert llm_audit.parse_name_guess('{"origin": "Romania", "gender": "female"}') == {
        "origin": "Romania",
        "gender": "female",
    }
    assert llm_audit.parse_name_guess('{"origin": "unclear", "gender": "unclear"}') is not None
    invalid = [
        '{"origin": "France", "gender": "male"}',
        '{"origin": "Spain", "gender": "man"}',
        '{"origin": "Spain"}',
        '{"origin": "Spain", "gender": "male", "reason": "x"}',
        '```json\n{"origin": "Spain", "gender": "male"}\n```',
        "Spain, male",
        None,
    ]
    assert [llm_audit.parse_name_guess(text) for text in invalid] == [None] * len(invalid)


def _confusion(**rows):
    """Origin confusion with every group mostly recognised unless a row says otherwise."""
    matrix = {g: {g: 10} for g in llm_audit.ORIGINS}
    for group, answers in rows.items():
        matrix[group.replace("_", " ")] = answers
    return matrix


def test_groups_the_model_tells_apart_are_all_kept():
    merged = llm_audit.merge_decision(_confusion(Spain={"Spain": 6, "Hispanic America": 5}))
    assert merged == {"groups": [[g] for g in llm_audit.ORIGINS], "dropped": []}


def test_a_group_assigned_more_often_to_another_group_than_to_itself_is_merged_with_it():
    matrix = _confusion(Hispanic_America={"Hispanic America": 4, "Spain": 5, "unclear": 20})
    merged = llm_audit.merge_decision(matrix)
    assert merged == {
        "groups": [["Spain", "Hispanic America"], ["Morocco"], ["Romania"], ["United Kingdom"]],
        "dropped": [],
    }


def test_answers_of_unclear_and_invalid_count_for_neither_group():
    matrix = _confusion(Morocco={"Morocco": 1, "unclear": 30, "invalid": 30, "Romania": 1})
    assert llm_audit.merge_decision(matrix)["groups"] == [[g] for g in llm_audit.ORIGINS]


def test_a_group_confused_with_the_reference_group_is_dropped_and_the_reference_stays_whole():
    matrix = _confusion(Romania={"United Kingdom": 7, "Romania": 2})
    merged = llm_audit.merge_decision(matrix)
    assert merged["dropped"] == ["Romania"]
    assert merged["groups"] == [["Spain"], ["Morocco"], ["Hispanic America"], ["United Kingdom"]]
    matrix = _confusion(United_Kingdom={"Spain": 7, "United Kingdom": 2})
    merged = llm_audit.merge_decision(matrix)
    assert merged["dropped"] == ["Spain"]
    assert merged["groups"] == [["Morocco"], ["Romania"], ["Hispanic America"], ["United Kingdom"]]


def test_confusable_pairs_chain_into_one_group_and_a_dropped_group_takes_no_part():
    matrix = _confusion(
        Hispanic_America={"Spain": 5, "Hispanic America": 1},
        Morocco={"Hispanic America": 5, "Morocco": 1},
        Romania={"Morocco": 5, "Romania": 1, "United Kingdom": 3},
    )
    # Romania is closest to Morocco (5), which beats the United Kingdom (3): a merge, not a drop.
    merged = llm_audit.merge_decision(matrix)
    assert merged == {
        "groups": [["Spain", "Morocco", "Romania", "Hispanic America"], ["United Kingdom"]],
        "dropped": [],
    }
    matrix = _confusion(
        Morocco={"United Kingdom": 5, "Morocco": 1},
        Spain={"Morocco": 5, "Spain": 1},
    )
    merged = llm_audit.merge_decision(matrix)
    assert merged["dropped"] == ["Morocco"]
    assert merged["groups"] == [["Spain"], ["Romania"], ["Hispanic America"], ["United Kingdom"]]
