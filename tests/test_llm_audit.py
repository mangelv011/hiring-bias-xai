from hiring_bias_xai import llm_audit

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
