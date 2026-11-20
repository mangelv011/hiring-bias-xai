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
