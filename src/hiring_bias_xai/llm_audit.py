"""Local LLM used as a black-box screener."""

import hashlib
import json


def request_key(model: str, messages: list[dict], settings: dict) -> str:
    """The sha256 of a request: the same model, messages and settings always give the same key."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "settings": settings}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def parse_decision(content: str | None) -> dict | None:
    """`{"reason": str, "decision": "yes" | "no"}` read from `content`, or None if it is not
    exactly that: no extra text, no other keys, no other value. Nothing is coerced."""
    try:
        answer = json.loads(content)
    except (TypeError, ValueError):
        return None
    if (
        isinstance(answer, dict)
        and set(answer) == {"reason", "decision"}
        and isinstance(answer["reason"], str)
        and answer["decision"] in ("yes", "no")
    ):
        return answer
    return None
