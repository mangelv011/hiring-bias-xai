"""Local LLM used as a black-box screener."""

import hashlib
import json


def request_key(model: str, messages: list[dict], settings: dict) -> str:
    """The sha256 of a request: the same model, messages and settings always give the same key."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "settings": settings}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
