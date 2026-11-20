"""Local LLM used as a black-box screener."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path

from hiring_bias_xai.config import SEED

# Greedy decoding with a fixed seed. The model reasons by default, which is switched off per
# request through the chat template.
SETTINGS = {
    "temperature": 0,
    "seed": SEED,
    "max_tokens": 400,
    "chat_template_kwargs": {"enable_thinking": False},
}
TIMEOUT_SECONDS = 300


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


class Client:
    """Asks an OpenAI-compatible chat endpoint and keeps every call in a JSONL log.

    A request already answered in the log is not sent again, so a run can be stopped and resumed.
    """

    def __init__(self, log: Path, base_url: str, model: str):
        self.log = log
        self.base_url = base_url
        self.model = model
        self._answered = {}
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self._answered[record["key"]] = record

    def ask(self, messages: list[dict]) -> dict:
        """The stored record of `messages`: key, model, messages, settings, the reply `content`
        and the `seconds` the call took. Sends the request only if the log has no answer yet."""
        key = request_key(self.model, messages, SETTINGS)
        if key in self._answered:
            return self._answered[key]
        body = json.dumps({"model": self.model, "messages": messages, **SETTINGS}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", body, {"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            reply = json.load(response)["choices"][0]["message"]
        record = {
            "key": key,
            "model": self.model,
            "messages": messages,
            "settings": SETTINGS,
            "content": reply["content"],
            "seconds": time.perf_counter() - start,
        }
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._answered[key] = record
        return record
