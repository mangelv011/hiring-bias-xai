"""Local LLM used as a black-box screener."""

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from hiring_bias_xai.config import REPO_ROOT, SEED

# Greedy decoding with a fixed seed. The model reasons by default, which is switched off per
# request through the chat template.
SETTINGS = {
    "temperature": 0,
    "seed": SEED,
    "max_tokens": 400,
    "chat_template_kwargs": {"enable_thinking": False},
}
TIMEOUT_SECONDS = 300
DEFAULT_BASE_URL = "http://127.0.0.1:11400/v1"
DEFAULT_MODEL = "pc-equilibrado"

PROMPTS_DIR = REPO_ROOT / "prompts"
SCREENING_VERSIONS = ("v1", "v2", "v3")  # v1 is the instruction; v2 and v3 are paraphrases


def request_key(model: str, messages: list[dict], settings: dict) -> str:
    """The sha256 of a request: the same model, messages and settings always give the same key."""
    canonical = json.dumps(
        {"model": model, "messages": messages, "settings": settings}, sort_keys=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def job_ad(occupation: str) -> str:
    """The short job ad for `occupation`, from `prompts/job_ads/`."""
    return _prompt(f"job_ads/{occupation}.txt").strip()


def name_check_messages(name: str) -> list[dict]:
    """The chat messages that ask for the likely origin and gender of a full name."""
    return [{"role": "user", "content": _prompt("name_check_v1.txt").replace("{name}", name)}]


def screening_messages(
    occupation: str, bio: str, header: str | None = None, version: str = "v1"
) -> list[dict]:
    """The chat messages that ask whether the candidate fits the job ad of `occupation`.

    `header`, if given, is the first line of the candidate block, before the bio. The bio is
    inserted last, so braces in it are never read as placeholders.
    """
    candidate = bio if header is None else f"{header}\n{bio}"
    template = _prompt(f"screening_{version}.txt").replace("{job_ad}", job_ad(occupation))
    return [{"role": "user", "content": template.replace("{candidate}", candidate)}]


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


ORIGINS = ("Spain", "Morocco", "Romania", "Hispanic America", "United Kingdom")
GENDERS = ("female", "male")


def parse_name_guess(content: str | None) -> dict | None:
    """`{"origin": ..., "gender": ...}` read from `content`, or None if it is not exactly that,
    with an origin among `ORIGINS` or "unclear" and a gender among `GENDERS` or "unclear"."""
    try:
        guess = json.loads(content)
    except (TypeError, ValueError):
        return None
    if (
        isinstance(guess, dict)
        and set(guess) == {"origin", "gender"}
        and guess["origin"] in (*ORIGINS, "unclear")
        and guess["gender"] in (*GENDERS, "unclear")
    ):
        return guess
    return None


REFERENCE_GROUP = "United Kingdom"


def merge_decision(confusion: dict[str, dict[str, int]]) -> dict:
    """The groups to analyse, given how often the model gave each answer for each group's names.

    `confusion[group][answer]` counts answers that are an origin group; "unclear" and "invalid"
    answers count for no group. A group is confusable with the other group it is assigned to most
    often (the first in `ORIGINS` on a tie) when that count is higher than its own. A pair with
    the reference group drops the other group, so the reference group stays whole. The remaining
    confusable pairs are merged, and chains of pairs end up in one group. Applied once.
    Returns `{"groups": [[origin, ...], ...], "dropped": [origin, ...]}` in `ORIGINS` order.
    """
    pairs = []
    for group in ORIGINS:
        others = [h for h in ORIGINS if h != group]
        closest = max(others, key=lambda h: confusion[group].get(h, 0))
        if confusion[group].get(closest, 0) > confusion[group].get(group, 0):
            pairs.append((group, closest))
    dropped = {a if b == REFERENCE_GROUP else b for a, b in pairs if REFERENCE_GROUP in (a, b)}
    merged = {group: {group} for group in ORIGINS if group not in dropped}
    for a, b in pairs:
        if a in merged and b in merged and merged[a] is not merged[b]:
            union = merged[a] | merged[b]
            for member in union:
                merged[member] = union
    groups = []
    for group in ORIGINS:
        if group in merged and group == min(merged[group], key=ORIGINS.index):
            groups.append(sorted(merged[group], key=ORIGINS.index))
    return {"groups": groups, "dropped": [g for g in ORIGINS if g in dropped]}


def thinking_leaked(record: dict) -> bool:
    """Whether a stored call carries reasoning text, so thinking was not switched off."""
    return bool(record["reasoning"]) or "<think>" in record["content"]


class Client:
    """Asks an OpenAI-compatible chat endpoint and keeps every call in a JSONL log.

    The server and model are `HBX_LLM_BASE_URL` and `HBX_LLM_MODEL`, by default the local
    llama-swap endpoint and `pc-equilibrado`. A request already answered in the log is not sent
    again, so a run can be stopped and resumed.
    """

    def __init__(self, log: Path, base_url: str | None = None, model: str | None = None):
        self.log = log
        self.base_url = base_url or os.environ.get("HBX_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("HBX_LLM_MODEL", DEFAULT_MODEL)
        self._answered = {}
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self._answered[record["key"]] = record

    def ask(self, messages: list[dict]) -> dict:
        """The stored record of `messages`: key, model, messages, settings, the reply `content`,
        the `reasoning` text if any and the `seconds` the call took. The request is sent only if
        the log has no answer for it yet."""
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
            "reasoning": reply.get("reasoning_content") or "",
            "seconds": time.perf_counter() - start,
        }
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._answered[key] = record
        return record
