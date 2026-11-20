"""Manipulation check of the LLM audit: can the model read origin and gender from a bare name?

Each full name is a first name from `results/llm/names.json` with a few surnames of the same
group, drawn with the global seed. The model guesses its origin and gender from the name alone.
`results/llm/name_check.json` holds both confusion matrices, the merge decision (the rule of
ADR 0008, `llm_audit.merge_decision`), the number of calls and the time they took. Every call is
kept in `results/llm/name_check.jsonl`, so a rerun asks nothing again. Needs `llm_names.py` run
first and the LLM server up.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from hiring_bias_xai import llm_audit, results
from hiring_bias_xai.config import RESULTS_DIR, SEED

SURNAMES_PER_FIRST_NAME = 3


def _full_names(groups: dict, per_first: int) -> list[tuple[str, str, str]]:
    """(group, sex, full name) for every first name with `per_first` surnames of its group."""
    rng = np.random.default_rng(SEED)
    names = []
    for group in llm_audit.ORIGINS:
        surnames = groups[group]["surnames"]
        for sex in llm_audit.GENDERS:
            for first in groups[group]["first_names"][sex]:
                picked = rng.choice(
                    len(surnames), size=min(per_first, len(surnames)), replace=False
                )
                names += [(group, sex, f"{first} {surnames[i]}") for i in picked]
    return names


def main(
    results_dir: Path = RESULTS_DIR,
    client: llm_audit.Client | None = None,
    surnames_per_first: int = SURNAMES_PER_FIRST_NAME,
) -> None:
    groups = json.loads((results_dir / "llm" / "names.json").read_text(encoding="utf-8"))["groups"]
    client = client or llm_audit.Client(results_dir / "llm" / "name_check.jsonl")
    origin = {group: Counter() for group in llm_audit.ORIGINS}
    gender = {sex: Counter() for sex in llm_audit.GENDERS}
    records = []
    for group, sex, name in _full_names(groups, surnames_per_first):
        record = client.ask(llm_audit.name_check_messages(name))
        records.append(record)
        guess = llm_audit.parse_name_guess(record["content"])
        origin[group][guess["origin"] if guess else "invalid"] += 1
        gender[sex][guess["gender"] if guess else "invalid"] += 1
    seconds = [record["seconds"] for record in records]
    results.write_json(
        results_dir / "llm" / "name_check.json",
        {
            "model": client.model,
            "seed": SEED,
            "surnames_per_first_name": surnames_per_first,
            "n_calls": len(records),
            "n_invalid": sum(origin[group]["invalid"] for group in origin),
            "thinking_leaks": sum(llm_audit.thinking_leaked(record) for record in records),
            "seconds": {"total": sum(seconds), "mean": sum(seconds) / len(seconds)},
            "origin_confusion": {group: dict(counts) for group, counts in origin.items()},
            "gender_confusion": {sex: dict(counts) for sex, counts in gender.items()},
            "merge": llm_audit.merge_decision(origin),
        },
    )


if __name__ == "__main__":
    main()
