"""Pilot of the LLM audit: 30 profiles through each arm type, to measure the time per call, the
share of invalid answers and whether thinking is off. Writes `results/llm/pilot.json`; the calls
are kept in `results/llm/pilot.jsonl`. The decisions are not analysed: the pilot only sizes the
main run (ADR 0008). Needs `llm_names.py` and `llm_name_check.py` run first and the LLM server up.
"""

import json
from pathlib import Path
from statistics import mean, median

import numpy as np

from hiring_bias_xai import data, llm_audit, llm_profiles, results
from hiring_bias_xai.config import RESULTS_DIR, SEED

N_MATCH = 2  # per job: 6 jobs x (2 + 3) = 30 profiles
N_NONMATCH = 3
ARM_TYPES = ("none", "neutral", "swap", "name")


def main(results_dir: Path = RESULTS_DIR, client: llm_audit.Client | None = None) -> None:
    client = client or llm_audit.Client(results_dir / "llm" / "pilot.jsonl")
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    jobs = [e["occupation"] for e in showcase["female_dominated"] + showcase["male_dominated"]]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    test = data.load_split("test")
    hardest = llm_profiles.hardest_confusions(
        test["occupation"], results.load_predictions("deberta", test), jobs
    )
    names = json.loads((results_dir / "llm" / "names.json").read_text(encoding="utf-8"))
    merge = json.loads((results_dir / "llm" / "name_check.json").read_text())["merge"]
    pools = llm_profiles.analysis_groups(names["groups"], merge)
    name_arms = [arm for arm in llm_profiles.arm_labels(pools) if arm.startswith("name|")]
    profiles = llm_profiles.sample_profiles(subset, hardest, N_MATCH, N_NONMATCH)
    name_arm_of = np.random.default_rng(SEED).choice(name_arms, len(profiles))

    records = {kind: [] for kind in ARM_TYPES}
    for row in profiles.itertuples():
        for kind in ARM_TYPES:
            arm = name_arm_of[row.Index] if kind == "name" else kind
            header, text = llm_profiles.candidate(row.text, row.gender, row.id, arm, pools)
            records[kind].append(client.ask(llm_audit.screening_messages(row.job, text, header)))
    arm_types = {}
    for kind, calls in records.items():
        seconds = [call["seconds"] for call in calls]
        arm_types[kind] = {
            "n_calls": len(calls),
            "n_invalid": sum(llm_audit.parse_decision(call["content"]) is None for call in calls),
            "thinking_leaks": sum(llm_audit.thinking_leaked(call) for call in calls),
            "seconds_mean": mean(seconds),
            "seconds_median": median(seconds),
            "seconds_max": max(seconds),
        }
    everything = [call for calls in records.values() for call in calls]
    results.write_json(
        results_dir / "llm" / "pilot.json",
        {
            "model": client.model,
            "n_calls": len(everything),
            "seconds_total": sum(call["seconds"] for call in everything),
            "arm_types": arm_types,
        },
    )


if __name__ == "__main__":
    main()
