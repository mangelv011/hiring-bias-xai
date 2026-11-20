"""Main run of the LLM audit, in chunks: `python scripts/llm_run.py --max-minutes 8`.

The first call samples the profiles (`results/llm/profiles.csv`, with the counts in
`profiles.json`): per showcase occupation, `N_MATCH` matches and `N_NONMATCH` non-matches, half of
the non-matches hard, drawn only from bios without a first name in the text so the text never
contradicts a name header (ADR 0008). Later calls read that file. Then it asks every call of the
plan (`llm_plan.build`) that `results/llm/main.jsonl` does not hold yet, and stops before the time
limit. Run it again until it reports no calls remaining. Needs `llm_names.py` and
`llm_name_check.py` run first, the LLM server up and nothing else on the GPU.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hiring_bias_xai import (
    counterfactual,
    data,
    first_names,
    llm_audit,
    llm_plan,
    llm_profiles,
    results,
)
from hiring_bias_xai.config import RESULTS_DIR, SEED

N_MATCH = 70
N_NONMATCH = 30
N_NOISE = 200


def _sample_profiles(
    results_dir: Path, n_match: int, n_nonmatch: int, n_noise: int
) -> pd.DataFrame:
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    jobs = [e["occupation"] for e in showcase["female_dominated"] + showcase["male_dominated"]]
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    test = data.load_split("test")
    hardest = llm_profiles.hardest_confusions(
        test["occupation"], results.load_predictions("deberta", test), jobs
    )
    names = first_names.load()
    named = np.array([bool(counterfactual.first_names_in(t, names)) for t in subset["text"]])
    profiles = llm_profiles.sample_profiles(subset[~named], hardest, n_match, n_nonmatch)
    pids = profiles["job"] + "|" + profiles["id"]
    chosen = np.random.default_rng(SEED).choice(len(profiles), size=n_noise, replace=False)
    profiles["noise_sample"] = pids.index.isin(chosen)
    results.write_json(
        results_dir / "llm" / "profiles.json",
        {
            "n_showcase_bios": len(subset),
            "n_with_first_name": int(named.sum()),
            "n_without_first_name": int((~named).sum()),
            "n_profiles": len(profiles),
            "n_match": n_match,
            "n_nonmatch": n_nonmatch,
            "hardest": hardest,
        },
    )
    profiles.to_csv(results_dir / "llm" / "profiles.csv", index=False)
    return profiles


def main(
    results_dir: Path = RESULTS_DIR,
    client: llm_audit.Client | None = None,
    max_minutes: float = 8.0,
    n_match: int = N_MATCH,
    n_nonmatch: int = N_NONMATCH,
    n_noise: int = N_NOISE,
    clock=time.monotonic,
) -> dict:
    """Ask the calls of the plan not yet in the log until `max_minutes` have passed. Returns the
    number `asked` now, the number `done` in total and the `total` of the plan."""
    client = client or llm_audit.Client(results_dir / "llm" / "main.jsonl")
    path = results_dir / "llm" / "profiles.csv"
    if path.is_file():
        profiles = pd.read_csv(path)
    else:
        profiles = _sample_profiles(results_dir, n_match, n_nonmatch, n_noise)
    names = json.loads((results_dir / "llm" / "names.json").read_text(encoding="utf-8"))
    merge = json.loads((results_dir / "llm" / "name_check.json").read_text())["merge"]
    pools = llm_profiles.analysis_groups(names["groups"], merge)
    noise = list((profiles["job"] + "|" + profiles["id"])[profiles["noise_sample"]])
    plan = llm_plan.build(profiles, pools, noise)
    start, asked = clock(), 0
    for call in plan.itertuples():
        if client.lookup(call.messages, call.run) is not None:
            continue
        if clock() - start >= max_minutes * 60:
            break
        client.ask(call.messages, call.run)
        asked += 1
    done = sum(
        client.lookup(m, r) is not None for m, r in zip(plan["messages"], plan["run"], strict=True)
    )
    return {"asked": asked, "done": done, "total": len(plan)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-minutes", type=float, default=8.0)
    state = main(max_minutes=parser.parse_args().max_minutes)
    print(f"asked {state['asked']}, done {state['done']} of {state['total']}")
