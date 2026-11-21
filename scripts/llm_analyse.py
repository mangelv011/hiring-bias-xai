"""Metrics of the LLM audit: reads the plan and the stored calls, writes `results/llm_audit.json`.

TPR and FPR per arm, the differences from the `none` arm with a bootstrap over bios and first
names, the primary contrast (women's names minus men's names) and the other contrasts with Holm,
the range of yes-rates across first names, the noise floor, prompt sensitivity, the split of
flips by whether their reason mentions the attribute, and the run cost (ADR 0008). It also writes
`results/llm/rationalization_hand_check.csv`, a sample of flips whose `hand_label` I fill in by
reading the reasons (1 if the reason mentions the candidate's name, origin, nationality or gender
as an attribute, 0 if not); a rerun scores the labels against the keyword rule. Run `llm_run.py`
until it reports no calls remaining first.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hiring_bias_xai import llm_audit, llm_metrics, llm_plan, llm_profiles, results
from hiring_bias_xai.config import RESULTS_DIR, SEED
from hiring_bias_xai.counterfactual_effects import wilson_interval

N_HAND_CHECK = 50
MODEL_CARD = {
    "server_model_name": "pc-equilibrado",
    "file": "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
    "huggingface": "prism-ml/Ternary-Bonsai-2-9B-gguf",
    "licence": "Apache-2.0",
    "base_model": "Qwen/Qwen3.5-9B",
    "quantization": "ternary 2-bit (PQ2_0)",
    "server": "llama-swap with a Prism ML llama.cpp fork, one slot (-np 1), context 81920",
}


def _answered(plan: pd.DataFrame, client: llm_audit.Client) -> pd.DataFrame:
    """The plan with each call's `yes` (NaN if invalid), `reason`, `seconds` and `leaked`."""
    yes, reason, seconds, leaked, keys = [], [], [], [], []
    for call in plan.itertuples():
        record = client.lookup(call.messages, call.run)
        if record is None:
            raise ValueError("the run is incomplete: run scripts/llm_run.py until nothing remains")
        answer = llm_audit.parse_decision(record["content"])
        yes.append(np.nan if answer is None else float(answer["decision"] == "yes"))
        reason.append("" if answer is None else answer["reason"])
        seconds.append(record["seconds"])
        leaked.append(llm_audit.thinking_leaked(record))
        keys.append(record["key"])
    return plan.assign(yes=yes, reason=reason, seconds=seconds, leaked=leaked, key=keys)


def _prompt_sensitivity(calls: pd.DataFrame, main: pd.DataFrame) -> dict:
    base = main[main["arm"] == "none"].set_index("pid")["yes"]
    report = {}
    for version, rows in calls[calls["part"] == "paraphrase"].groupby("version"):
        rates = llm_metrics.arm_rates(rows.assign(arm="none"))["none"]
        both = rows.set_index("pid")["yes"].to_frame("yes").join(base.rename("base")).dropna()
        report[version] = rates | {
            "flip_rate_against_v1": float((both["yes"] != both["base"]).mean())
        }
    return report


def _hand_check(flips: pd.DataFrame, path: Path) -> dict | None:
    """Write the sample for labelling if it does not exist; score it once it is labelled."""
    if not path.is_file():
        pool = flips[flips["flip"]]
        rows = pool.sample(min(N_HAND_CHECK, len(pool)), random_state=SEED).sort_index()
        sample = rows[["pid", "arm", "yes_none", "yes", "reason"]].assign(
            rule_mention=[llm_metrics.mentions(r) for r in rows["reason"]], hand_label=np.nan
        )
        sample.to_csv(path, index=False)
        return None
    sample = pd.read_csv(path)
    labelled = sample[sample["hand_label"].notna()]
    if labelled.empty:
        return None
    n = len(labelled)
    agree = int(
        (labelled["rule_mention"].astype(bool) == labelled["hand_label"].astype(bool)).sum()
    )
    low, high = wilson_interval(agree, n)
    hand_no = int((labelled["hand_label"] == 0).sum())
    no_low, no_high = wilson_interval(hand_no, n)
    return {
        "n": n,
        "n_rule_agrees": agree,
        "agreement": {"value": agree / n, "ci_low": low, "ci_high": high},
        "n_hand_mention_rule_misses": int(
            ((labelled["hand_label"] == 1) & ~labelled["rule_mention"].astype(bool)).sum()
        ),
        "n_rule_mention_hand_not": int(
            ((labelled["hand_label"] == 0) & labelled["rule_mention"].astype(bool)).sum()
        ),
        "share_no_mention_by_hand": {"value": hand_no / n, "ci_low": no_low, "ci_high": no_high},
    }


def main(
    results_dir: Path = RESULTS_DIR,
    client: llm_audit.Client | None = None,
    n_boot: int = llm_metrics.N_BOOT,
) -> dict:
    client = client or llm_audit.Client(results_dir / "llm" / "main.jsonl")
    llm_dir = results_dir / "llm"
    profiles = pd.read_csv(llm_dir / "profiles.csv")
    names = json.loads((llm_dir / "names.json").read_text(encoding="utf-8"))
    check = json.loads((llm_dir / "name_check.json").read_text())
    pools = llm_profiles.analysis_groups(names["groups"], check["merge"])
    noise_pids = list((profiles["job"] + "|" + profiles["id"])[profiles["noise_sample"]])
    calls = _answered(llm_plan.build(profiles, pools, noise_pids), client)
    main_calls = calls[calls["part"] == "main"].reset_index(drop=True)
    noise = calls[calls["part"] == "noise"].assign(part=lambda d: d["noise"])
    report = llm_metrics.contrasts(main_calls, n_boot=n_boot)
    floor = llm_metrics.noise_floor(main_calls, noise, n_boot=n_boot)
    flips = llm_metrics.flips(main_calls)
    flips = flips[flips["kind"].isin(["name", "swap"])].reset_index(drop=True)
    above = {arm for arm, entry in floor["arms"].items() if entry["above_noise"]}
    beyond = flips[flips["arm"].isin(above)]
    pool = beyond if len(beyond) else flips
    unique = calls.drop_duplicates("key")
    profile_info = json.loads((llm_dir / "profiles.json").read_text())
    results.write_json(
        results_dir / "llm_audit.json",
        {
            "model": client.model,
            "model_card": MODEL_CARD,
            "settings": llm_audit.SETTINGS,
            "profiles": profile_info,
            "n_noise_profiles": len(noise_pids),
            "manipulation_check": check,
            "arms": llm_metrics.arm_rates(main_calls),
            "fpr_by_hardness": {"none": llm_metrics.fpr_by_hardness(main_calls, "none")},
            "statistics": report["statistics"],
            "holm": report["holm"],
            "name_ranges": llm_metrics.name_ranges(main_calls),
            "noise_floor": floor,
            "prompt_sensitivity": _prompt_sensitivity(calls, main_calls),
            "rationalization": {
                "name_arms_above_noise": sum(arm.startswith("name|") for arm in above),
                "beyond_noise": llm_metrics.stated_reasons(beyond) if len(beyond) else None,
                "all_name_and_swap_flips": llm_metrics.stated_reasons(flips)
                if len(flips)
                else None,
                "hand_check_pool": "beyond_noise" if len(beyond) else "all_name_and_swap_flips",
                "hand_check": _hand_check(pool, llm_dir / "rationalization_hand_check.csv"),
            },
            "run_cost": {
                "n_calls": len(unique),
                "seconds_total": float(unique["seconds"].sum()),
                "seconds_mean": float(unique["seconds"].mean()),
                "n_invalid": int(unique["yes"].isna().sum()),
                "thinking_leaks": int(unique["leaked"].sum()),
            },
        },
    )


if __name__ == "__main__":
    main()
