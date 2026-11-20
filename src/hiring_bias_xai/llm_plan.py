"""Every call of the main run of the LLM audit, as a table, from the profiles and the name pools."""

import pandas as pd

from hiring_bias_xai import llm_audit, llm_profiles

PARAPHRASES = ("v2", "v3")


def edit_whitespace(text: str) -> str:
    """The trivial edit of the noise floor: a second space after every full stop."""
    return text.replace(". ", ".  ")


def build(profiles: pd.DataFrame, pools: dict, noise_pids: list[str]) -> pd.DataFrame:
    """One row per call. `part` is `noise` (the `none` arm asked again: `noise` is `rerun`, an
    identical request with run index 1, or `edit`, the bio with the whitespace edit), `main`
    (every arm, version v1) or `paraphrase` (the `none` arm with the other instructions).
    Columns: `part`, `noise`, `pid` (`<job>|<bio id>`), `job`, `bio_id`, `match`, `hard`,
    `bio_gender`, `arm`, `kind`, `group`, `name_gender`, `first`, `version`, `run` and the chat
    `messages`."""
    rows = []
    for profile in profiles.itertuples():
        pid = f"{profile.job}|{profile.id}"

        def add(part, arm, version="v1", run=0, noise=None, edit=False, profile=profile, pid=pid):
            header, text = llm_profiles.candidate(
                profile.text, profile.gender, profile.id, arm, pools
            )
            kind, _, rest = arm.partition("|")
            group, _, gender = rest.partition("|")
            first = None
            if kind == "name":
                first = llm_profiles.draw_name(pools, group, gender, profile.id)[0]
            if edit:
                text = edit_whitespace(text)
            messages = llm_audit.screening_messages(profile.job, text, header, version)
            rows.append(
                {
                    "part": part,
                    "noise": noise,
                    "pid": pid,
                    "job": profile.job,
                    "bio_id": profile.id,
                    "match": bool(profile.match),
                    "hard": bool(profile.hard),
                    "bio_gender": profile.gender,
                    "arm": arm,
                    "kind": kind,
                    "group": group or None,
                    "name_gender": gender or None,
                    "first": first,
                    "version": version,
                    "run": run,
                    "messages": messages,
                }
            )

        if pid in noise_pids:
            add("noise", "none", run=1, noise="rerun")
            add("noise", "none", noise="edit", edit=True)
        for arm in llm_profiles.arm_labels(pools):
            add("main", arm)
        for version in PARAPHRASES:
            add("paraphrase", "none", version=version)
    return pd.DataFrame(rows)
