"""Build the data splits and write the small result files to results/."""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR


def _counts(df: pd.DataFrame) -> dict:
    table = df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
    return {
        occupation: {"female": int(row.get("female", 0)), "male": int(row.get("male", 0))}
        for occupation, row in table.iterrows()
    }


def _write_json(path: Path, content: dict) -> None:
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": data.load_split("train"),
        "dev": data.load_split("dev"),
        "test": data.load_split("test"),
        "xai_subset": data.xai_subset(),
    }
    ids = "".join(f"{bio_id}\n" for bio_id in frames["xai_subset"]["id"])
    (results_dir / "xai_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    counts = {name: _counts(df) for name, df in frames.items()}
    _write_json(
        results_dir / "data_summary.json",
        {
            "bios": {name: len(df) for name, df in frames.items()},
            "counts": counts,
            "share_women_train": {
                occupation: round(women["female"] / (women["female"] + women["male"]), 4)
                for occupation, women in counts["train"].items()
            },
        },
    )


if __name__ == "__main__":
    main()
