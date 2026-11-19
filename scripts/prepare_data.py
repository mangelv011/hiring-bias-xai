"""Build the data splits and write the small result files to results/."""

from pathlib import Path

import pandas as pd

from hiring_bias_xai import data, results
from hiring_bias_xai.config import RESULTS_DIR


def _counts(df: pd.DataFrame) -> dict:
    table = df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
    return {
        occupation: {"female": int(row.get("female", 0)), "male": int(row.get("male", 0))}
        for occupation, row in table.iterrows()
    }


def main(results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": data.load_split("train"),
        "dev": data.load_split("dev"),
        "test": data.load_split("test"),
    }
    showcase = data.showcase_occupations(frames["train"], frames["test"])
    names = {
        entry["occupation"]
        for side in ("male_dominated", "female_dominated")
        for entry in showcase[side]
    }
    frames["showcase_subset"] = data.showcase_subset(frames["test"], sorted(names))
    ids = "".join(f"{bio_id}\n" for bio_id in frames["showcase_subset"]["id"])
    (results_dir / "showcase_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    counts = {name: _counts(df) for name, df in frames.items()}
    results.write_json(
        results_dir / "data_summary.json",
        {
            "bios": {name: len(df) for name, df in frames.items()},
            "counts": counts,
            "share_women_train": {
                occupation: round(share, 4)
                for occupation, share in data.share_women(frames["train"]).items()
            },
        },
    )
    results.write_json(results_dir / "showcase_occupations.json", showcase)


if __name__ == "__main__":
    main()
