"""Download Bias in Bios, build the subsamples and write the small result files to results/.

Writes xai_subset_ids.csv, data_summary.json and showcase_occupations.json. Output is
deterministic: the same revision and seed give byte-identical files.
"""

import json

import pandas as pd

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR


def _counts(df: pd.DataFrame) -> dict:
    table = df.groupby(["occupation", "gender"]).size().unstack(fill_value=0)
    return {
        occupation: {gender: int(row[gender]) for gender in ("female", "male")}
        for occupation, row in table.iterrows()
    }


def _write_json(name: str, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    (RESULTS_DIR / name).write_text(text + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    splits = {split: data.load_split(split) for split in ("train", "dev", "test")}
    xai = data.xai_subset()

    xai[["id"]].to_csv(RESULTS_DIR / "xai_subset_ids.csv", index=False, lineterminator="\n")

    train = splits["train"]
    share_women = (train["gender"] == "female").groupby(train["occupation"]).mean()
    _write_json(
        "data_summary.json",
        {
            "bios": {**{split: len(df) for split, df in splits.items()}, "xai_subset": len(xai)},
            "counts": {
                **{split: _counts(df) for split, df in splits.items()},
                "xai_subset": _counts(xai),
            },
            "share_women_train": {o: round(float(s), 4) for o, s in share_women.items()},
        },
    )
    _write_json("showcase_occupations.json", data.showcase_occupations(train, splits["test"]))


if __name__ == "__main__":
    main()
