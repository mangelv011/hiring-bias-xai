"""Reading and writing the results that scripts hand to each other."""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai.config import artifacts_dir


def write_json(path: Path, content: dict) -> None:
    """Write `content` as JSON with 2-space indent, sorted keys and one trailing newline."""
    path.write_bytes((json.dumps(content, indent=2, sort_keys=True) + "\n").encode())


def _predictions_path(screener: str) -> Path:
    return artifacts_dir() / "predictions" / f"{screener}_test.csv"


def save_predictions(screener: str, test: pd.DataFrame, predicted) -> None:
    """Save the occupations `screener` predicted for the bios of `test` as artifacts."""
    path = _predictions_path(screener)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"], "predicted_occupation": predicted}).to_csv(path, index=False)


def load_predictions(screener: str, test: pd.DataFrame) -> pd.Series:
    """The saved predictions of `screener`, one per bio of `test`, in its order and index.

    Raises if a bio has no prediction or more than one.
    """
    saved = pd.read_csv(_predictions_path(screener))
    aligned = test[["id"]].merge(saved, on="id", how="left", validate="one_to_one")
    if aligned["predicted_occupation"].isna().any():
        raise ValueError(f"{screener}: some test bios have no prediction")
    return aligned["predicted_occupation"].set_axis(test.index)
