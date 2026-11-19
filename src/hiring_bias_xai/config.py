"""Global settings: artifacts directory, random seed and results paths."""

import os
from pathlib import Path

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def artifacts_dir() -> Path:
    """Directory for data and model weights: HBX_ARTIFACTS, else ~/.cache/hiring-bias-xai."""
    return Path(os.environ.get("HBX_ARTIFACTS", "~/.cache/hiring-bias-xai")).expanduser()
