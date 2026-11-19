"""Figure helpers shared by the scripts."""

from pathlib import Path

from hiring_bias_xai.config import FIGURES_DIR

DPI = 300


def save(fig, name: str, directory: Path = FIGURES_DIR) -> None:
    """Write `<name>.pdf` and `<name>.png` (300 dpi) to `directory`, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", dpi=DPI)
