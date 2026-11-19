"""Figure helpers shared by the scripts: one style, semantic colours and `save`.

Colours are Okabe-Ito values and mean the same thing in every figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from hiring_bias_xai.config import FIGURES_DIR

DPI = 300
WIDTH_INCHES = 3.3  # one column
MIN_FONT_PT = 7

WOMEN = "#CC79A7"
MEN = "#0072B2"
HIGHLIGHT = "#D55E00"
NEUTRAL = "#999999"

STYLE = {
    "font.size": MIN_FONT_PT,
    "axes.titlesize": MIN_FONT_PT,
    "axes.labelsize": MIN_FONT_PT,
    "xtick.labelsize": MIN_FONT_PT,
    "ytick.labelsize": MIN_FONT_PT,
    "legend.fontsize": MIN_FONT_PT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
}


def new_figure(height: float):
    """A single-column figure of the given height in inches, in the project style: (fig, ax)."""
    plt.rcParams.update(STYLE)
    return plt.subplots(figsize=(WIDTH_INCHES, height), layout="constrained")


def save(fig, name: str, directory: Path = FIGURES_DIR) -> None:
    """Write `<name>.pdf` and `<name>.png` (300 dpi) to `directory`, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", dpi=DPI)
