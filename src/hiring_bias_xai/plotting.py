"""Figure helpers shared by the scripts: one style, semantic colours and `save`.

Colours are Okabe-Ito values and mean the same thing in every figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hiring_bias_xai.config import FIGURES_DIR

DPI = 300
WIDTH_INCHES = 3.03  # one column of the report
MIN_FONT_PT = 7

WOMEN = "#CC79A7"
MEN = "#0072B2"
HIGHLIGHT = "#D55E00"
NEUTRAL = "#999999"

STYLE = {
    "font.size": MIN_FONT_PT,
    "axes.titlesize": MIN_FONT_PT,
    "figure.titlesize": MIN_FONT_PT,
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


# Machado, Oliveira & Fernandes (2009), deuteranopia at severity 1.0, applied to linear RGB.
DEUTERANOPIA = np.array(
    [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]
)


def simulate_deuteranopia(image: np.ndarray) -> np.ndarray:
    """An 8-bit sRGB image (RGB or RGBA) as a deuteranope sees it; any alpha channel is kept."""
    srgb = image[..., :3] / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    seen = np.clip(linear @ DEUTERANOPIA.T, 0, 1)
    encoded = np.where(seen <= 0.0031308, seen * 12.92, 1.055 * seen ** (1 / 2.4) - 0.055)
    out = image.copy()
    out[..., :3] = np.round(encoded * 255)
    return out


def job_name(occupation: str) -> str:
    """How a figure spells a job: spaces for underscores, and "fashion model" for `model`."""
    return "fashion model" if occupation == "model" else occupation.replace("_", " ")


def spread(positions: list[float], gap: float) -> list[float]:
    """`positions` moved apart so that neighbours are at least `gap` apart, order kept."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    moved = list(positions)
    for previous, current in zip(order, order[1:], strict=False):
        moved[current] = max(moved[current], moved[previous] + gap)
    return moved


def new_figure(height: float):
    """A single-column figure of the given height in inches, in the project style: (fig, ax)."""
    plt.rcParams.update(STYLE)
    return plt.subplots(figsize=(WIDTH_INCHES, height), layout="constrained")


def save(fig, name: str, directory: Path = FIGURES_DIR) -> None:
    """Write `<name>.pdf` and `<name>.png` (300 dpi) to `directory`, creating it if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        metadata = {"CreationDate": None} if extension == "pdf" else None  # byte-identical reruns
        fig.savefig(directory / f"{name}.{extension}", dpi=DPI, metadata=metadata)
