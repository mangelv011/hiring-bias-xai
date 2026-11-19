"""F1: the main model's TPR gap against the share of women in each occupation."""

import json
from pathlib import Path

import numpy as np

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

HEIGHT_INCHES = 3.2
DARK = "#333333"
LABEL_DX_PT = 5
LABEL_GAP_PT = 9
MALE_RAISE_PT = 10


def _read(results_dir: Path, name: str):
    return json.loads((results_dir / name).read_text())


def _label(fig, ax, points: dict, sides: dict) -> None:
    """Label the showcase occupations to the right of their dots, with no two labels overlapping.

    Male-dominated labels start above their dots, clear of the zero line and the grey dots near
    it. Within a side, labels are pushed up until they are `LABEL_GAP_PT` apart; a label that is
    not level with its dot gets a leader line.
    """
    fig.canvas.draw()  # fixes the axes position so data units can be converted to points
    points_per_pixel = 72 / fig.dpi
    placed = {}
    for side in ("male_dominated", "female_dominated"):
        names = sorted((n for n in sides if sides[n] == side), key=lambda n: points[n][1])
        label_y = -np.inf
        for name in names:
            dot_y = ax.transData.transform(points[name])[1] * points_per_pixel
            label_y = max(dot_y + (MALE_RAISE_PT if side == "male_dominated" else 0), label_y + LABEL_GAP_PT)
            placed[name] = label_y - dot_y
    for name, raise_pt in placed.items():
        ax.annotate(
            name.replace("_", " "),
            points[name],
            xytext=(LABEL_DX_PT, raise_pt),
            textcoords="offset points",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": DARK, "shrinkA": 0, "shrinkB": 2}
            if raise_pt
            else None,
        )


def build(results_dir: Path):
    per_occupation = _read(results_dir, "metrics_deberta.json")["test"]["per_occupation"]
    shares = _read(results_dir, "gap_statistics.json")["main_model"]["occupations"]
    names = list(per_occupation)
    x = np.array([shares[name]["share_women"] for name in names])
    gap = np.array([per_occupation[name]["tpr_gap"]["value"] for name in names])
    low = np.array([per_occupation[name]["tpr_gap"]["ci_low"] for name in names])
    high = np.array([per_occupation[name]["tpr_gap"]["ci_high"] for name in names])

    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    ax.errorbar(x, gap, yerr=[gap - low, high - gap], fmt="none", ecolor=plotting.NEUTRAL, lw=0.6)
    showcase = _read(results_dir, "showcase_occupations.json")
    labelled = {
        row["occupation"]: side
        for side in ("male_dominated", "female_dominated")
        for row in showcase[side]
    }
    is_showcase = np.array([name in labelled for name in names])
    ax.scatter(x[~is_showcase], gap[~is_showcase], s=12, color=plotting.NEUTRAL, zorder=3)
    ax.scatter(x[is_showcase], gap[is_showcase], s=16, color=plotting.HIGHLIGHT, zorder=4)
    ax.axhline(0, color=DARK, lw=0.6, zorder=1)
    slope, intercept = np.polyfit(x, gap, 1)
    ax.plot([x.min(), x.max()], [intercept + slope * x.min(), intercept + slope * x.max()], color=DARK, lw=1)
    pearson = _read(results_dir, "gap_statistics.json")["main_model"]["pearson"]
    fig.suptitle(
        f"In male-dominated occupations the main model misses more women (r = {pearson['value']:.2f}).",
        wrap=True,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        f"r = {pearson['value']:.2f} [{pearson['fisher_ci_low']:.2f}, {pearson['fisher_ci_high']:.2f}]",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel("Share of women in the training subsample")
    ax.set_ylabel("TPR gap (women minus men)")
    ax.set_xlim(right=1.15)  # room for the labels of the female-dominated showcase occupations
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    _label(fig, ax, {name: (xi, yi) for name, xi, yi in zip(names, x, gap, strict=True)}, labelled)
    return fig


def main(results_dir: Path = RESULTS_DIR):
    fig = build(results_dir)
    plotting.save(fig, "f1_gap_vs_share", results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
