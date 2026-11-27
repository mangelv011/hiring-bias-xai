"""F1: the main model's TPR gap against the share of women in each occupation.

Written for a reader who has never seen a TPR: the y axis is how many more qualified women than
men the model shortlists for the recruiter, per 100 of each. Only two familiar jobs are
labelled, with their numbers. All 28 dots stay, since r is computed over them; only the six
showcase jobs, which the explanations use, are highlighted and carry their 95% interval. The title
states the finding in plain words when the Fisher interval of r excludes 0."""

import json
from pathlib import Path

import matplotlib.text
import numpy as np
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Bbox

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

HEIGHT_INCHES = 3.2
DARK = "#333333"
LABEL_RADII_PT = (6, 12, 20, 30, 42, 56)
LEADER_FROM_PT = 9
LEADER_COST = 50
DISTANCE_COST = 0.05
DOT_COST, DOT_PAD_PX, LINE_PAD_PX = 50, 4, 2
BAR_COST, FIT_COST, ZERO_COST = 10, 40, 5
TAKEN_COST = OUTSIDE_COST = 1000
EXAMPLES = ("surgeon", "nurse")  # familiar jobs whose labels carry their numbers per 100
FINDING = "The more a job is held by one gender, the more the model favours that gender."
NO_LINK = "No clear link between who holds a job and whom the model favours."


def _read(results_dir: Path, name: str):
    return json.loads((results_dir / name).read_text())


def _inside(box, points, pad=0.0) -> int:
    """How many of the (n, 2) pixel points lie inside the box (x0, y0, x1, y1), grown by `pad`."""
    x, y = points[:, 0], points[:, 1]
    return int(
        ((x > box[0] - pad) & (x < box[2] + pad) & (y > box[1] - pad) & (y < box[3] + pad)).sum()
    )


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _label(fig, ax, points: dict, obstacles: dict, texts: dict) -> None:
    """Label each occupation of `texts` at the nearest clear spot around its dot.

    Each label is tried at several distances and directions from its dot. The spot with the least
    overlap with dots, error bars, the fitted line, the zero line, texts already in the axes,
    earlier labels and the axes edge wins; ties go to the nearest. A label away from its dot gets
    a leader line.
    `obstacles` maps "dots" to an (n, 2) array and "bars", "fit" and "zero" to lists of (n, 2)
    arrays sampled along them, all in data coordinates.
    """
    fig.canvas.draw()  # fixes the axes position and gives the text renderer
    renderer = fig.canvas.get_renderer()
    pixels_per_point = fig.dpi / 72
    to_px = ax.transData.transform
    dots = to_px(obstacles["dots"])
    lines = [
        (cost, to_px(line))
        for kind, cost in (("bars", BAR_COST), ("fit", FIT_COST), ("zero", ZERO_COST))
        for line in obstacles[kind]
    ]
    axes = ax.get_window_extent(renderer)
    taken = [tuple(t.get_window_extent(renderer).extents) for t in ax.texts]  # notes and r
    leaders = []
    offsets = {}
    for name in sorted(texts, key=lambda name: len(texts[name]), reverse=True):
        text = ax.text(0, 0, texts[name])
        extent = matplotlib.text.Text.get_window_extent(text, renderer)
        text.remove()
        dot = to_px([points[name]])[0]
        best = None
        for radius in LABEL_RADII_PT:
            for angle in np.radians(np.arange(0, 360, 30)):
                dx, dy = radius * np.cos(angle), radius * np.sin(angle)
                left = dot[0] + dx * pixels_per_point
                if dx < 0:
                    left -= extent.width
                bottom = dot[1] + dy * pixels_per_point - extent.height / 2
                box = (left, bottom, left + extent.width, bottom + extent.height)
                cost = radius * DISTANCE_COST + DOT_COST * _inside(box, dots, DOT_PAD_PX)
                cost += sum(c for c, line in lines if _inside(box, line, LINE_PAD_PX))
                leader = None
                if radius > LEADER_FROM_PT:
                    end = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    leader = PlotPath([dot, end])
                    along = dot + np.linspace(0, 1, 30)[:, None] * (end - dot)
                    gaps = np.linalg.norm(dots[:, None, :] - along[None], axis=2).min(axis=1)
                    cost += DOT_COST * int(((gaps < DOT_PAD_PX) & (gaps > 0.5)).sum())
                    cost += LEADER_COST * sum(
                        leader.intersects_bbox(Bbox.from_extents(*other), filled=False)
                        for other in taken
                    )
                cost += TAKEN_COST * sum(_overlap(box, other) for other in taken)
                cost += LEADER_COST * sum(
                    line.intersects_bbox(Bbox.from_extents(*box), filled=False) for line in leaders
                )
                cost += OUTSIDE_COST * (
                    box[0] < axes.x0 or box[2] > axes.x1 or box[1] < axes.y0 or box[3] > axes.y1
                )
                if best is None or cost < best[0]:
                    best = (cost, (dx, dy), box, leader)
        _, offsets[name], box, leader = best
        taken.append(box)
        if leader is not None:
            leaders.append(leader)
    for name, (dx, dy) in offsets.items():
        label = ax.annotate(
            texts[name],
            points[name],
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            linespacing=1.1,
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": DARK, "shrinkA": 0, "shrinkB": 2}
            if np.hypot(dx, dy) > LEADER_FROM_PT
            else None,
        )
        label.set_in_layout(False)  # a label must not move the axes it was placed against


def build(results_dir: Path):
    per_occupation = _read(results_dir, "metrics_deberta.json")["test"]["per_occupation"]
    shares = _read(results_dir, "gap_statistics.json")["main_model"]["occupations"]
    names = list(per_occupation)
    x = np.array([shares[name]["share_women"] for name in names])
    gap = np.array([per_occupation[name]["tpr_gap"]["value"] for name in names])
    low = np.array([per_occupation[name]["tpr_gap"]["ci_low"] for name in names])
    high = np.array([per_occupation[name]["tpr_gap"]["ci_high"] for name in names])

    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    showcase = _read(results_dir, "showcase_occupations.json")
    shown = {
        row["occupation"]
        for side in ("male_dominated", "female_dominated")
        for row in showcase[side]
    }
    is_showcase = np.array([name in shown for name in names])
    ax.errorbar(
        x[is_showcase],
        gap[is_showcase],
        yerr=[(gap - low)[is_showcase], (high - gap)[is_showcase]],
        fmt="none",
        ecolor=plotting.NEUTRAL,
        lw=0.6,
    )
    ax.scatter(x[~is_showcase], gap[~is_showcase], s=12, color=plotting.NEUTRAL, zorder=3)
    ax.scatter(x[is_showcase], gap[is_showcase], s=16, color=plotting.HIGHLIGHT, zorder=4)
    ax.axhline(0, color=DARK, lw=0.6, zorder=1)
    slope, intercept = np.polyfit(x, gap, 1)
    fit_x = np.linspace(x.min(), x.max(), 200)
    ax.plot(fit_x[[0, -1]], intercept + slope * fit_x[[0, -1]], color=DARK, lw=1)
    pearson = _read(results_dir, "gap_statistics.json")["main_model"]["pearson"]
    r = pearson["value"]
    fig.suptitle(
        FINDING if pearson["fisher_ci_low"] > 0 else NO_LINK,
        wrap=True,
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.03,
        f"r = {r:.2f} [{pearson['fisher_ci_low']:.2f}, {pearson['fisher_ci_high']:.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=DARK,
    )
    # the lower note sits right of the low-share dots of the male-dominated jobs
    for note_x, y, note, va in (
        (0.03, 0.97, "↑ favours women", "top"),
        (0.22, 0.03, "↓ favours men", "bottom"),
    ):
        ax.text(note_x, y, note, transform=ax.transAxes, va=va, color=DARK)
    ax.set_xlabel("Share of women in the job")
    ax.set_ylabel("Qualified women minus qualified men\nshortlisted, per 100")
    ax.set_xlim(right=1.15)  # room for the labels of the female-dominated showcase occupations
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(
        lambda value, _: "0" if round(value * 100) == 0 else f"{value * 100:+.0f}"
    )
    obstacles = {
        "dots": np.column_stack([x, gap]),
        "bars": [
            np.column_stack([np.full(20, xi), np.linspace(lo, hi, 20)])
            for xi, lo, hi in zip(x[is_showcase], low[is_showcase], high[is_showcase], strict=True)
        ],
        "fit": [np.column_stack([fit_x, intercept + slope * fit_x])],
        "zero": [np.column_stack([fit_x, np.zeros(len(fit_x))])],
    }
    points = {name: (xi, yi) for name, xi, yi in zip(names, x, gap, strict=True)}
    texts = {}
    for name in EXAMPLES:
        if name in per_occupation:
            women = round(100 * per_occupation[name]["tpr_female"]["value"])
            men = round(100 * per_occupation[name]["tpr_male"]["value"])
            texts[name] = (
                f"{plotting.job_name(name)}: {women} of 100 women,\n{men} of 100 men shortlisted"
            )
    _label(fig, ax, points, obstacles, texts)
    return fig


def main(results_dir: Path = RESULTS_DIR):
    fig = build(results_dir)
    plotting.save(fig, "f1_gap_vs_share", results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
