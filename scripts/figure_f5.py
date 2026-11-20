"""F5: do the Shapley explanations hold up? Deletion curves and stability of the main model.

Panel a: the main model's deletion curves from `results/deletion_curves.csv`: the fall in the
log-odds of the true occupation as words are removed most important first (MoRF, highlight), at
random (grey, with its 95% interval over bios) and least important first (LeRF). Panel b: the
median Spearman correlation and the median top-10 overlap, with 95% bootstrap intervals, of the
three checks in `results/stability.json`, sorted by the correlation, each marked passed or failed,
with a mark at each measure's pass threshold.
The title states the finding from the pass flags of `results/deletion_areas.json` and
`results/stability.json`; the numbers are in the panels.
"""

import json
from pathlib import Path

import pandas as pd

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.plotting import HIGHLIGHT, NEUTRAL

SCREENER = "deberta"
FIGURE_NAME = "f5_explanation_checks"
DARK = "#333333"
LERF = "#666666"  # a mid grey: darker than random's NEUTRAL, with its own dashed line
HEIGHT_INCHES = 5.2
LAST_STEP = 0.5  # the deletion curves stop at half of the words
LABEL_ROOM = 0.22  # room right of the last step for the curve labels, in fractions of the bio
LABEL_GAP = 0.13  # least distance between curve labels, as a share of the y range
NAME_X = 1.03  # the two measures are named right of every dot and mark
LABEL_OFFSET_PT = 8  # gap between a dot and its value, in points
OFFSET = 0.18  # distance of each dot from the row's centre line
CURVES = (
    ("morf", HIGHLIGHT, "-", "highest first\n(MoRF)"),
    ("random", NEUTRAL, "-", "random"),
    ("lerf", LERF, "--", "lowest first\n(LeRF)"),
)
CHECK_NAMES = {
    "seed": "seed 7 vs 42",
    "removal": "[UNK] vs delete",
    "method": "leave-one-out vs Shapley",
}


def title(areas: dict, stability: dict) -> str:
    """The finding: whether the top-ranked words matter to the model, and whether the rankings of
    single words are stable. The numbers are in the panels."""
    deletion_passed = areas["screeners"][SCREENER]["passed"]
    flags = [check["passed"] for check in stability["checks"].values()]
    lead = (
        "Shapley finds the words the model relies on"
        if deletion_passed
        else "Shapley's top words matter no more than random ones"
    )
    joiner = "but" if deletion_passed and not all(flags) else "and"
    if all(flags):
        rankings = "its single-word rankings hold up"
    elif not any(flags):
        rankings = "its single-word rankings are unstable"
    else:
        rankings = f"its single-word rankings hold up in {sum(flags)} of {len(flags)} checks"
    return f"{lead}, {joiner} {rankings}."


def _spread(positions: list[float], gap: float) -> list[float]:
    """`positions` moved apart so that neighbours are at least `gap` apart, order kept."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    spread = list(positions)
    for previous, current in zip(order, order[1:], strict=False):
        spread[current] = max(spread[current], spread[previous] + gap)
    return spread


def _deletion_panel(ax, curves: pd.DataFrame) -> None:
    main = curves[curves["screener"] == SCREENER]
    ax.plot([0, LAST_STEP], [0, 0], color=DARK, lw=0.6, label="zero")
    ends = []
    for order, colour, style, _ in CURVES:
        rows = main[main["order"] == order].sort_values("fraction")
        ax.fill_between(
            rows["fraction"],
            rows["drop_ci_low"],
            rows["drop_ci_high"],
            color=colour,
            alpha=0.25,
            lw=0,
        )
        ax.plot(
            rows["fraction"], rows["mean_logodds_drop"], color=colour, ls=style, lw=1.2, label=order
        )
        ends.append((rows["fraction"].iloc[-1], rows["mean_logodds_drop"].iloc[-1]))
    span = max(y for _, y in ends) - min(0, min(y for _, y in ends))
    label_y = _spread([y for _, y in ends], LABEL_GAP * span)
    for (_, colour, _, label), (x, _), y in zip(CURVES, ends, label_y, strict=True):
        ax.text(x + 0.015, y, label, color=colour, va="center", ha="left")
    ax.set_xticks([0, 0.25, LAST_STEP])
    ax.set_xlim(0, LAST_STEP + LABEL_ROOM)
    ax.spines["bottom"].set_bounds(0, LAST_STEP)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Words removed (share of the bio)")
    ax.set_ylabel("Fall in true-class log-odds")
    ax.set_title("a  Deletion curves, main model", loc="left", fontweight="bold")


def _agreement_panel(ax, stability: dict) -> None:
    checks = stability["checks"]
    names = sorted(checks, key=lambda name: -checks[name]["spearman"]["median"])
    rows = list(range(len(names)))
    ax.axvline(0, color=DARK, lw=0.6)
    for metric, colour, shift, label in (
        ("spearman", HIGHLIGHT, -OFFSET, "rank correlation"),
        ("top10_overlap", NEUTRAL, OFFSET, "top-10 overlap"),
    ):
        ys = [row + shift for row in rows]
        values = [checks[name][metric]["median"] for name in names]
        bounds = [checks[name][metric]["median_ci"] for name in names]
        ax.hlines(ys, [b[0] for b in bounds], [b[1] for b in bounds], colors=colour)
        ax.scatter(values, ys, c=colour, zorder=3)
        thresholds = [checks[name]["thresholds"][f"median_{metric}"] for name in names]
        ax.plot(thresholds, ys, ls="none", marker="|", color=DARK, markersize=8, markeredgewidth=1)
        for y, value, bound, threshold in zip(ys, values, bounds, thresholds, strict=True):
            below = value < threshold  # the mark is to the right of the dot, so the value goes left
            ax.annotate(
                f"{value:.2f}",
                (min(value, bound[0]) if below else max(value, bound[1]), y),
                xytext=(-LABEL_OFFSET_PT if below else LABEL_OFFSET_PT, 0),
                textcoords="offset points",
                color=colour,
                va="center",
                ha="right" if below else "left",
            )
        ax.text(NAME_X, ys[-1], label, color=colour, va="center", ha="left")
    ax.set_yticks(rows)
    ax.set_yticklabels(
        [
            f"{CHECK_NAMES[name]}\n{'passed' if checks[name]['passed'] else 'failed'}"
            for name in names
        ]
    )
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(min(0, min(checks[n]["spearman"]["median_ci"][0] for n in names)) - 0.15, 1.9)
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    ax.set_xlabel("Median over bios, 95% interval\n| marks the pass threshold")
    ax.set_title("b  Stability and agreement", loc="left", fontweight="bold")


def build(curves: pd.DataFrame, areas: dict, stability: dict):
    """Panel a above panel b, under a title that states the finding."""
    fig, probe = plotting.new_figure(HEIGHT_INCHES)
    probe.remove()
    top, bottom = fig.subplots(2, 1, height_ratios=[1.15, 1])
    fig.suptitle(
        title(areas, stability),
        fontsize=plotting.MIN_FONT_PT,
        fontweight="bold",
        wrap=True,
        x=0,
        ha="left",
    )
    _deletion_panel(top, curves)
    _agreement_panel(bottom, stability)
    return fig


def main(results_dir: Path = RESULTS_DIR):
    curves = pd.read_csv(results_dir / "deletion_curves.csv")
    areas = json.loads((results_dir / "deletion_areas.json").read_text())
    stability = json.loads((results_dir / "stability.json").read_text())
    fig = build(curves, areas, stability)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
