"""F5: do the Shapley explanations hold up? Deletion curves and stability of the main model.

Panel a: the main model's deletion curves from `results/deletion_curves.csv`: the fall in the
score of the true occupation as words are removed most important first (highlight), at
random (grey, with its 95% interval over bios) and least important first. Panel b: the
median Spearman correlation, with its 95% bootstrap interval, of the
checks in `results/stability.json`, sorted by the correlation. Each row says what changed
from the main run (Shapley, delete removal, seed 42).
The title states the finding: whether the deletion check of `results/deletion_areas.json` holds,
and whether another way of computing the values moves the ranking further than a rerun with
another seed does (their correlation intervals do not overlap). The numbers are in the panels.
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
HEIGHT_INCHES = 4.0
LAST_STEP = 0.5  # the deletion curves stop at half of the words
LABEL_ROOM = 0.22  # room right of the last step for the curve labels, in fractions of the bio
LABEL_GAP = 0.13  # least distance between curve labels, as a share of the y range
LABEL_OFFSET_PT = 8  # gap between a dot and its value, in points
CURVES = (
    ("morf", HIGHLIGHT, "-", "highest value\nfirst"),
    ("random", NEUTRAL, "-", "random"),
    ("lerf", LERF, "--", "lowest value\nfirst"),
)
CHECK_NAMES = {
    "seed": "same method,\nanother seed",
    "removal": "a blank instead\nof deleting",
    "method": "leave-one-out,\nnot Shapley",
    "randomisation": "every weight\nscrambled",
}
OTHER_WAYS = ("removal", "method")  # compared with the seed check, the noise of a rerun


def title(areas: dict, stability: dict) -> str:
    """The finding: whether the top-ranked words matter to the model, and whether computing the
    values another way changes their order more than rerunning with another seed does."""
    deletion_passed = areas["screeners"][SCREENER]["passed"]
    checks = stability["checks"]
    rerun_low = checks["seed"]["spearman"]["median_ci"][0]
    depends = any(checks[name]["spearman"]["median_ci"][1] < rerun_low for name in OTHER_WAYS)
    lead = (
        "Shapley finds the words the model relies on"
        if deletion_passed
        else "Shapley's top words matter no more than random ones"
    )
    joiner = "but" if deletion_passed and depends else "and"
    order = (
        "their order depends on how it is computed"
        if depends
        else "their order barely depends on the method"
    )
    return f"{lead}, {joiner} {order}."


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
    label_y = plotting.spread([y for _, y in ends], LABEL_GAP * span)
    for (_, colour, _, label), (x, _), y in zip(CURVES, ends, label_y, strict=True):
        ax.text(x + 0.015, y, label, color=colour, va="center", ha="left")
    ax.set_xticks([0, 0.25, LAST_STEP])
    ax.set_xlim(0, LAST_STEP + LABEL_ROOM)
    ax.spines["bottom"].set_bounds(0, LAST_STEP)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("Words removed (share of the bio)")
    ax.set_ylabel("Fall in the true job's score\n(score points)")
    ax.set_title("a  Deletion curves, main model", loc="left", fontweight="bold")


def _agreement_panel(ax, stability: dict) -> None:
    checks = {**stability["checks"], "randomisation": stability["randomisation"]}
    names = sorted(checks, key=lambda name: -checks[name]["spearman"]["median"])
    rows = list(range(len(names)))
    ax.axvline(0, color=DARK, lw=0.6)
    values = [checks[name]["spearman"]["median"] for name in names]
    bounds = [checks[name]["spearman"]["median_ci"] for name in names]
    ax.hlines(rows, [b[0] for b in bounds], [b[1] for b in bounds], colors=HIGHLIGHT)
    ax.scatter(values, rows, c=HIGHLIGHT, zorder=3)
    for y, value, bound in zip(rows, values, bounds, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (max(value, bound[1]), y),
            xytext=(LABEL_OFFSET_PT, 0),
            textcoords="offset points",
            color=HIGHLIGHT,
            va="center",
            ha="left",
        )
    ax.set_yticks(rows)
    ax.set_yticklabels([CHECK_NAMES[name] for name in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(min(0, min(b[0] for b in bounds)) - 0.15, 1.3)
    ax.set_xticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    ax.set_xlabel("Rank correlation with the main run\n(median over bios, 95% interval)")
    ax.set_title("b  Do the rankings agree?", loc="left", fontweight="bold")


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
