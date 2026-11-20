"""F7: what each mitigation does to the main model's accuracy and gender gap.

Reads `results/mitigation.json`, the test bios outside the showcase subset. Each of the five
conditions is a dot at its macro-F1 (x) and RMS TPR gap (y) with 95% bootstrap intervals in both
directions. Arrows run from the original model (a) to the others. The band is the gap that
shuffled genders give (mean to 95th percentile of the permutation null of the original model).
Condition d, the scrubbed model on scrubbed bios, is in the highlight colour: it is the one the
title is about.
"""

import json
from pathlib import Path

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

FIGURE_NAME = "f7_mitigation"
ORIGINAL = "a"
HEADLINE = "d"
DARK = "#333333"
HEIGHT_INCHES = 4.0
# name, text offset in points, horizontal alignment
LABELS = {
    "a": ("a original", (6, 0), "left"),
    "b": ("b scrubbed bios", (-6, 3), "right"),
    "c": ("c scrubbed model", (7, -4), "left"),
    "d": ("d scrubbed both", (-7, -8), "right"),
    "e": ("e proxy words deleted", (-3, 9), "right"),
}


def _block(report: dict, condition: str) -> dict:
    return report["conditions"][condition]["outside_showcase"]


def _verdict(change: dict) -> int:
    """-1 when the 95% interval of a change is below 0, 1 when above, 0 when it includes 0."""
    if change["ci_high"] < 0:
        return -1
    return 1 if change["ci_low"] > 0 else 0


def title(report: dict) -> str:
    """The finding for condition d: what scrubbing does to the gap and to macro-F1.

    The gap and macro-F1 "fall" or "rise" only when the interval of the paired change excludes 0.
    """
    block = _block(report, HEADLINE)
    gap_change = block["change_from_a"]["rms_tpr_gap"]
    f1_change = block["change_from_a"]["macro_f1"]
    before = _block(report, ORIGINAL)["rms_tpr_gap"]["value"]
    after = block["rms_tpr_gap"]["value"]
    gap = {
        -1: f"cuts the gender gap from {before:.2f} to {after:.2f}",
        0: f"does not change the gender gap ({before:.2f} to {after:.2f})",
        1: f"raises the gender gap from {before:.2f} to {after:.2f}",
    }[_verdict(gap_change)]
    points = abs(f1_change["value"]) * 100
    accuracy = {
        -1: f"for {points:.1f} points of macro-F1",
        0: "and leaves macro-F1 unchanged",
        1: f"and macro-F1 by {points:.1f} points",
    }[_verdict(f1_change)]
    return f"Scrubbing pronouns {gap} {accuracy}"


def build(report: dict):
    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    fig.suptitle(title(report), fontweight="bold", wrap=True, x=0, ha="left")
    points = {c: _block(report, c) for c in LABELS}
    null = points[ORIGINAL]["rms_tpr_gap"]["null"]
    ax.axhspan(null["mean"], null["p95"], color=plotting.NEUTRAL, alpha=0.3, lw=0)
    x_low = min(p["macro_f1"]["ci_low"] for p in points.values())
    x_high = max(p["macro_f1"]["ci_high"] for p in points.values())
    y_high = max(p["rms_tpr_gap"]["ci_high"] for p in points.values())
    span = x_high - x_low
    ax.set_xlim(x_low - 0.3 * span, x_high + 0.5 * span)
    ax.set_ylim(0, y_high * 1.12)
    ax.text(
        x_low - 0.22 * span,
        (null["mean"] + null["p95"]) / 2,
        f"chance alone: {null['mean']:.2f} to {null['p95']:.2f}",
        color=DARK,
        va="center",
        ha="left",
    )
    start = (points[ORIGINAL]["macro_f1"]["value"], points[ORIGINAL]["rms_tpr_gap"]["value"])
    for condition, (label, offset, align) in LABELS.items():
        f1, gap = points[condition]["macro_f1"], points[condition]["rms_tpr_gap"]
        colour = plotting.HIGHLIGHT if condition == HEADLINE else plotting.NEUTRAL
        ax.hlines(gap["value"], f1["ci_low"], f1["ci_high"], colors=colour, lw=1)
        ax.vlines(f1["value"], gap["ci_low"], gap["ci_high"], colors=colour, lw=1)
        ax.scatter([f1["value"]], [gap["value"]], c=colour, zorder=3)
        end = (f1["value"], gap["value"])
        if condition != ORIGINAL:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": colour,
                    "lw": 0.8,
                    "shrinkA": 3,
                    "shrinkB": 3,
                },
            )
        ax.annotate(
            label,
            xy=end,
            xytext=offset,
            textcoords="offset points",
            color=colour if condition == HEADLINE else DARK,
            ha=align,
            va="center",
        )
    ax.annotate(
        "better",
        xy=(0.97, 0.03),
        xytext=(0.80, 0.12),
        xycoords="axes fraction",
        textcoords="axes fraction",
        color=DARK,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 0.8},
    )
    ax.set_xlabel("Macro-F1 (higher is better)")
    ax.set_ylabel("Gender gap: RMS TPR gap (lower is better)")
    return fig


def main(results_dir: Path = RESULTS_DIR):
    report = json.loads((results_dir / "mitigation.json").read_text())
    fig = build(report)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
