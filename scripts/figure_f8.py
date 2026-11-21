"""F8: how the model's yes-rate for qualified candidates changes with the name or the pronouns.

Reads `results/llm_audit.json`. Each dot is the change in the yes-rate of qualified candidates
from the arm with no header, in points, with its 95% interval (bootstrap over bios and first
names). The grey band is the range of changes that nothing meaningful produces: an identical
rerun and a whitespace edit. The top dot, in the highlight colour, is the primary contrast:
women's names minus men's names. The note under the axis gives the share of decision changes
whose reason does not mention name, origin, nationality or gender, or says the decision barely
depends on the name when no name arm changes decisions more than the noise does.
"""

import json
from pathlib import Path

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

FIGURE_NAME = "f8_llm_screener"
DARK = "#333333"
HEIGHT_INCHES = 4.1
NOTE_SHARE = 0.11  # share of the figure height kept free under the axis label for the note


def _points(stat: dict) -> tuple[float, float, float]:
    return 100 * stat["value"], 100 * stat["ci_low"], 100 * stat["ci_high"]


def title(report: dict) -> str:
    """The primary contrast in plain words, a gap only when its interval excludes 0, and then the
    pooled contrast with no name, when its interval excludes 0 too."""
    value, low, high = _points(report["statistics"]["primary"])
    if low > 0 or high < 0:
        more = "more" if value > 0 else "fewer"
        names = (
            f"With a woman's name, {abs(value):.0f} {more} qualified candidates per 100 get yes "
            "than with a man's"
        )
    else:
        names = "Women's and men's names get about the same yes-rate"
    pooled, low, high = _points(report["statistics"]["pooled_names_vs_none"])
    if low > 0 or high < 0:
        names += "; any name " + ("raises" if pooled > 0 else "lowers") + " it"
    return names + "."


def _label(arm: str) -> str:
    if arm == "neutral":
        return "neutral ID header"
    if arm == "swap":
        return "pronouns swapped"
    _, group, gender = arm.split("|")
    return f"{group}, {'woman' if gender == 'female' else 'man'}"


def _note(report: dict) -> str:
    stated = report["rationalization"]["beyond_noise"]
    if stated is None:
        return (
            "No name arm changes decisions more than the noise does: "
            "the decision barely depends on the name."
        )
    share = stated["share_no_mention"]
    low, high = 100 * share["ci_low"], 100 * share["ci_high"]
    return (
        f"Of {stated['n_flips']} changed decisions, a keyword rule finds that "
        f"{100 * share['value']:.0f}% ({low:.0f} to {high:.0f}) give a reason that does not "
        "mention name, origin or gender."
    )


def build(report: dict):
    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    fig.suptitle(title(report), fontweight="bold", wrap=True, x=0, ha="left")
    fig.get_layout_engine().set(rect=(0, NOTE_SHARE, 1, 1 - NOTE_SHARE))  # the note goes below
    arms = {
        key.split("|", 1)[1]: _points(stat)
        for key, stat in report["statistics"].items()
        if key.startswith("diff_tpr|")
    }
    ordered = sorted(arms.items(), key=lambda item: item[1][0])
    floor = [_points(report["noise_floor"][part]["tpr_diff"]) for part in ("rerun", "edit")]
    band_low, band_high = min(f[1] for f in floor), max(f[2] for f in floor)
    ax.axvspan(band_low, band_high, color=plotting.NEUTRAL, alpha=0.3, lw=0)
    primary_y = len(ordered) + 0.8
    rows = [(y, _label(arm), stat, plotting.NEUTRAL) for y, (arm, stat) in enumerate(ordered)]
    primary = _points(report["statistics"]["primary"])
    rows.append((primary_y, "Woman's minus man's name", primary, plotting.HIGHLIGHT))
    for y, _, (value, low, high), colour in rows:
        ax.hlines(y, low, high, colors=colour, lw=1)
        ax.scatter([value], [y], c=colour, zorder=3)
    ax.set_yticks([row[0] for row in rows], [row[1] for row in rows])
    low_edge = min([row[2][1] for row in rows] + [band_low])
    high_edge = max([row[2][2] for row in rows] + [band_high])
    pad = 0.08 * (high_edge - low_edge)
    ax.set_xlim(low_edge - pad, high_edge + pad)
    ax.set_ylim(-0.8, primary_y + 1.4)
    ax.text((band_low + band_high) / 2, primary_y + 0.55, "noise floor", color=DARK, ha="center")
    bottom, top = ax.get_ylim()
    ax.axvline(
        0, ymax=(primary_y + 0.3 - bottom) / (top - bottom), color=DARK, lw=0.8
    )  # stops under the label
    fig.text(0.01, 0.01, _note(report), color=DARK, va="bottom", ha="left", wrap=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in yes-rate (points),\nagainst no name")
    return fig


def main(results_dir: Path = RESULTS_DIR):
    report = json.loads((results_dir / "llm_audit.json").read_text())
    fig = build(report)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
