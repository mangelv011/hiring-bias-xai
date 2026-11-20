"""F3: the words that make up the main model's score gap, in two showcase occupations.

One panel for a male-dominated and one for a female-dominated showcase occupation. The rule,
fixed before looking at the figure: in each kind, the showcase occupation with the largest
absolute score gap of the main model (mean score of women's bios minus men's, `score_gap` in
`results/gap_explained.json`). Each panel shows the ten words with the largest absolute
contribution to that gap, with their 95% bootstrap intervals.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd

from hiring_bias_xai import plotting, results
from hiring_bias_xai.config import RESULTS_DIR

DARK = "#333333"
ROW_INCHES = 0.17
PANEL_EXTRA_INCHES = 0.9


class Panel(NamedTuple):
    occupation: str
    kind: str  # "male_dominated" or "female_dominated"
    words: pd.DataFrame  # the top rows of `gap_decomposition.csv` for this occupation
    summary: dict  # the main model's entry in `gap_explained.json`


KINDS = ("male_dominated", "female_dominated")


def _name(occupation: str) -> str:
    return occupation.replace("_", " ")


def title(panels: Sequence[Panel]) -> str:
    """The finding as one sentence: how far the gendered words put women's bios below or above
    men's in each panel, from the summed gendered contribution (`gendered_sum`).

    The sum needs no division, so it is stated even when the total gap includes 0. A panel whose
    sum has an interval that includes 0 is said to be within noise.
    """
    clauses, reference = [], " men's"  # named once, to keep the title at two lines
    for panel in panels:
        total = panel.summary["gendered_sum"]
        if total["ci_low"] <= 0 <= total["ci_high"]:
            clauses.append(f"within noise in {_name(panel.occupation)}")
        else:
            side = "above" if total["value"] > 0 else "below"
            clauses.append(
                f"{abs(total['value']):.1f} {side}{reference} in {_name(panel.occupation)}"
            )
            reference = ""
    return f"Gendered words put women's bios {' and '.join(clauses)}."


def _interval(share: dict) -> str:
    return f"{share['value']:.0%} [{share['ci_low']:.0%}, {share['ci_high']:.0%}]"


def header(panel: Panel) -> str:
    """The occupation with its total score gap, then the shares or why there are none.

    Each line is short because the header starts at the axes and must end inside the figure.
    """
    gap = panel.summary["score_gap"]
    first = (
        f"{_name(panel.occupation).capitalize()} ({panel.kind.replace('_', '-')}), "
        f"gap {gap['value']:+.1f} [{gap['ci_low']:+.1f}, {gap['ci_high']:+.1f}]"
    )
    total = panel.summary["gendered_sum"]
    first += (
        f"\ngendered words: {total['value']:+.2f} [{total['ci_low']:+.2f}, {total['ci_high']:+.2f}]"
    )
    if panel.summary["gap_unstable"]:
        return first + "\ngap interval includes 0: no shares given"
    return (
        f"{first}\ntop {len(panel.summary['top_words'])} words: "
        f"{_interval(panel.summary['top_share'])}\n"
        f"gendered share: {_interval(panel.summary['gendered_share'])}"
    )


def build(panels: Sequence[Panel]):
    """One panel per occupation: its top words as dots, sorted by contribution, largest on top."""
    plt.rcParams.update(plotting.STYLE)
    height = sum(len(p.words) * ROW_INCHES + PANEL_EXTRA_INCHES for p in panels)
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(plotting.WIDTH_INCHES, height), layout="constrained", squeeze=False
    )
    fig.suptitle(title(panels), fontweight="bold", wrap=True, x=0, ha="left")
    for ax, panel in zip(axes[:, 0], panels, strict=True):
        words = panel.words.sort_values("contribution", ascending=False, kind="stable")
        rows = range(len(words))
        ax.set_title(header(panel), loc="left")
        ax.set_xlabel("Contribution to the women-minus-men gap (log-odds)")
        ax.axvline(0, color=DARK, lw=0.6)
        colours = [plotting.HIGHLIGHT if g else plotting.NEUTRAL for g in words["gendered"]]
        ax.hlines(rows, words["contribution_ci_low"], words["contribution_ci_high"], colors=colours)
        ax.scatter(words["contribution"], rows, c=colours, zorder=3)
        ax.set_yticks(rows)
        ax.set_yticklabels(words["word"])
        for tick, gendered in zip(ax.get_yticklabels(), words["gendered"], strict=True):
            tick.set_color(plotting.HIGHLIGHT if gendered else DARK)
        ax.set_ylim(len(words) - 0.5, -0.5)
    return fig


SCREENER = "deberta"
FIGURE_NAME = "f3_gap_words"
RULE = (
    "In each of the two kinds of showcase occupation (male-dominated, female-dominated), the "
    "occupation with the largest absolute score gap of the main model; ties go to the "
    "occupation listed first."
)


def pick(summaries: Mapping[str, dict], showcase: Mapping[str, Sequence[dict]]) -> dict:
    """For each kind of showcase occupation, the one with the largest absolute score gap.

    `summaries` maps occupation to the main model's summary in `gap_explained.json`; a tie goes
    to the occupation listed first.
    """
    return {
        kind: max(
            (row["occupation"] for row in showcase[kind]),
            key=lambda occupation: abs(summaries[occupation]["score_gap"]["value"]),
        )
        for kind in KINDS
    }


def main(results_dir: Path = RESULTS_DIR):
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    explained = json.loads((results_dir / "gap_explained.json").read_text())
    summaries = explained["screeners"][SCREENER]
    picked = pick(summaries, showcase)
    table = pd.read_csv(results_dir / "gap_decomposition.csv")
    table = table[table["screener"] == SCREENER]
    panels = [
        Panel(
            picked[kind],
            kind,
            table[(table["occupation"] == picked[kind]) & table["top"]],
            summaries[picked[kind]],
        )
        for kind in KINDS
    ]
    fig = build(panels)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    results.write_json(
        results_dir / "f3_panels.json", {"rule": RULE, "picked": picked, "title": title(panels)}
    )
    return fig


if __name__ == "__main__":
    main()
