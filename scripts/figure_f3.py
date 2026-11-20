"""F3: the words that make up the main model's score gap, in two showcase occupations.

One panel for a male-dominated and one for a female-dominated showcase occupation. The rule,
fixed before looking at the figure: in each kind, the showcase occupation with the largest
absolute score gap of the main model (mean score of women's bios minus men's, `score_gap` in
`results/gap_explained.json`). Each panel shows the ten words with the largest absolute
contribution to that gap, with their 95% bootstrap intervals.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import pandas as pd

from hiring_bias_xai import plotting

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
    """The finding as one sentence, with every number from the panels' summaries.

    A panel whose total gap has an interval that includes 0 has no stable share, so the title
    says its gap is too close to 0 to split instead of quoting one.
    """
    stable = [p for p in panels if not p.summary["gap_unstable"]]
    unstable = [p for p in panels if p.summary["gap_unstable"]]
    clauses = [
        f"{p.summary['gendered_share']['value']:.0%} of the {p.summary['score_gap']['value']:+.1f} "
        f"{_name(p.occupation)}-bio gap"
        for p in stable
    ]
    small = " and ".join(_name(p.occupation) for p in unstable)
    if not unstable:
        return f"Gendered words are {' and '.join(clauses)}."
    if not stable:
        return f"The {small} gaps are too small to split."
    return f"Gendered words are {' and '.join(clauses)}; the {small} gap is too small to split."


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
