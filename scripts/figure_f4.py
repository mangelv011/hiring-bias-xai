"""F4: how often the main model's decision flips when pronouns are swapped, against the removal
control, per showcase occupation.

Reads `results/counterfactual_gender.csv` (main model, all bios). Occupations are sorted by the
swap flip rate, the pooled row comes last. Each row has two dots with their 95% bootstrap
intervals: the swap in the highlight colour, the removal control in grey.
"""

from pathlib import Path

import pandas as pd

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

SCREENER = "deberta"
FIGURE_NAME = "f4_flip_rates"
POOLED = "pooled"
DARK = "#333333"
ROW_INCHES = 0.3
EXTRA_INCHES = 1.2
OFFSET = 0.18  # distance of each dot from the row's centre line
POOLED_GAP = 0.5


def _main_model(table: pd.DataFrame) -> pd.DataFrame:
    """The main model's rows for all bios, indexed by occupation, one frame per condition."""
    rows = table[(table["screener"] == SCREENER) & (table["subset"] == "all")]
    return {name: group.set_index("occupation") for name, group in rows.groupby("condition")}


def title(table: pd.DataFrame) -> str:
    """The finding: how the pooled flip rate of the swap compares with the removal control.

    "More" or "less" only when the 95% intervals do not overlap; otherwise "about as often".
    """
    by_condition = _main_model(table)
    swap = by_condition["swap"].loc[POOLED]
    removal = by_condition["removal"].loc[POOLED]
    if swap["flip_rate_ci_low"] > removal["flip_rate_ci_high"]:
        comparison = "more often than"
    elif swap["flip_rate_ci_high"] < removal["flip_rate_ci_low"]:
        comparison = "less often than"
    else:
        comparison = "about as often as"
    return f"Swapping pronouns changes the main model's decisions {comparison} deleting them."


def build(table: pd.DataFrame):
    """One row per showcase occupation sorted by the swap flip rate, then the pooled row."""
    by_condition = _main_model(table)
    swap, removal = by_condition["swap"], by_condition["removal"]
    occupations = swap.drop(POOLED).sort_values("flip_rate", ascending=False, kind="stable").index
    order = [*occupations, POOLED]
    rows = [*range(len(occupations)), len(occupations) - 1 + 1 + POOLED_GAP]
    fig, ax = plotting.new_figure(ROW_INCHES * len(order) + EXTRA_INCHES)
    fig.suptitle(title(table), fontweight="bold", wrap=True, x=0, ha="left")
    ax.axvline(0, color=DARK, lw=0.6)
    for frame, colour, shift, label in (
        (swap, plotting.HIGHLIGHT, -OFFSET, "pronouns swapped"),
        (removal, plotting.NEUTRAL, OFFSET, "pronouns removed"),
    ):
        values = frame.loc[order]
        ys = [row + shift for row in rows]
        ax.hlines(ys, values["flip_rate_ci_low"], values["flip_rate_ci_high"], colors=colour)
        ax.scatter(values["flip_rate"], ys, c=colour, zorder=3)
        ax.text(
            values["flip_rate_ci_high"].iloc[0] + 0.004,
            ys[0],
            label,
            color=colour,
            va="center",
            ha="left",
        )
        ax.text(
            values["flip_rate_ci_high"].iloc[-1] + 0.004,
            ys[-1],
            f"{values['flip_rate'].iloc[-1]:.1%}",
            color=colour,
            va="center",
            ha="left",
        )
    ax.set_yticks(rows)
    ax.set_yticklabels(["all six" if name == POOLED else plotting.job_name(name) for name in order])
    ax.set_ylim(rows[-1] + 0.6, -0.6)
    ax.set_xlim(0, max(swap["flip_rate_ci_high"].max(), removal["flip_rate_ci_high"].max()) * 1.9)
    ax.set_xlabel("Share of bios whose prediction changes")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return fig


def main(results_dir: Path = RESULTS_DIR):
    table = pd.read_csv(results_dir / "counterfactual_gender.csv")
    fig = build(table)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
