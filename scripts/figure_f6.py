"""F6: can a probe read gender from the main model's layers, once pronouns are removed?

The accuracy of linear probes that read gender from scrubbed bios, for the main model, with its
95% bootstrap intervals, from `results/probing.csv` (all bios), and two references on the same
split as horizontal lines: the occupation alone and TF-IDF. The pretrained encoder and the
majority class stay in the CSV; activation patching is in `docs/results.md` only.
The title states the finding; the numbers are in the CSV file.
"""

from pathlib import Path

import pandas as pd

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.plotting import HIGHLIGHT

FIGURE_NAME = "f6_probing"
DARK = "#333333"
HEIGHT_INCHES = 2.9
MAIN = "deberta"
REFERENCES = (
    ("tfidf", "TF-IDF", "-."),
    ("occupation_only", "occupation only", "--"),
)
OCCUPATION_ONLY = "occupation_only"
LABEL_GAP = 0.07  # least distance between the labels, as a share of the y range
LABEL_ROOM = 2.0  # room right of the last layer for the labels, in layers


def _all_bios(probing: pd.DataFrame) -> pd.DataFrame:
    return probing[probing["subset"] == "all"]


def title(probing: pd.DataFrame) -> str:
    """The finding: whether a probe reads gender beyond the occupation alone once the pronouns
    are removed.

    A probe counts when its interval starts above the interval of the occupation-only reference;
    the title says "still readable" if all the main model's layers do, "in only some layers" if
    some do.
    """
    rows = _all_bios(probing)
    probes = rows[rows["model"] == MAIN]
    reference = rows[rows["model"] == OCCUPATION_ONLY].iloc[0]
    above = int((probes["ci_low"] > reference["ci_high"]).sum())
    if above == len(probes):
        return "Gender is still readable from bios without pronouns."
    if above:
        return "Gender is readable from bios without pronouns in only some layers."
    return "Gender is not readable beyond the occupation from bios without pronouns."


def _set_layers(ax, layers: list[int]) -> None:
    ax.set_xticks(layers)
    ax.set_xticklabels(["emb." if layer == 0 else str(layer) for layer in layers])
    ax.set_xlim(min(layers), max(layers) + LABEL_ROOM)
    ax.spines["bottom"].set_bounds(min(layers), max(layers))
    ax.set_xlabel("Layer output")


def _probe_panel(ax, probing: pd.DataFrame) -> None:
    rows = _all_bios(probing)
    layers = sorted(rows["layer"].dropna().astype(int).unique())
    model = rows[rows["model"] == MAIN].sort_values("layer")
    ax.fill_between(
        model["layer"], model["ci_low"], model["ci_high"], color=HIGHLIGHT, alpha=0.25, lw=0
    )
    ax.plot(model["layer"], model["accuracy"], color=HIGHLIGHT, lw=1.2, label=MAIN)
    ends = [("main model", HIGHLIGHT, model["accuracy"].iloc[-1])]
    shown = [model]
    for name, label, style in REFERENCES:
        row = rows[rows["model"] == name].iloc[0]
        ax.fill_between(
            [layers[0], layers[-1]], row["ci_low"], row["ci_high"], color=DARK, alpha=0.1, lw=0
        )
        ax.plot(
            [layers[0], layers[-1]],
            [row["accuracy"]] * 2,
            color=DARK,
            lw=0.8,
            ls=style,
            label=name,
        )
        ends.append((label, DARK, row["accuracy"]))
        shown.append(rows[rows["model"] == name])
    shown = pd.concat(shown)
    low, high = shown["ci_low"].min() - 0.02, shown["ci_high"].max() + 0.02
    ax.set_ylim(low, high)
    label_y = plotting.spread([y for *_, y in ends], LABEL_GAP * (high - low))
    for (label, colour, _), y in zip(ends, label_y, strict=True):
        ax.text(layers[-1] + 0.15, y, label, color=colour, va="center", ha="left")
    _set_layers(ax, layers)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:g}%")
    ax.set_ylabel("Probe accuracy")


def build(probing: pd.DataFrame):
    """One panel under a title that states the finding."""
    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    fig.suptitle(
        title(probing),
        fontsize=plotting.MIN_FONT_PT,
        fontweight="bold",
        wrap=True,
        x=0,
        ha="left",
    )
    _probe_panel(ax, probing)
    return fig


def main(results_dir: Path = RESULTS_DIR):
    fig = build(pd.read_csv(results_dir / "probing.csv"))
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
