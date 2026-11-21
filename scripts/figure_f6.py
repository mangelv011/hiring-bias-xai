"""F6: where gender sits inside the main model, by layer.

Panel a: the accuracy of linear probes that read gender from scrubbed bios, for the pretrained
encoder and the main model, with 95% bootstrap intervals, from `results/probing.csv` (all bios),
and the three references on the same split as horizontal lines: the majority class, the
occupation alone and TF-IDF. Panel b: the share of the swap's effect on the true-class log-odds
that patching clean states into the corrupted run restores, at the positions where the tokens
differ and at all positions, with 95% bootstrap intervals, from `results/patching.csv`, with
reference lines at 0 and 1.
The title states the finding; the numbers are in the CSV files.
"""

from pathlib import Path

import pandas as pd

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR
from hiring_bias_xai.plotting import HIGHLIGHT, NEUTRAL

FIGURE_NAME = "f6_probing_patching"
DARK = "#333333"
HEIGHT_INCHES = 5.6
MODELS = (("pretrained", NEUTRAL, "pretrained"), ("deberta", HIGHLIGHT, "main model"))
REFERENCES = (
    ("tfidf", "TF-IDF", "-."),
    ("occupation_only", "occupation only", "--"),
    ("majority", "majority class", ":"),
)
OCCUPATION_ONLY = "occupation_only"
LABEL_GAP = 0.07  # least distance between the labels of panel a, as a share of its y range
LABEL_ROOM = 2.0  # room right of the last layer for the labels, in layers
LAST_LAYER_NOTE = "Last layer:\nthe classifier\nreads only [CLS],\nso edited words\ngive 0%."
HALF = 0.5  # the share of the effect that marks where it has left the edited words


def _all_bios(probing: pd.DataFrame) -> pd.DataFrame:
    return probing[probing["subset"] == "all"]


def title(probing: pd.DataFrame, patching: pd.DataFrame) -> str:
    """The finding: whether a probe reads gender beyond the occupation alone once the pronouns
    are removed, and the first layer where the effect of the swap has left the edited words.

    A probe counts when its interval starts above the interval of the occupation-only reference;
    the title says "stays readable" if all do, "in only some layers" if some do. The effect has
    left the edited words when the interval of the restored share is below one half.
    """
    rows = _all_bios(probing)
    probes = rows[rows["model"].isin([name for name, _, _ in MODELS])]
    reference = rows[rows["model"] == OCCUPATION_ONLY].iloc[0]
    above = int((probes["ci_low"] > reference["ci_high"]).sum())
    if above == len(probes):
        readable = "Gender stays readable after the pronouns are removed"
    elif above:
        readable = "Gender stays readable in only some layers after the pronouns are removed"
    else:
        readable = "Gender is not readable beyond the occupation once the pronouns are removed"
    differing = patching[patching["position_set"] == "differing"].sort_values("layer")
    left = differing[differing["mean_ci_high"] < HALF]
    if left.empty:
        where = "stays in the edited words in every layer"
    else:
        where = f"leaves the edited words by layer {left['layer'].iloc[0]}"
    return f"{readable}, and the swap's effect {where}."


def _set_layers(ax, layers: list[int]) -> None:
    ax.set_xticks(layers)
    ax.set_xticklabels(["emb." if layer == 0 else str(layer) for layer in layers])
    ax.set_xlim(min(layers), max(layers) + LABEL_ROOM)
    ax.spines["bottom"].set_bounds(min(layers), max(layers))
    ax.set_xlabel("Layer output")


def _probe_panel(ax, probing: pd.DataFrame) -> None:
    rows = _all_bios(probing)
    layers = sorted(rows["layer"].dropna().astype(int).unique())
    ends = []
    for name, colour, label in MODELS:
        model = rows[rows["model"] == name].sort_values("layer")
        ax.fill_between(
            model["layer"], model["ci_low"], model["ci_high"], color=colour, alpha=0.25, lw=0
        )
        ax.plot(model["layer"], model["accuracy"], color=colour, lw=1.2, label=name)
        ends.append((label, colour, model["accuracy"].iloc[-1]))
    for name, label, style in REFERENCES:
        accuracy = rows[rows["model"] == name]["accuracy"].iloc[0]
        ax.plot(
            [layers[0], layers[-1]], [accuracy, accuracy], color=DARK, lw=0.8, ls=style, label=name
        )
        ends.append((label, DARK, accuracy))
    low, high = rows["ci_low"].min() - 0.02, rows["ci_high"].max() + 0.02
    ax.set_ylim(low, high)
    label_y = plotting.spread([y for *_, y in ends], LABEL_GAP * (high - low))
    for (label, colour, _), y in zip(ends, label_y, strict=True):
        ax.text(layers[-1] + 0.15, y, label, color=colour, va="center", ha="left")
    _set_layers(ax, layers)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("a  Reading gender from scrubbed bios", loc="left", fontweight="bold")


def _patch_panel(ax, patching: pd.DataFrame) -> None:
    layers = sorted(patching["layer"].unique())
    for value, name in ((0, "zero"), (1, "one")):
        ax.plot([layers[0], layers[-1]], [value, value], color=DARK, lw=0.6, label=name)
    for position_set, colour, label in (
        ("others", NEUTRAL, "all other words"),
        ("differing", HIGHLIGHT, "edited words only"),
    ):
        rows = patching[patching["position_set"] == position_set].sort_values("layer")
        ax.fill_between(
            rows["layer"], rows["mean_ci_low"], rows["mean_ci_high"], color=colour, alpha=0.25, lw=0
        )
        ax.plot(rows["layer"], rows["mean"], color=colour, lw=1.2, label=position_set)
        y = rows["mean"].iloc[-1] + (-0.1 if position_set == "others" else 0.12)
        ax.text(layers[-1] + 0.15, y, label, color=colour, va="center", ha="left")
    ax.text(0.02, 0.48, LAST_LAYER_NOTE, transform=ax.transAxes, color=DARK, va="center", ha="left")
    _set_layers(ax, layers)
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylabel("Restored share of the effect")
    ax.set_title("b  Patching clean into swapped bios", loc="left", fontweight="bold")


def build(probing: pd.DataFrame, patching: pd.DataFrame):
    """Panel a above panel b, under a title that states the finding."""
    fig, probe = plotting.new_figure(HEIGHT_INCHES)
    probe.remove()
    top, bottom = fig.subplots(2, 1, height_ratios=[1.15, 1])
    fig.suptitle(
        title(probing, patching),
        fontsize=plotting.MIN_FONT_PT,
        fontweight="bold",
        wrap=True,
        x=0,
        ha="left",
    )
    _probe_panel(top, probing)
    _patch_panel(bottom, patching)
    return fig


def main(results_dir: Path = RESULTS_DIR):
    probing = pd.read_csv(results_dir / "probing.csv")
    patching = pd.read_csv(results_dir / "patching.csv")
    fig = build(probing, patching)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
