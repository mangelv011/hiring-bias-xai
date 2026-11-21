"""F2: which words drove the main model's wrong screening decisions for women and for men.

Two showcase-subset bios, from the saved Shapley values of the main model, with values recomputed
for the occupation it predicted. The cases mirror the finding of F1 and are picked by rules fixed
before looking at the bios:

- Women whose true occupation is one of the three male-dominated showcase occupations and whom
  the main model gets wrong: the bio where the gender words add the most to the log-odds of the
  predicted (wrong) occupation, that is, the largest sum of their Shapley values, which must be
  above zero.
- Men in the three female-dominated showcase occupations whom it gets wrong: the same score.
- A gender word is a word unit whose core, without punctuation at its ends, is a scrubbed word.
- A tie goes to the bio that comes first in the subset. Bios under 15 word units, and bios with a
  run of 6 or more digits (they may hold an identifier such as a licence number), are never
  picked. A bio is also left out when the stored full-bio logits predict another occupation than
  the saved prediction does (the stored bio is the words rejoined with single spaces).
- If a group has no qualifying bio, the figure and the record say so; no other rule replaces it.
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib.font_manager
import numpy as np
import pandas as pd

from hiring_bias_xai import attribution, data, plotting, results
from hiring_bias_xai.config import RESULTS_DIR

SCREENER = "deberta"
IDENTIFIER = re.compile(r"[0-9]{6,}")
MIN_UNITS = 15


class Case(NamedTuple):
    bio_id: str
    words: list[str]
    true: str
    predicted: str
    gender: str
    values: np.ndarray  # Shapley value of each word for the predicted occupation


def _gendered_mask(case: Case) -> np.ndarray:
    return np.array([data.is_gendered_word(word) for word in case.words], dtype=bool)


def _eligible(case: Case) -> bool:
    return len(case.words) >= MIN_UNITS and IDENTIFIER.search(" ".join(case.words)) is None


def gendered_push(case: Case) -> float:
    """Sum of the Shapley values of the gendered words, for the predicted occupation."""
    return float(case.values[_gendered_mask(case)].sum())


def pick(cases: Sequence[Case], gender: str, occupations: Sequence[str]) -> Case | None:
    """The misclassified bio of `gender` in `occupations` with the largest `gendered_push`.

    The push must be above zero and the bio eligible; None if no bio qualifies.
    """
    scored = [
        (gendered_push(c), c)
        for c in cases
        if c.gender == gender and c.true in occupations and c.true != c.predicted and _eligible(c)
    ]
    scored = [(push, case) for push, case in scored if push > 0]
    return max(scored, key=lambda pair: pair[0])[1] if scored else None  # max keeps the first


def load_cases(subset: pd.DataFrame, predicted: Sequence[str]) -> tuple[list[Case], int]:
    """A case per bio of `subset` from its saved attribution, and how many bios were skipped."""
    cases, skipped = [], 0
    for bio, label in zip(subset.itertuples(), predicted, strict=True):
        saved = attribution.load(attribution.path(SCREENER, bio.id))
        column = data.OCCUPATIONS.index(label)
        if saved.logits[saved.index[0, -1]].argmax() != column:
            skipped += 1
            continue
        values = saved.values_for(column)
        cases.append(Case(bio.id, saved.words, bio.occupation, label, bio.gender, values))
    return cases, skipped


TOP_WORDS = 8
FIGURE_NAME = "f2_shapley_cases"
RULE = (
    "Women whose true occupation is one of the three male-dominated showcase occupations, and "
    "men whose true occupation is one of the three female-dominated ones, whom the main model "
    "gets wrong: in each group, the bio where the gender words add the most to the log-odds of "
    "the predicted occupation (largest sum of their Shapley values, above zero). A gender word "
    "is a word unit whose core is a scrubbed word. Bios under 15 word units or with six or more "
    "digits in a row are never picked. A group with no qualifying bio is shown as empty."
)
DARK = "#333333"
FONT_PT = plotting.MIN_FONT_PT
LINE_PT = 10
MARGIN_PT = 4
LABEL_PT = 70
VALUE_ROOM_PT = 16
BAR_PT = 11
AXIS_PT = 30
GAP_PT = 8
LABEL_CHARS = 14


def _text_height_pt(fig, text: str, **kwargs) -> float:
    """Height in points of `text` wrapped at the width of `fig`."""
    probe = fig.text(
        MARGIN_PT / (fig.get_figwidth() * 72),
        0.9,
        text,
        wrap=True,
        va="top",
        fontsize=FONT_PT,
        **kwargs,
    )
    fig.canvas.draw()
    height = probe.get_window_extent().height * 72 / fig.dpi
    probe.remove()
    return height


def _char_width_pt(fig) -> float:
    """Width of one character of the monospaced font at the figure's font size."""
    font = matplotlib.font_manager.FontProperties(family="monospace", size=FONT_PT)
    width, _, _ = fig.canvas.get_renderer().get_text_width_height_descent("0" * 100, font, False)
    return width / 100 * 72 / fig.dpi


def _layout(words: Sequence[str], capacity: int) -> list[tuple[int, int, int, str]]:
    """The bio as pieces on lines of `capacity` characters: (line, column, word index, text).

    A word longer than a line is cut into several pieces.
    """
    pieces, line, column = [], 0, 0
    for index, word in enumerate(words):
        for start in range(0, len(word), capacity):
            piece = word[start : start + capacity]
            if column and column + len(piece) > capacity:
                line, column = line + 1, 0
            pieces.append((line, column, index, piece))
            column += len(piece) + 1
    return pieces


def _axes(fig, left: float, bottom: float, width: float, height: float):
    """Axes at a rectangle given in points from the bottom left of the figure."""
    page_w, page_h = fig.get_figwidth() * 72, fig.get_figheight() * 72
    return fig.add_axes([left / page_w, bottom / page_h, width / page_w, height / page_h])


def _sentence(fig, case: Case, rect, char_pt: float, capacity: int, top: set[int]) -> None:
    """The bio as words on lines with the `top` words underlined; one data unit is one point."""
    ax = _axes(fig, *rect)
    ax.set_xlim(0, rect[2])
    ax.set_ylim(0, rect[3])
    ax.axis("off")
    gendered = _gendered_mask(case)
    for line, column, index, piece in _layout(case.words, capacity):
        x, baseline = column * char_pt, rect[3] - (line + 1) * LINE_PT + 2.5
        ax.text(x, baseline, piece, family="monospace", fontsize=FONT_PT, color=DARK)
        if index in top:
            colour = plotting.HIGHLIGHT if gendered[index] else DARK
            end = x + len(piece) * char_pt
            ax.plot([x, end], [baseline - 1.6] * 2, color=colour, lw=0.7, gid="underline")


def _bars(fig, case: Case, order: np.ndarray, rect) -> None:
    """Horizontal bars of the Shapley values of the words at `order`, largest at the top."""
    ax = _axes(fig, *rect)
    gendered = _gendered_mask(case)
    values = case.values[order]
    colours = [plotting.HIGHLIGHT if gendered[i] else plotting.NEUTRAL for i in order]
    ax.barh(range(len(order)), values, color=colours, height=0.7)
    ax.axvline(0, color=DARK, lw=0.6)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [
            word if len(word) <= LABEL_CHARS else word[: LABEL_CHARS - 1] + "…"
            for word in (case.words[i] for i in order)
        ]
    )
    for tick, i in zip(ax.get_yticklabels(), order, strict=True):
        tick.set_color(plotting.HIGHLIGHT if gendered[i] else DARK)
    ax.tick_params(axis="y", length=0)
    pad = 0.35 * np.abs(values).max()  # room for the value written at the end of a bar
    ax.set_xlim(min(0.0, values.min() - pad), max(0.0, values.max() + pad))
    for row, value in enumerate(values):
        ax.annotate(
            f"{value:+.1f}",
            (value, row),
            xytext=(2 if value >= 0 else -2, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
        )
    ax.set_xlabel(f"Shapley value for {plotting.job_name(case.predicted)} (score points)")


GROUPS = (
    ("Woman", "women_in_male_dominated", "female", "male_dominated"),
    ("Man", "men_in_female_dominated", "male", "female_dominated"),
)


def _title(woman: Case | None, man: Case | None) -> str:
    if woman and man:
        return (
            f"Gender words added {gendered_push(woman):.1f} score points to a wrong job "
            f"for a woman and {gendered_push(man):.1f} for a man."
        )
    for case, who, other in ((woman, "a woman", "man"), (man, "a man", "woman")):
        if case:
            return (
                f"Gender words added {gendered_push(case):.1f} score points to a wrong job "
                f"for {who}; no misclassified {other} qualified."
            )
    return "No misclassified woman or man met the rules, so there is no case to show."


def build(woman: Case | None, man: Case | None):
    """For each group, the bio with its eight largest words underlined and a bar for each.

    A group without a case gets a one-line panel that says so.
    """
    fig, probe = plotting.new_figure(1.0)
    probe.remove()
    fig.set_size_inches(plotting.WIDTH_INCHES, 4)  # room for the wrapped texts that are measured
    fig.set_dpi(plotting.DPI)  # glyph widths depend on the resolution: measure at the saved one
    fig.set_layout_engine("none")  # every axes and text is placed by hand, in points
    width = plotting.WIDTH_INCHES * 72
    char_pt = _char_width_pt(fig)
    text_width = width - 2 * MARGIN_PT
    capacity = int(text_width // char_pt)
    title = _title(woman, man)
    note = (
        "A score point is one unit of the model's score for a job. Per-word values carry sampling "
        "noise; no per-word interval is shown."
    )
    title_pt = _text_height_pt(fig, title, fontweight="bold")
    note_pt = _text_height_pt(fig, note)
    panels = []
    for (label, _, _, side), case in zip(GROUPS, (woman, man), strict=True):
        if case is None:
            kind = "male" if side == "male_dominated" else "female"
            header = f"{label} in a {kind}-dominated occupation: no bio met the rules."
            panels.append((header, _text_height_pt(fig, header), None, None, 0))
            continue
        units = int(_gendered_mask(case).sum())
        header = (
            f"{label}, true {plotting.job_name(case.true)}, "
            f"predicted {plotting.job_name(case.predicted)} "
            f"({case.bio_id}). Gender words: {gendered_push(case):+.1f} over {units} units."
        )
        order = np.argsort(-np.abs(case.values), kind="stable")[:TOP_WORDS]
        sentence_pt = (_layout(case.words, capacity)[-1][0] + 1) * LINE_PT + 2
        panels.append((header, _text_height_pt(fig, header), case, order, sentence_pt))
    bars_pt = TOP_WORDS * BAR_PT
    total = 2 * MARGIN_PT + title_pt + note_pt
    for _, header_pt, case, _, sentence_pt in panels:
        total += GAP_PT + header_pt
        if case is not None:
            total += sentence_pt + GAP_PT + bars_pt + AXIS_PT
    fig.set_size_inches(plotting.WIDTH_INCHES, total / 72)
    left = MARGIN_PT / width
    fig.text(left, 1 - MARGIN_PT / total, title, wrap=True, va="top", fontweight="bold")
    cursor = total - MARGIN_PT - title_pt
    for header, header_pt, case, order, sentence_pt in panels:
        cursor -= GAP_PT
        fig.text(left, cursor / total, header, wrap=True, va="top")
        cursor -= header_pt
        if case is None:
            continue
        cursor -= sentence_pt
        top = {int(i) for i in order}
        _sentence(fig, case, (MARGIN_PT, cursor, text_width, sentence_pt), char_pt, capacity, top)
        cursor -= GAP_PT + bars_pt
        bar_width = width - LABEL_PT - MARGIN_PT - VALUE_ROOM_PT
        _bars(fig, case, order, (LABEL_PT, cursor, bar_width, bars_pt))
        cursor -= AXIS_PT
    fig.text(left, (MARGIN_PT + note_pt) / total, note, wrap=True, va="top")
    return fig


def _record(case: Case) -> dict:
    return {
        "bio_id": case.bio_id,
        "true": case.true,
        "predicted": case.predicted,
        "words": len(case.words),
        "gendered_push": gendered_push(case),
    }


def main(results_dir: Path = RESULTS_DIR):
    subset = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    predicted = results.load_predictions(SCREENER, subset)
    cases, skipped = load_cases(subset, list(predicted))
    showcase = json.loads((results_dir / "showcase_occupations.json").read_text())
    picks, record = [], {}
    for _, key, gender, side in GROUPS:
        group = [row["occupation"] for row in showcase[side]]
        n_wrong = sum(
            c.gender == gender and c.true in group and c.true != c.predicted for c in cases
        )
        picked = pick(cases, gender, group)
        picks.append(picked)
        record[key] = {
            "n_wrong": n_wrong,
            "pick": _record(picked) if picked else None,
        }
    fig = build(*picks)
    plotting.save(fig, FIGURE_NAME, results_dir / "figures")
    results.write_json(
        results_dir / "f2_cases.json",
        {
            "rule": RULE,
            "min_units": MIN_UNITS,
            "n_bios": len(subset),
            "n_skipped": skipped,
            **record,
        },
    )
    return fig


if __name__ == "__main__":
    main()
