import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
import pytest

from hiring_bias_xai import plotting

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f3.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f3():
    return _load_script()


def _words(rows):
    """Top-word rows as `gap_decomposition.csv` has them: word, contribution, interval, gendered."""
    return pd.DataFrame(
        rows,
        columns=[
            "word",
            "contribution",
            "contribution_ci_low",
            "contribution_ci_high",
            "gendered",
        ],
    )


SURGEON_WORDS = _words(
    [
        ("he", -0.40, -0.60, -0.20, True),
        ("surgery", 0.30, 0.10, 0.50, False),
        ("medical", -0.10, -0.30, 0.10, False),
        ("she", 0.20, 0.05, 0.35, True),
    ]
)
MODEL_WORDS = _words(
    [
        ("born", 1.0, 0.5, 1.4, False),
        ("her", -0.5, -0.8, -0.2, True),
        ("ranked", 2.0, 1.5, 2.6, False),
    ]
)


def _shares(value, low, high):
    return {"value": value, "ci_low": low, "ci_high": high}


# A summary as `gap_explained.json` has it, with the fields the figure reads. The surgeon gap is
# too close to 0 to split; the model gap is not.
_SURGEON_SUMMARY = {
    "score_gap": _shares(-0.774, -1.540, 0.076),
    "decomposed_gap": _shares(-0.774, -1.540, 0.076),
    "gap_unstable": True,
    "top_share": _shares(0.45, -1.81, 1.98),
    "gendered_share": _shares(1.45, -8.15, 8.56),
}
_MODEL_SUMMARY = {
    "score_gap": _shares(5.402, 4.444, 6.404),
    "decomposed_gap": _shares(5.402, 4.444, 6.404),
    "gap_unstable": False,
    "top_words": ["w"] * 10,  # only the number of words is read
    "top_share": _shares(0.61, 0.51, 0.77),
    "gendered_share": _shares(0.15, 0.12, 0.19),
}


@pytest.fixture
def panels(f3):
    return [
        f3.Panel("surgeon", "male_dominated", SURGEON_WORDS, _SURGEON_SUMMARY),
        f3.Panel("model", "female_dominated", MODEL_WORDS, _MODEL_SUMMARY),
    ]


@pytest.fixture
def figure(f3, panels):
    fig = f3.build(panels)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _data_axes(fig):
    return [ax for ax in fig.axes if ax.collections]


def _dots(ax):
    (dots,) = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    return dots


def _intervals(ax):
    (lines,) = [c for c in ax.collections if isinstance(c, matplotlib.collections.LineCollection)]
    return lines


def test_each_panel_lists_its_words_sorted_by_contribution_with_the_largest_at_the_top(figure):
    surgeon, model = _data_axes(figure)
    assert [t.get_text() for t in surgeon.get_yticklabels()] == ["surgery", "she", "medical", "he"]
    assert [t.get_text() for t in model.get_yticklabels()] == ["ranked", "born", "her"]
    assert surgeon.yaxis_inverted()
    offsets = _dots(surgeon).get_offsets()
    assert offsets[:, 0].tolist() == pytest.approx([0.30, 0.20, -0.10, -0.40])
    assert offsets[:, 1].tolist() == pytest.approx([0, 1, 2, 3])


def test_every_dot_has_its_interval_and_there_is_a_zero_line(figure):
    surgeon, _ = _data_axes(figure)
    segments = [seg.tolist() for seg in _intervals(surgeon).get_segments()]
    assert segments == [
        [[0.10, 0], [0.50, 0]],
        [[0.05, 1], [0.35, 1]],
        [[-0.30, 2], [0.10, 2]],
        [[-0.60, 3], [-0.20, 3]],
    ]
    zero_lines = [line for line in surgeon.lines if list(line.get_xdata()) == [0, 0]]
    assert len(zero_lines) == 1


def _hex(colours):
    return [matplotlib.colors.to_hex(c).upper() for c in colours]


def test_gendered_words_are_in_the_highlight_colour_and_the_others_in_grey(figure):
    surgeon, _ = _data_axes(figure)
    expected = [plotting.NEUTRAL, plotting.HIGHLIGHT, plotting.NEUTRAL, plotting.HIGHLIGHT]
    assert _hex(_dots(surgeon).get_facecolor()) == expected
    assert _hex(_intervals(surgeon).get_color()) == expected
    labels = _hex(t.get_color() for t in surgeon.get_yticklabels())
    assert labels == [plotting.HIGHLIGHT if c == plotting.HIGHLIGHT else "#333333" for c in expected]


def _title_lines(fig):
    fig.canvas.draw()
    title = fig._suptitle
    return title.get_text(), title._get_wrapped_text().count("\n") + 1


def test_the_title_gives_the_gendered_share_of_a_stable_gap_and_says_when_a_gap_is_too_small(
    figure,
):
    # Model: gendered words 0.15 of a +5.4 gap. Surgeon: the gap -0.77 has an interval with 0.
    text, lines = _title_lines(figure)
    assert text == (
        "Gendered words are 15% of the +5.4 model-bio gap; the surgeon gap is too small to split."
    )
    assert lines <= 2


def test_the_title_covers_two_stable_gaps_and_two_small_ones_in_two_lines(f3):
    stable = {**_MODEL_SUMMARY, "gendered_share": _shares(0.39, 0.25, 0.86)}
    stable_gap = {**stable, "score_gap": _shares(2.34, 1.08, 3.68)}
    both_stable = [
        f3.Panel("nurse", "female_dominated", MODEL_WORDS, stable_gap),
        f3.Panel("model", "female_dominated", MODEL_WORDS, _MODEL_SUMMARY),
    ]
    text, lines = _title_lines(f3.build(both_stable))
    assert text == "Gendered words are 39% of the +2.3 nurse-bio gap and 15% of the +5.4 model-bio gap."
    assert lines <= 2
    both_small = [
        f3.Panel("surgeon", "male_dominated", SURGEON_WORDS, _SURGEON_SUMMARY),
        f3.Panel("composer", "male_dominated", SURGEON_WORDS, _SURGEON_SUMMARY),
    ]
    text, lines = _title_lines(f3.build(both_small))
    assert text == "The surgeon and composer gaps are too small to split."
    assert lines <= 2


def test_each_panel_header_gives_its_total_gap_and_either_the_shares_or_why_there_are_none(figure):
    surgeon, model = _data_axes(figure)
    assert surgeon.get_title(loc="left") == (
        "Surgeon (male-dominated), gap -0.8 [-1.5, +0.1]\n"
        "interval includes 0: no shares given"
    )
    assert model.get_title(loc="left") == (
        "Model (female-dominated), gap +5.4 [+4.4, +6.4]\n"
        "top 10 words: 61% [51%, 77%]\n"
        "gendered words: 15% [12%, 19%]"
    )


def test_the_x_axis_names_the_quantity_and_its_unit(figure):
    for ax in _data_axes(figure):
        assert ax.get_xlabel() == "Contribution to the women-minus-men gap (log-odds)"


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    # guard: the style sets 7 pt and new_figure's width; this pins both for F3
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert texts
    assert min(t.get_fontsize() for t in texts) >= 7


def test_nothing_is_drawn_outside_the_figure(figure):
    # panel headers start at the axes, so each line must be short enough to end inside the figure
    box = figure.bbox
    hidden = set()  # labels of ticks outside the axis limits are not drawn
    for ax in figure.axes:
        low, high = sorted(ax.get_xlim())
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False):
            if not low <= tick <= high:
                hidden.add(id(label))
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and id(t) not in hidden:
            extent = t.get_window_extent()
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def _summary(gap):
    return {"score_gap": {"value": gap}}


def test_each_panel_is_the_showcase_occupation_of_its_kind_with_the_largest_absolute_score_gap(f3):
    showcase = {
        "male_dominated": [{"occupation": o} for o in ("surgeon", "composer", "engineer")],
        "female_dominated": [{"occupation": o} for o in ("nurse", "model", "dietitian")],
    }
    summaries = {  # the main model's; a negative gap counts by its size
        "surgeon": _summary(-0.8),
        "composer": _summary(0.5),
        "engineer": _summary(-0.2),
        "nurse": _summary(1.3),
        "model": _summary(-5.4),
        "dietitian": _summary(2.3),
    }
    assert f3.pick(summaries, showcase) == {"male_dominated": "surgeon", "female_dominated": "model"}
