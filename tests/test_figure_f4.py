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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f4.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f4", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f4():
    return _load_script()


# (occupation, swap flip rate with interval, removal flip rate with interval)
RATES = [
    ("surgeon", (0.063, 0.037, 0.093), (0.057, 0.030, 0.087)),
    ("nurse", (0.093, 0.060, 0.127), (0.063, 0.037, 0.090)),
    ("model", (0.087, 0.057, 0.120), (0.080, 0.050, 0.113)),
    ("pooled", (0.069, 0.057, 0.081), (0.063, 0.052, 0.075)),
]


def _row(screener, occupation, subset, condition, rate):
    value, low, high = rate
    return {
        "screener": screener,
        "occupation": occupation,
        "subset": subset,
        "condition": condition,
        "flip_rate": value,
        "flip_rate_ci_low": low,
        "flip_rate_ci_high": high,
    }


@pytest.fixture
def table():
    """A csv as `scripts/counterfactual_gender.py` writes it, with rows the figure must ignore:
    the baseline model and the unflagged subset."""
    rows = []
    for occupation, swap, removal in RATES:
        rows += [
            _row("deberta", occupation, "all", "swap", swap),
            _row("deberta", occupation, "all", "removal", removal),
            _row("deberta", occupation, "unflagged", "swap", (0.5, 0.4, 0.6)),
            _row("baseline", occupation, "all", "swap", (0.4, 0.3, 0.5)),
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def figure(f4, table):
    fig = f4.build(table)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _ax(fig):
    (ax,) = [a for a in fig.axes if a.collections]
    return ax


def _dots(ax):
    return [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]


def _hex(colours):
    return [matplotlib.colors.to_hex(c).upper() for c in colours]


def test_the_title_gives_the_pooled_flip_rates_of_the_main_model(f4, table):
    assert f4.title(table) == (
        "Swapping pronouns flips 6.9% of the main model's decisions; removing them flips 6.3%."
    )


def test_occupations_are_sorted_by_the_swap_flip_rate_with_the_pooled_row_last(figure):
    ax = _ax(figure)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["nurse", "model", "surgeon", "all six"]
    assert ax.yaxis_inverted()


def test_each_condition_has_a_dot_and_an_interval_per_row_swap_in_the_highlight(figure):
    ax = _ax(figure)
    swap, removal = _dots(ax)
    assert swap.get_offsets()[:, 0].tolist() == pytest.approx([0.093, 0.087, 0.063, 0.069])
    assert removal.get_offsets()[:, 0].tolist() == pytest.approx([0.063, 0.080, 0.057, 0.063])
    assert set(_hex(swap.get_facecolor())) == {plotting.HIGHLIGHT}
    assert set(_hex(removal.get_facecolor())) == {plotting.NEUTRAL}
    intervals = [c for c in ax.collections if isinstance(c, matplotlib.collections.LineCollection)]
    ends = [[seg.tolist() for seg in c.get_segments()] for c in intervals]
    assert [seg[0][0] for seg in ends[0]] == pytest.approx([0.060, 0.057, 0.037, 0.057])
    assert [seg[1][0] for seg in ends[0]] == pytest.approx([0.127, 0.120, 0.093, 0.081])


def test_the_conditions_are_named_next_to_the_data_and_there_is_no_legend(figure):
    ax = _ax(figure)
    texts = {t.get_text() for t in ax.texts}
    assert {"pronouns swapped", "pronouns removed"} <= texts
    assert ax.get_legend() is None


def test_the_axis_names_the_quantity_and_starts_at_the_zero_reference(figure):
    ax = _ax(figure)
    assert ax.get_xlabel() == "Share of bios whose prediction changes"
    assert ax.get_xlim()[0] == 0
    assert any(list(line.get_xdata()) == [0, 0] for line in ax.lines)


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    # guard: the style sets 7 pt and new_figure's width; this pins both for F4
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    ax = _ax(figure)
    low, high = sorted(ax.get_xlim())
    hidden = {  # labels of ticks outside the axis limits are not drawn
        id(label)
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False)
        if not low <= tick <= high
    }
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and id(t) not in hidden:
            extent = t.get_window_extent()
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def test_main_saves_the_figure_from_the_csv(f4, table, tmp_path):
    table.to_csv(tmp_path / "counterfactual_gender.csv", index=False)
    fig = f4.main(tmp_path)
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "f4_flip_rates.pdf",
        "f4_flip_rates.png",
    ]
    assert fig._suptitle.get_text() == f4.title(table)
    plt.close(fig)
