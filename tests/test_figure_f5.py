import importlib.util
import json
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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f5.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f5():
    return _load_script()


FRACTIONS = [0.0, 0.25, 0.5]
# (order, mean drop per fraction): MoRF falls fastest, LeRF is below random.
DROPS = {"morf": [0.0, 4.0, 8.0], "random": [0.0, 1.0, 2.0], "lerf": [0.0, -0.5, -1.0]}


@pytest.fixture
def curves():
    """A csv as `scripts/deletion_curves.py` writes it: the main model's three curves, and a
    baseline model's curve the figure must ignore."""
    rows = []
    for screener, scale in (("deberta", 1.0), ("baseline", 100.0)):
        for order, drops in DROPS.items():
            for fraction, drop in zip(FRACTIONS, drops, strict=True):
                rows.append(
                    {
                        "screener": screener,
                        "order": order,
                        "fraction": fraction,
                        "mean_agreement": 1.0 - fraction,
                        "agreement_ci_low": 0.9 - fraction,
                        "agreement_ci_high": 1.0 - fraction,
                        "mean_logodds_drop": drop * scale,
                        "drop_ci_low": (drop - 0.2) * scale,
                        "drop_ci_high": (drop + 0.2) * scale,
                    }
                )
    return pd.DataFrame(rows)


def _area(value):
    return {"value": value, "ci_low": value - 0.1, "ci_high": value + 0.1}


@pytest.fixture
def areas():
    return {
        "screeners": {
            "deberta": {
                "passed": True,
                "logodds_drop": {"area": {"morf": _area(4.0), "random": _area(1.0)}},
            },
            "baseline": {
                "passed": True,
                "logodds_drop": {"area": {"morf": _area(99.0), "random": _area(98.0)}},
            },
        }
    }


def _check(spearman, overlap, passed):
    def stat(median):
        return {"median": median, "median_ci": [median - 0.05, median + 0.04]}

    return {"spearman": stat(spearman), "top10_overlap": stat(overlap), "passed": passed}


@pytest.fixture
def stability():
    return {
        "checks": {
            "seed": _check(0.89, 0.80, False),
            "removal": _check(0.70, 0.60, False),
            "method": _check(0.95, 0.90, True),
        }
    }


@pytest.fixture
def figure(f5, curves, areas, stability):
    fig = f5.build(curves, areas, stability)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _axes(fig):
    top, bottom = sorted(fig.axes, key=lambda ax: -ax.get_position().y0)
    return top, bottom


def _hex(colour):
    return matplotlib.colors.to_hex(colour).upper()


def test_the_title_quotes_the_two_areas_and_counts_the_checks_that_passed(f5, areas, stability):
    assert f5.title(areas, stability) == (
        "Removing the words Shapley ranks highest costs the main model 4.0 log-odds on average, "
        "random removal 1.0; its rankings pass 1 of 3 stability checks."
    )


def test_the_title_says_not_clearly_more_when_the_deletion_check_failed(f5, areas, stability):
    areas["screeners"]["deberta"]["passed"] = False
    stability["checks"]["seed"]["passed"] = True
    assert f5.title(areas, stability) == (
        "Removing the words Shapley ranks highest costs the main model 4.0 log-odds on average, "
        "not clearly more than random removal (1.0); its rankings pass 2 of 3 stability checks."
    )


def test_the_curves_are_the_main_models_with_morf_in_the_highlight_and_random_in_grey(figure):
    top, _ = _axes(figure)
    lines = {line.get_label(): line for line in top.lines if line.get_label() in DROPS}
    assert set(lines) == set(DROPS)
    for order, drops in DROPS.items():
        assert list(lines[order].get_ydata()) == pytest.approx(drops)
        assert list(lines[order].get_xdata()) == pytest.approx(FRACTIONS)
    assert _hex(lines["morf"].get_color()) == plotting.HIGHLIGHT
    assert _hex(lines["random"].get_color()) == plotting.NEUTRAL
    assert _hex(lines["lerf"].get_color()) not in (plotting.HIGHLIGHT, plotting.NEUTRAL)


def test_each_curve_has_its_interval_band_and_the_zero_reference_is_drawn(figure):
    top, _ = _axes(figure)
    bands = [c for c in top.collections if isinstance(c, matplotlib.collections.PolyCollection)]
    assert len(bands) == 3
    spans = sorted(
        (
            round(b.get_paths()[0].vertices[:, 1].min(), 2),
            round(b.get_paths()[0].vertices[:, 1].max(), 2),
        )
        for b in bands
    )
    assert spans == [(-1.2, 0.2), (-0.2, 2.2), (-0.2, 8.2)]  # LeRF, random, MoRF: mean +- 0.2
    assert any(list(line.get_ydata()) == [0, 0] for line in top.lines)


def test_curves_are_named_next_to_their_ends_and_there_is_no_legend(figure):
    top, _ = _axes(figure)
    texts = " ".join(t.get_text() for t in top.texts)
    assert "MoRF" in texts and "LeRF" in texts and "random" in texts
    assert top.get_legend() is None


def test_the_deletion_axes_name_the_quantities_and_stop_at_half_of_the_words(figure):
    top, _ = _axes(figure)
    assert "log-odds" in top.get_ylabel()
    assert "words removed" in top.get_xlabel().lower()
    assert [t for t in top.get_xticks() if 0 <= t <= 0.5][-1] == 0.5
    assert top.get_xlim()[0] == 0


def test_checks_are_rows_sorted_by_the_median_correlation_with_pass_or_fail_in_the_label(figure):
    _, bottom = _axes(figure)
    labels = [t.get_text() for t in bottom.get_yticklabels()]
    assert [label.split("\n")[0] for label in labels] == [
        "leave-one-out vs Shapley",
        "seed 7 vs 42",
        "unk vs delete",
    ]
    assert [label.split("\n")[1] for label in labels] == ["passed", "failed", "failed"]
    assert bottom.yaxis_inverted()


def test_each_row_has_a_correlation_dot_in_the_highlight_and_an_overlap_dot_in_grey(figure):
    _, bottom = _axes(figure)
    dots = [c for c in bottom.collections if isinstance(c, matplotlib.collections.PathCollection)]
    correlation, overlap = dots
    assert correlation.get_offsets()[:, 0].tolist() == pytest.approx([0.95, 0.89, 0.70])
    assert overlap.get_offsets()[:, 0].tolist() == pytest.approx([0.90, 0.80, 0.60])
    assert {_hex(c) for c in correlation.get_facecolor()} == {plotting.HIGHLIGHT}
    assert {_hex(c) for c in overlap.get_facecolor()} == {plotting.NEUTRAL}
    intervals = [
        c for c in bottom.collections if isinstance(c, matplotlib.collections.LineCollection)
    ]
    first = [seg.tolist() for seg in intervals[0].get_segments()]
    assert [seg[0][0] for seg in first] == pytest.approx([0.90, 0.84, 0.65])
    assert [seg[1][0] for seg in first] == pytest.approx([0.99, 0.93, 0.74])


def test_the_two_measures_are_named_next_to_the_data_and_zero_is_marked(figure):
    _, bottom = _axes(figure)
    texts = " ".join(t.get_text() for t in bottom.texts)
    assert "rank correlation" in texts and "top-10 overlap" in texts
    assert bottom.get_legend() is None
    assert any(list(line.get_xdata()) == [0, 0] for line in bottom.lines)


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    hidden = set()
    for ax in figure.axes:
        low, high = sorted(ax.get_xlim())
        hidden |= {
            id(label)
            for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False)
            if not low <= tick <= high
        }
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and id(t) not in hidden:
            extent = t.get_window_extent()
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def test_main_saves_the_figure_from_the_three_result_files(f5, curves, areas, stability, tmp_path):
    curves.to_csv(tmp_path / "deletion_curves.csv", index=False)
    (tmp_path / "deletion_areas.json").write_text(json.dumps(areas))
    (tmp_path / "stability.json").write_text(json.dumps(stability))
    fig = f5.main(tmp_path)
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "f5_explanation_checks.pdf",
        "f5_explanation_checks.png",
    ]
    assert fig._suptitle.get_text() == f5.title(areas, stability)
    plt.close(fig)


def test_curve_labels_are_kept_apart_when_two_curves_end_close_together(
    f5, curves, areas, stability
):
    close = curves.copy()
    ends = (close["fraction"] == 0.5) & (close["screener"] == "deberta")
    close.loc[ends & (close["order"] == "random"), "mean_logodds_drop"] = 0.50
    close.loc[ends & (close["order"] == "lerf"), "mean_logodds_drop"] = 0.45
    fig = f5.build(close, areas, stability)
    top, _ = _axes(fig)
    ys = sorted(t.get_position()[1] for t in top.texts)
    span = 8.0 - 0.0  # the MoRF end is 8, the lowest end is at or above 0
    assert min(b - a for a, b in zip(ys, ys[1:], strict=False)) >= f5.LABEL_GAP * span - 1e-9
    plt.close(fig)
