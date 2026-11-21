import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
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


def _check(spearman, overlap, passed, thresholds):
    def stat(median):
        return {"median": median, "median_ci": [median - 0.05, median + 0.04]}

    return {
        "spearman": stat(spearman),
        "top10_overlap": stat(overlap),
        "passed": passed,
        "thresholds": {"median_spearman": thresholds[0], "median_top10_overlap": thresholds[1]},
    }


@pytest.fixture
def stability():
    return {
        "checks": {
            "seed": _check(0.89, 0.80, False, (0.9, 0.7)),
            "removal": _check(0.70, 0.60, False, (0.8, 0.5)),
            "method": _check(0.95, 0.90, True, (0.7, 0.5)),
        },
        "randomisation": {
            "spearman": {"median": 0.06, "median_ci": [0.02, 0.09]},
            "top10_overlap": {"median": 0.20, "median_ci": [0.20, 0.20]},
        },
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


def _set_passed(stability, *flags):
    for check, flag in zip(stability["checks"].values(), flags, strict=True):
        check["passed"] = flag


PASSED_DELETION = "Shapley finds the words the model relies on"
FAILED_DELETION = "Shapley's top words matter no more than random ones"
DEPENDS = "their order depends on how it is computed"
STEADY = "their order barely depends on how it is computed"


def _set_spearman(stability, name, low, high):
    stat = stability["checks"][name]["spearman"]
    stat["median"] = (low + high) / 2
    stat["median_ci"] = [low, high]


@pytest.mark.parametrize(
    ("deletion", "removal", "method", "expected"),
    [
        # the seed check (0.84 to 0.93 in the fixture) is the noise of rerunning the same method
        (True, (0.41, 0.53), (0.40, 0.43), f"{PASSED_DELETION}, but {DEPENDS}."),
        (True, (0.85, 0.90), (0.40, 0.43), f"{PASSED_DELETION}, but {DEPENDS}."),
        (True, (0.85, 0.90), (0.80, 0.95), f"{PASSED_DELETION}, and {STEADY}."),
        (
            True,
            (0.70, 0.84),
            (0.84, 0.90),
            f"{PASSED_DELETION}, and {STEADY}.",
        ),  # touching = overlap
        (False, (0.41, 0.53), (0.40, 0.43), f"{FAILED_DELETION}, and {DEPENDS}."),
        (False, (0.85, 0.90), (0.86, 0.95), f"{FAILED_DELETION}, and {STEADY}."),
    ],
)
def test_the_title_compares_other_ways_of_computing_with_the_noise_of_a_rerun(
    f5, areas, stability, deletion, removal, method, expected
):
    areas["screeners"]["deberta"]["passed"] = deletion
    _set_spearman(stability, "removal", *removal)
    _set_spearman(stability, "method", *method)
    assert f5.title(areas, stability) == expected


@pytest.mark.parametrize("deletion", [True, False])
@pytest.mark.parametrize("method", [(0.40, 0.43), (0.85, 0.95)])
def test_the_title_fits_in_two_lines_in_every_case(f5, curves, areas, stability, deletion, method):
    areas["screeners"]["deberta"]["passed"] = deletion
    _set_spearman(stability, "removal", *method)
    _set_spearman(stability, "method", *method)
    figure = f5.build(curves, areas, stability)
    figure.canvas.draw()
    title = figure._suptitle
    line_px = title.get_fontsize() * figure.dpi / 72 * 1.4
    assert title.get_window_extent().height <= 2 * line_px
    plt.close(figure)


def test_the_curves_are_the_main_models_with_morf_in_the_highlight_and_random_in_grey(figure):
    top, _ = _axes(figure)
    lines = {line.get_label(): line for line in top.lines if line.get_label() in DROPS}
    assert set(lines) == set(DROPS)
    for order, drops in DROPS.items():
        assert list(lines[order].get_ydata()) == pytest.approx(drops)
        assert list(lines[order].get_xdata()) == pytest.approx(FRACTIONS)
    assert _hex(lines["morf"].get_color()) == plotting.HIGHLIGHT
    assert _hex(lines["random"].get_color()) == plotting.NEUTRAL
    lerf = _hex(lines["lerf"].get_color())
    assert lerf[1:3] == lerf[3:5] == lerf[5:7]  # a grey
    assert 0x55 <= int(lerf[1:3], 16) <= 0x88  # a mid grey: darker than random's, not black
    assert lines["lerf"].get_linestyle() == "--"
    assert lines["random"].get_linestyle() == "-"


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
    assert (
        "highest value first" in " ".join(texts.split())
        and "lowest value first" in " ".join(texts.split())
        and "random" in texts
    )
    assert not any(word in texts for word in ("MoRF", "LeRF"))
    assert top.get_legend() is None


def test_the_deletion_axes_name_the_quantities_and_stop_at_half_of_the_words(figure):
    top, _ = _axes(figure)
    assert "log-odds" not in top.get_ylabel() and "score points" in top.get_ylabel()
    assert "words removed" in top.get_xlabel().lower()
    assert [t for t in top.get_xticks() if 0 <= t <= 0.5][-1] == 0.5
    assert top.get_xlim()[0] == 0
    assert top.get_xlim()[1] <= 0.75  # the data end at 0.5; the rest is room for the labels
    assert top.spines["bottom"].get_bounds() == (0, 0.5)


def test_checks_are_rows_sorted_by_the_median_correlation_and_say_what_changed(figure):
    _, bottom = _axes(figure)
    labels = [t.get_text().replace("\n", " ") for t in bottom.get_yticklabels()]
    assert labels == [
        "leave-one-out, not Shapley",
        "same method, another seed",
        "a blank instead of deleting",
        "every weight scrambled",
    ]
    assert not any(word in " ".join(labels) for word in ("passed", "failed"))
    assert "main run" in bottom.get_xlabel()
    assert bottom.yaxis_inverted()


def test_each_row_has_a_correlation_dot_in_the_highlight_and_an_overlap_dot_in_grey(figure):
    _, bottom = _axes(figure)
    dots = [c for c in bottom.collections if isinstance(c, matplotlib.collections.PathCollection)]
    correlation, overlap = dots
    assert correlation.get_offsets()[:, 0].tolist() == pytest.approx([0.95, 0.89, 0.70, 0.06])
    assert overlap.get_offsets()[:, 0].tolist() == pytest.approx([0.90, 0.80, 0.60, 0.20])
    assert {_hex(c) for c in correlation.get_facecolor()} == {plotting.HIGHLIGHT}
    assert {_hex(c) for c in overlap.get_facecolor()} == {plotting.NEUTRAL}
    intervals = [
        c for c in bottom.collections if isinstance(c, matplotlib.collections.LineCollection)
    ]
    first = [seg.tolist() for seg in intervals[0].get_segments()]
    assert [seg[0][0] for seg in first] == pytest.approx([0.90, 0.84, 0.65, 0.02])
    assert [seg[1][0] for seg in first] == pytest.approx([0.99, 0.93, 0.74, 0.09])


def test_the_two_measures_are_named_next_to_the_data_and_zero_is_marked(figure):
    _, bottom = _axes(figure)
    texts = " ".join(t.get_text() for t in bottom.texts)
    assert "rank correlation" in texts and "shared top-10 words" in texts
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


def _points_in(box, line):
    """Whether a point of `line`, sampled finely, falls inside the display box."""
    xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
    steps = np.linspace(0, len(xs) - 1, 400)
    points = np.column_stack(
        [np.interp(steps, range(len(xs)), xs), np.interp(steps, range(len(xs)), ys)]
    )
    x, y = line.get_transform().transform(points).T
    return bool(((x >= box.x0) & (x <= box.x1) & (y >= box.y0) & (y <= box.y1)).any())


def test_no_curve_label_sits_on_a_curve_or_on_the_zero_line(f5, curves, areas, stability):
    realistic = curves.copy()  # LeRF ends just above zero and random well above it, as in the data
    end = (realistic["fraction"] == 0.5) & (realistic["screener"] == "deberta")
    for order, value in (("morf", 9.7), ("random", 4.3), ("lerf", 0.2)):
        realistic.loc[end & (realistic["order"] == order), "mean_logodds_drop"] = value
    figure = f5.build(realistic, areas, stability)
    figure.canvas.draw()
    top, _ = _axes(figure)
    for text in top.texts:
        box = text.get_window_extent()
        for line in top.lines:
            assert not _points_in(box, line), (text.get_text(), line.get_label())


def test_value_labels_stand_right_of_their_interval_and_clear_their_dots(
    f5, curves, areas, stability
):
    measured = {"seed": (0.89, 0.80), "removal": (0.47, 0.50), "method": (0.42, 0.50)}
    for name, medians in measured.items():
        check = stability["checks"][name]
        for metric, median in zip(("spearman", "top10_overlap"), medians, strict=True):
            check[metric]["median"] = median
            check[metric]["median_ci"] = [median - 0.001, median + 0.03]
    figure = f5.build(curves, areas, stability)
    figure.canvas.draw()
    _, bottom = _axes(figure)
    dots = [c for c in bottom.collections if isinstance(c, matplotlib.collections.PathCollection)]
    radius = (dots[0].get_sizes()[0] ** 0.5) / 2 * figure.dpi / 72
    values = [t for t in bottom.texts if re.fullmatch(r"\d\.\d\d", t.get_text())]
    assert len(values) == 8
    for dot in dots:
        for x, y in dot.get_offsets():
            px, py = bottom.transData.transform((x, y))
            high = bottom.transData.transform((x + 0.03, y))[0]
            (label,) = [
                t
                for t in values
                if abs(t.get_window_extent().get_points()[:, 1].mean() - py) < 3
                and t.get_text() == f"{x:.2f}"
            ]
            box = label.get_window_extent()
            assert box.x0 >= max(px + radius, high) + 2, label.get_text()


def test_the_series_names_stand_right_of_the_value_labels_of_their_row(figure):
    _, bottom = _axes(figure)
    names = [t for t in bottom.texts if t.get_text() in ("rank correlation", "shared top-10 words")]
    assert len(names) == 2
    values = [t for t in bottom.texts if re.fullmatch(r"\d\.\d\d", t.get_text())]
    for name in names:
        box = name.get_window_extent()
        centre = box.get_points()[:, 1].mean()
        same_row = [
            v for v in values if abs(v.get_window_extent().get_points()[:, 1].mean() - centre) < 3
        ]
        assert same_row
        assert all(box.x0 > v.get_window_extent().x1 for v in same_row)


def test_no_pass_threshold_is_drawn(figure):
    _, bottom = _axes(figure)
    assert not [line for line in bottom.lines if line.get_marker() == "|"]
    assert "threshold" not in bottom.get_xlabel()


def test_curve_labels_stay_apart_when_curves_end_below_zero(f5, curves, areas, stability):
    below = curves.copy()
    ends = (below["fraction"] == 0.5) & (below["screener"] == "deberta")
    below.loc[ends & (below["order"] == "random"), "mean_logodds_drop"] = -2.9
    below.loc[ends & (below["order"] == "lerf"), "mean_logodds_drop"] = -3.0
    figure = f5.build(below, areas, stability)
    top, _ = _axes(figure)
    ys = sorted(t.get_position()[1] for t in top.texts)
    span = 8.0 - (-3.0)  # the MoRF end is 8, the lowest end -3
    assert min(b - a for a, b in zip(ys, ys[1:], strict=False)) >= f5.LABEL_GAP * span - 1e-9
    plt.close(figure)


def test_the_measure_names_stand_on_the_rows_of_the_last_check(figure):
    _, bottom = _axes(figure)
    last_row = {
        round(bottom.transData.transform((0, y))[1])
        for y in (3 - 0.18, 3 + 0.18)  # the correlation and overlap dots of the fourth row
    }
    names = [t for t in bottom.texts if t.get_text() in ("rank correlation", "shared top-10 words")]
    centres = {round(t.get_window_extent().get_points()[:, 1].mean()) for t in names}
    assert all(any(abs(c - y) <= 3 for y in last_row) for c in centres)


def test_the_dots_are_drawn_above_the_lines(figure):
    _, bottom = _axes(figure)
    dots = [c for c in bottom.collections if isinstance(c, matplotlib.collections.PathCollection)]
    lines = [c for c in bottom.collections if isinstance(c, matplotlib.collections.LineCollection)]
    assert min(d.get_zorder() for d in dots) > max(
        [line.get_zorder() for line in bottom.lines] + [c.get_zorder() for c in lines]
    )


def test_curve_labels_are_at_least_a_tenth_of_the_range_apart_and_close_to_their_curve_ends(
    f5, curves, areas, stability
):
    close = curves.copy()
    ends = (close["fraction"] == 0.5) & (close["screener"] == "deberta")
    close.loc[ends & (close["order"] == "random"), "mean_logodds_drop"] = 0.50
    close.loc[ends & (close["order"] == "lerf"), "mean_logodds_drop"] = 0.45
    figure = f5.build(close, areas, stability)
    figure.canvas.draw()
    top, _ = _axes(figure)
    ys = sorted(t.get_position()[1] for t in top.texts)
    assert min(b - a for a, b in zip(ys, ys[1:], strict=False)) >= 0.1 * 8.0
    end_x = top.transData.transform((0.5, 0))[0]
    for text in top.texts:
        gap = text.get_window_extent().x0 - end_x
        assert 2 <= gap <= 15, (text.get_text(), gap)
    plt.close(figure)


def test_panel_b_ticks_are_zero_one_half_and_one(figure):
    _, bottom = _axes(figure)
    assert [t for t in bottom.get_xticks() if 0 <= t <= 1] == [0, 0.5, 1]
