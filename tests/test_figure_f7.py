import copy
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.text
import pytest

from hiring_bias_xai import plotting

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f7.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f7", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f7():
    return _load_script()


def _metric(value, half_width):
    return {"value": value, "ci_low": value - half_width, "ci_high": value + half_width}


# (macro-F1, RMS TPR gap) of each condition, and the change from a of the two
CONDITIONS = {
    "a": (0.7835, 0.1435),
    "b": (0.7759, 0.1188),
    "c": (0.7797, 0.1104),
    "d": (0.7788, 0.1152),
    "e": (0.7819, 0.1412),
}
F1_HALF, GAP_HALF = 0.0043, 0.0135


@pytest.fixture
def report():
    """A mitigation.json with the keys the figure reads, and some it must ignore."""
    conditions = {}
    for name, (f1, gap) in CONDITIONS.items():
        block = {
            "macro_f1": _metric(f1, F1_HALF),
            "accuracy": _metric(0.5, 0.1),
            "rms_tpr_gap": _metric(gap, GAP_HALF) | {"null": {"mean": 0.04, "p95": 0.05}},
        }
        if name != "a":
            block["change_from_a"] = {
                "macro_f1": _metric(f1 - CONDITIONS["a"][0], 0.003),
                "rms_tpr_gap": _metric(gap - CONDITIONS["a"][1], 0.01),
            }
        conditions[name] = {"outside_showcase": block, "full_test": {"macro_f1": _metric(0.1, 0.1)}}
    return {"conditions": conditions}


@pytest.fixture
def figure(f7, report):
    fig = f7.build(report)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _ax(fig):
    (ax,) = [a for a in fig.axes if a.collections]
    return ax


NO_CLEAR_CHANGE = "Removing pronouns leaves no clear change in the gap between women and men."


def _hex(colours):
    return [matplotlib.colors.to_hex(c).upper() for c in colours]


def test_the_title_says_by_how_much_the_gap_narrows_when_the_interval_excludes_zero(f7, report):
    # d: gap 0.1435 to 0.1152, a fall of 20%, change -0.028 [-0.038, -0.018]
    assert f7.title(report) == (
        "Removing pronouns narrows the gap between women and men by 20%; most of it remains."
    )


def test_the_title_drops_most_of_it_remains_when_the_gap_falls_by_half_or_more(f7, report):
    report["conditions"]["d"]["outside_showcase"]["rms_tpr_gap"]["value"] = 0.0717
    assert f7.title(report) == "Removing pronouns narrows the gap between women and men by 50%."


def test_the_title_says_no_change_when_the_interval_of_the_change_includes_zero(f7, report):
    d = report["conditions"]["d"]["outside_showcase"]["change_from_a"]
    d["rms_tpr_gap"] = _metric(-0.05, 0.06)
    d["macro_f1"] = _metric(-0.005, 0.01)
    assert f7.title(report) == NO_CLEAR_CHANGE


def test_the_title_says_the_gap_widens_when_the_interval_is_above_zero(f7, report):
    d = report["conditions"]["d"]["outside_showcase"]["change_from_a"]
    d["rms_tpr_gap"] = _metric(0.05, 0.01)
    d["macro_f1"] = _metric(0.012, 0.002)
    report["conditions"]["d"]["outside_showcase"]["rms_tpr_gap"]["value"] = 0.25
    assert f7.title(report) == "Removing pronouns widens the gap between women and men by 74%."


def test_the_title_fits_in_two_lines(figure):
    title = figure._suptitle
    line = title.get_fontsize() * 1.5 * figure.dpi / 72
    assert title.get_window_extent().height <= 2 * line


def test_each_condition_is_a_dot_at_its_macro_f1_and_rms_gap_and_d_is_the_highlight(
    figure,
):
    ax = _ax(figure)
    dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    offsets = sorted(map(tuple, (o for d in dots for o in d.get_offsets().tolist())))
    assert offsets == sorted(CONDITIONS.values())
    colours = {tuple(o): _hex(d.get_facecolor())[0] for d in dots for o in d.get_offsets().tolist()}
    assert colours[CONDITIONS["d"]] == plotting.HIGHLIGHT
    assert [colours[CONDITIONS[c]] for c in "bce"] == [plotting.NEUTRAL] * 3


def test_every_dot_has_a_horizontal_and_a_vertical_95_percent_interval(figure):
    ax = _ax(figure)
    segments = [
        seg.tolist()
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
        for seg in c.get_segments()
    ]
    horizontal = sorted(s for s in segments if s[0][1] == s[1][1])
    vertical = sorted(s for s in segments if s[0][0] == s[1][0])
    assert len(horizontal) == len(vertical) == 5
    f1_of_a, gap_of_a = CONDITIONS["a"]
    assert [[f1_of_a - F1_HALF, gap_of_a], [f1_of_a + F1_HALF, gap_of_a]] in horizontal
    assert [[f1_of_a, gap_of_a - GAP_HALF], [f1_of_a, gap_of_a + GAP_HALF]] in vertical


def test_an_arrow_runs_from_the_original_model_to_each_other_condition(figure):
    ax = _ax(figure)
    arrows = [t for t in ax.texts if getattr(t, "arrow_patch", None) is not None]
    ends = sorted(tuple(t.xy) for t in arrows if t.get_text() == "")
    assert ends == sorted(v for k, v in CONDITIONS.items() if k != "a")
    for arrow in (t for t in arrows if t.get_text() == ""):
        assert tuple(arrow.xyann) == CONDITIONS["a"]


def test_the_band_of_the_gap_that_chance_gives_runs_from_the_null_mean_to_its_95th_percentile(
    figure,
):
    ax = _ax(figure)
    (band,) = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert (band.get_y(), band.get_y() + band.get_height()) == pytest.approx((0.04, 0.05))
    assert any(t.get_text() == "chance alone: 0.04 to 0.05" for t in ax.texts)


def test_the_better_corner_is_marked_and_the_conditions_are_named_without_a_legend(figure):
    ax = _ax(figure)
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "better" in texts
    for name in ("original", "scrubbed"):
        assert name in texts
    assert ax.get_legend() is None


def test_the_axes_name_the_quantities_and_the_vertical_axis_starts_at_zero(figure):
    ax = _ax(figure)
    assert ax.get_xlabel() == "Macro-F1 (higher is better)"
    assert " ".join(ax.get_ylabel().split()) == (
        "Shortlist gap, test bios outside the showcase subset (lower is better)"
    )
    assert "RMS" not in ax.get_ylabel()
    assert ax.get_ylim()[0] == 0


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
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


def test_the_labels_do_not_overlap_each_other(figure):
    ax = _ax(figure)
    boxes = [(t.get_text(), t.get_window_extent()) for t in ax.texts if t.get_text().strip()]
    for i, (name, box) in enumerate(boxes):
        for other, other_box in boxes[i + 1 :]:
            assert not box.overlaps(other_box), (name, other)


def test_main_saves_the_figure_from_mitigation_json(f7, report, tmp_path):
    (tmp_path / "mitigation.json").write_text(json.dumps(report))
    fig = f7.main(tmp_path)
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "f7_mitigation.pdf",
        "f7_mitigation.png",
    ]
    assert fig._suptitle.get_text() == f7.title(copy.deepcopy(report))
    plt.close(fig)


def test_a_change_whose_interval_only_touches_zero_is_not_called_a_fall(f7, report):
    d = report["conditions"]["d"]["outside_showcase"]["change_from_a"]
    d["rms_tpr_gap"] = {"value": -0.01, "ci_low": -0.02, "ci_high": 0.0}
    d["macro_f1"] = {"value": 0.01, "ci_low": 0.0, "ci_high": 0.02}
    assert f7.title(report) == NO_CLEAR_CHANGE


def test_the_chance_band_is_a_translucent_grey(figure):
    (band,) = [p for p in _ax(figure).patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert 0 < band.get_alpha() < 1
    assert _hex([band.get_facecolor()[:3]]) == [plotting.NEUTRAL]


def _annotation(fig, text):
    (note,) = [t for t in _ax(fig).texts if t.get_text() == text]
    return note


def test_each_condition_has_its_label_beside_its_dot(figure):
    # text offset in points from the dot, horizontal alignment
    labels = {
        "a original": ((6, 0), "left"),
        "b scrubbed bios": ((-6, 3), "right"),
        "c scrubbed model": ((7, -4), "left"),
        "d scrubbed both": ((-7, -8), "right"),
        "e proxy words deleted": ((-3, 9), "right"),
    }
    for text, (offset, align) in labels.items():
        note = _annotation(figure, text)
        assert tuple(note.xyann) == offset
        assert (note.get_ha(), note.get_va(), note.anncoords) == (align, "center", "offset points")


def test_only_the_label_of_d_has_the_highlight_colour(figure):
    assert _hex([_annotation(figure, "d scrubbed both").get_color()]) == [plotting.HIGHLIGHT]
    for text in ("a original", "b scrubbed bios", "c scrubbed model", "e proxy words deleted"):
        assert _hex([_annotation(figure, text).get_color()]) == ["#333333"]


def test_the_better_arrow_points_to_the_corner_of_high_macro_f1_and_low_gap(figure):
    note = _annotation(figure, "better")
    assert (note.xycoords, note.anncoords) == ("axes fraction", "axes fraction")
    assert (tuple(note.xy), tuple(note.xyann), note.get_ha()) == (
        (0.97, 0.03),
        (0.80, 0.12),
        "center",
    )


def test_the_arrows_have_a_head_the_colour_of_their_dot_and_stop_short_of_the_dots(figure):
    ax = _ax(figure)
    arrows = [t for t in ax.texts if t.get_text() == "" and t.arrow_patch is not None]
    assert len(arrows) == 4
    for arrow in arrows:
        patch = arrow.arrow_patch
        assert isinstance(patch.get_arrowstyle(), matplotlib.patches.ArrowStyle.CurveB)
        assert (patch.shrinkA, patch.shrinkB) == (3, 3)
        expected = plotting.HIGHLIGHT if tuple(arrow.xy) == CONDITIONS["d"] else plotting.NEUTRAL
        assert _hex([patch.get_edgecolor()[:3]]) == [expected]


def test_the_dots_are_drawn_above_the_interval_lines(figure):
    ax = _ax(figure)
    dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    lines = [c for c in ax.collections if isinstance(c, matplotlib.collections.LineCollection)]
    assert min(d.get_zorder() for d in dots) > max(line.get_zorder() for line in lines)
    assert {tuple(line.get_linewidth()) for line in lines} == {(1.0,)}


def test_the_axes_leave_room_for_the_labels_beyond_the_intervals(figure):
    # Left 30% and right 50% of the span of the macro-F1 intervals, 12% above the highest gap.
    x_low = min(f1 for f1, _ in CONDITIONS.values()) - F1_HALF
    x_high = max(f1 for f1, _ in CONDITIONS.values()) + F1_HALF
    y_high = max(gap for _, gap in CONDITIONS.values()) + GAP_HALF
    span = x_high - x_low
    ax = _ax(figure)
    assert ax.get_xlim() == pytest.approx((x_low - 0.3 * span, x_high + 0.5 * span))
    assert ax.get_ylim() == pytest.approx((0, y_high * 1.12))


def test_the_chance_label_sits_in_the_middle_of_the_band_left_of_the_intervals(figure):
    x_low = min(f1 for f1, _ in CONDITIONS.values()) - F1_HALF
    x_high = max(f1 for f1, _ in CONDITIONS.values()) + F1_HALF
    note = _annotation(figure, "chance alone: 0.04 to 0.05")
    assert note.get_position() == pytest.approx((x_low - 0.22 * (x_high - x_low), 0.045))
    assert (note.get_ha(), note.get_va()) == ("left", "center")
    assert _hex([note.get_color()]) == ["#333333"]


def test_the_band_is_that_of_the_original_model_and_has_no_edge(f7, report):
    report["conditions"]["d"]["outside_showcase"]["rms_tpr_gap"]["null"] = {"mean": 0.3, "p95": 0.4}
    fig = f7.build(report)
    fig.canvas.draw()
    (band,) = [p for p in _ax(fig).patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert (band.get_y(), band.get_y() + band.get_height()) == pytest.approx((0.04, 0.05))
    assert (band.get_alpha(), band.get_linewidth()) == (0.3, 0)
    plt.close(fig)


def test_the_title_is_bold_and_the_figure_four_inches_high(figure):
    assert figure._suptitle.get_fontweight() == "bold"
    assert figure.get_size_inches()[1] == 4.0
