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
    "a": (0.780, 0.200),
    "b": (0.770, 0.160),
    "c": (0.775, 0.140),
    "d": (0.775, 0.150),
    "e": (0.779, 0.195),
}


@pytest.fixture
def report():
    """A mitigation.json with the keys the figure reads, and some it must ignore."""
    conditions = {}
    for name, (f1, gap) in CONDITIONS.items():
        block = {
            "macro_f1": _metric(f1, 0.005),
            "accuracy": _metric(0.5, 0.1),
            "rms_tpr_gap": _metric(gap, 0.02) | {"null": {"mean": 0.04, "p95": 0.05}},
        }
        if name != "a":
            block["change_from_a"] = {
                "macro_f1": _metric(f1 - 0.780, 0.003),
                "rms_tpr_gap": _metric(gap - 0.200, 0.01),
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


def _hex(colours):
    return [matplotlib.colors.to_hex(c).upper() for c in colours]


def test_the_title_says_the_gap_falls_and_what_it_costs_when_the_intervals_exclude_zero(f7, report):
    # d: gap 0.200 to 0.150 with a change of -0.05 [-0.06, -0.04]; macro-F1 -0.005 [-0.008, -0.002]
    assert f7.title(report) == (
        "Scrubbing pronouns cuts the gender gap from 0.20 to 0.15, costing 0.5 points of macro-F1; "
        "chance alone gives 0.04"
    )


def test_the_title_says_no_change_when_the_interval_of_the_change_includes_zero(f7, report):
    d = report["conditions"]["d"]["outside_showcase"]["change_from_a"]
    d["rms_tpr_gap"] = _metric(-0.05, 0.06)
    d["macro_f1"] = _metric(-0.005, 0.01)
    assert f7.title(report) == (
        "Scrubbing pronouns does not change the gender gap (0.20 to 0.15), "
        "with no change in macro-F1; chance alone gives 0.04"
    )


def test_the_title_says_the_gap_rises_and_macro_f1_rises_when_both_intervals_are_above_zero(
    f7, report
):
    d = report["conditions"]["d"]["outside_showcase"]["change_from_a"]
    d["rms_tpr_gap"] = _metric(0.05, 0.01)
    d["macro_f1"] = _metric(0.012, 0.002)
    report["conditions"]["d"]["outside_showcase"]["rms_tpr_gap"]["value"] = 0.25
    assert f7.title(report) == (
        "Scrubbing pronouns raises the gender gap from 0.20 to 0.25, and raises macro-F1 by 1.2 "
        "points; chance alone gives 0.04"
    )


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
    assert [[f1_of_a - 0.005, gap_of_a], [f1_of_a + 0.005, gap_of_a]] in horizontal
    assert [[f1_of_a, gap_of_a - 0.02], [f1_of_a, gap_of_a + 0.02]] in vertical


def test_an_arrow_runs_from_the_original_model_to_each_other_condition(figure):
    ax = _ax(figure)
    arrows = [t for t in ax.texts if t.arrow_patch is not None]
    ends = sorted(tuple(t.xy) for t in arrows if t.get_text() == "")
    assert ends == sorted(v for k, v in CONDITIONS.items() if k != "a")
    for arrow in (t for t in arrows if t.get_text() == ""):
        assert tuple(arrow.xyann) == CONDITIONS["a"]


def test_the_band_of_the_gap_that_chance_gives_runs_from_the_null_mean_to_its_95th_percentile(
    figure,
):
    ax = _ax(figure)
    (band,) = [p for p in ax.patches if isinstance(p, matplotlib.patches.Polygon)]
    ys = band.get_xy()[:, 1]
    assert (ys.min(), ys.max()) == pytest.approx((0.04, 0.05))
    assert any("chance" in t.get_text() for t in ax.texts)


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
    assert ax.get_ylabel() == "Gender gap: RMS TPR gap (lower is better)"
    assert ax.get_ylim()[0] == 0


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible():
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
