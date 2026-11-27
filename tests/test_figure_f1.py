import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hiring_bias_xai import plotting

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f1.py"

# Eight occupations near the line gap = 1.2 * share - 0.6 (each off it by a little), each with an
# asymmetric bootstrap interval [gap - 0.03, gap + 0.07]. The Pearson r and its Fisher interval
# are written in as the statistics file would.
SHARES = {"o1": 0.1, "o2": 0.2, "o3": 0.3, "o4": 0.4, "o5": 0.6, "o6": 0.7, "o7": 0.8, "o8": 0.9}
OFF_LINE = dict(zip(SHARES, [0.03, -0.04, 0.02, -0.05, 0.04, -0.03, 0.05, -0.02], strict=True))
GAPS = {name: 1.2 * share - 0.6 + OFF_LINE[name] for name, share in SHARES.items()}
BELOW, ABOVE = 0.03, 0.07
R, R_LOW, R_HIGH = 0.8547, 0.7071, 0.9309
SHOWCASE = ["o1", "o2", "o3", "o6", "o7", "o8"]
EXAMPLES = ("o1", "o8")  # the two labelled jobs; the real figure labels surgeon and nurse


def _write(results, name, content):
    (results / name).write_text(json.dumps(content))


@pytest.fixture
def results(tmp_path):
    per_occupation = {
        name: {
            "tpr_gap": {
                "value": GAPS[name],
                "ci_low": GAPS[name] - BELOW,
                "ci_high": GAPS[name] + ABOVE,
            },
            "tpr_female": {"value": 0.7 + GAPS[name] / 2},
            "tpr_male": {"value": 0.7 - GAPS[name] / 2},
        }
        for name in SHARES
    }
    _write(
        tmp_path,
        "metrics_deberta.json",
        {"test": {"per_occupation": per_occupation, "gap_share_pearson": {"value": R}}},
    )
    _write(
        tmp_path,
        "gap_statistics.json",
        {
            "main_model": {
                "pearson": {"value": R, "fisher_ci_low": R_LOW, "fisher_ci_high": R_HIGH},
                "occupations": {name: {"share_women": share} for name, share in SHARES.items()},
            }
        },
    )
    _write(
        tmp_path,
        "showcase_occupations.json",
        {
            "female_dominated": [{"occupation": o, "share_women": SHARES[o]} for o in SHOWCASE[3:]],
            "male_dominated": [{"occupation": o, "share_women": SHARES[o]} for o in SHOWCASE[:3]],
        },
    )
    return tmp_path


def _render(results, examples=EXAMPLES):
    spec = importlib.util.spec_from_file_location("figure_f1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXAMPLES = examples
    return module.main(results)


@pytest.fixture
def figure(results):
    fig = _render(results)
    yield fig
    plt.close(fig)


def test_the_figure_is_saved_as_pdf_and_png_under_the_results_directory(figure, results):
    saved = sorted(path.name for path in (results / "figures").iterdir())
    assert saved == ["f1_gap_vs_share.pdf", "f1_gap_vs_share.png"]


def test_the_figure_is_3_03_inches_wide_and_no_text_is_below_7_pt(figure):
    figure.canvas.draw()
    assert figure.get_size_inches()[0] == plotting.WIDTH_INCHES == 3.03
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert texts
    assert min(t.get_fontsize() for t in texts) >= 7


def test_there_is_one_dot_per_occupation_at_its_share_and_gap(figure):
    (ax,) = figure.axes
    offsets = np.concatenate(
        [
            c.get_offsets()
            for c in ax.collections
            if isinstance(c, matplotlib.collections.PathCollection)
        ]
    )
    expected = sorted((SHARES[o], GAPS[o]) for o in SHARES)
    assert np.array(sorted(map(tuple, offsets))) == pytest.approx(np.array(expected))


def _lines(ax):
    return [(line.get_xdata(), line.get_ydata()) for line in ax.get_lines()]


def test_there_is_a_zero_reference_line(figure):
    (ax,) = figure.axes
    assert any(np.all(np.asarray(y) == 0) for _, y in _lines(ax))


def test_the_fitted_line_is_the_least_squares_line_of_the_dots(figure):
    # The dots scatter around a line, so a straight line through them is the least-squares one.
    (ax,) = figure.axes
    fitted = [(x, y) for x, y in _lines(ax) if not np.all(np.asarray(y) == 0)]
    assert len(fitted) == 1
    x, y = fitted[0]
    expected = np.polyfit(list(SHARES.values()), list(GAPS.values()), 1)
    assert np.polyfit(x, y, 1) == pytest.approx(expected)
    assert (min(x), max(x)) == pytest.approx((min(SHARES.values()), max(SHARES.values())))


def test_the_figure_states_pearson_r_with_its_fisher_interval(figure):
    (ax,) = figure.axes
    assert "r = 0.85 [0.71, 0.93]" in [t.get_text() for t in ax.texts]


def test_only_the_two_example_jobs_are_labelled_next_to_their_dots(figure):
    (ax,) = figure.axes
    labels = {t.get_text().split(":")[0]: t for t in ax.texts if hasattr(t, "xyann")}
    assert sorted(labels) == sorted(EXAMPLES)
    for name in EXAMPLES:
        x, y = labels[name].xy
        assert (x, y) == pytest.approx((SHARES[name], GAPS[name]))


def test_showcase_dots_have_the_highlight_colour_and_the_others_are_grey(figure):
    (ax,) = figure.axes
    colour_of = {}
    for c in ax.collections:
        if isinstance(c, matplotlib.collections.PathCollection):
            hex_colour = matplotlib.colors.to_hex(c.get_facecolor()[0]).upper()
            for x, _ in c.get_offsets():
                colour_of[round(float(x), 3)] = hex_colour
    highlighted = {x for x, colour in colour_of.items() if colour == "#D55E00"}
    assert highlighted == {SHARES[o] for o in ["o1", "o2", "o3", "o6", "o7", "o8"]}
    assert {colour_of[SHARES["o4"]], colour_of[SHARES["o5"]]} == {"#999999"}


def test_labels_of_example_dots_that_nearly_coincide_do_not_overlap(results):
    # o1 and o2 get almost the same share and gap, so labels placed beside their dots would collide.
    metrics = json.loads((results / "metrics_deberta.json").read_text())
    gap = GAPS["o1"] - 0.004
    metrics["test"]["per_occupation"]["o2"]["tpr_gap"] = {
        "value": gap,
        "ci_low": gap - BELOW,
        "ci_high": gap + ABOVE,
    }
    (results / "metrics_deberta.json").write_text(json.dumps(metrics))
    stats = json.loads((results / "gap_statistics.json").read_text())
    stats["main_model"]["occupations"]["o2"]["share_women"] = SHARES["o1"] + 0.01
    (results / "gap_statistics.json").write_text(json.dumps(stats))
    fig = _render(results, examples=("o1", "o2"))
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            matplotlib.text.Text.get_window_extent(t, renderer)  # the text, not its leader line
            for t in fig.axes[0].texts
            if t.get_text().split(":")[0] in ("o1", "o2")
        ]
        assert len(boxes) == 2
        assert not boxes[0].overlaps(boxes[1])
    finally:
        plt.close(fig)


FINDING = "The more a job is held by one gender, the more the model favours that gender."


def test_the_title_states_the_finding_in_plain_words(figure):
    (ax,) = figure.axes
    assert figure.get_suptitle() == FINDING
    assert ax.get_title() == ""


def test_the_title_says_there_is_no_clear_link_when_the_interval_of_r_includes_0(results):
    stats = json.loads((results / "gap_statistics.json").read_text())
    stats["main_model"]["pearson"].update(value=0.2, fisher_ci_low=-0.1, fisher_ci_high=0.5)
    (results / "gap_statistics.json").write_text(json.dumps(stats))
    fig = _render(results)
    try:
        assert (
            fig.get_suptitle()
            == "No clear link between who holds a job and whom the model favours."
        )
    finally:
        plt.close(fig)


def test_the_axes_speak_of_jobs_and_candidates_shortlisted(figure):
    (ax,) = figure.axes
    figure.canvas.draw()
    assert ax.get_xlabel() == "Share of women in the job"
    assert "per 100" in ax.get_ylabel() and "shortlisted" in ax.get_ylabel()
    assert "TPR" not in ax.get_ylabel()
    xticks = {t.get_text() for t in ax.get_xticklabels() if t.get_text()}
    assert {"20%", "60%", "100%"} <= xticks
    yticks = {t.get_text() for t in ax.get_yticklabels() if t.get_text()}
    assert {"0", "+20", "-20"} <= {s.replace("−", "-") for s in yticks}


def test_the_direction_of_the_gap_is_written_above_and_below_zero(figure):
    (ax,) = figure.axes
    figure.canvas.draw()
    notes = {t.get_text(): t for t in ax.texts if "favours" in t.get_text()}
    assert set(notes) == {"↑ favours women", "↓ favours men"}
    zero = ax.transData.transform((0, 0))[1]
    renderer = figure.canvas.get_renderer()
    assert notes["↑ favours women"].get_window_extent(renderer).y0 > zero
    assert notes["↓ favours men"].get_window_extent(renderer).y1 < zero


def test_two_example_jobs_carry_their_numbers_per_100(results):
    fig = _render(results)
    try:
        (ax,) = fig.axes
        texts = [t.get_text() for t in ax.texts]
        women_o1, men_o1 = round(100 * (0.7 + GAPS["o1"] / 2)), round(100 * (0.7 - GAPS["o1"] / 2))
        assert f"o1: {women_o1} of 100 women,\n{men_o1} of 100 men shortlisted" in texts
        assert any(t.startswith("o8: ") for t in texts)
        assert not any(t.startswith(("o2", "o3", "o4", "o5", "o6", "o7")) for t in texts)
        _assert_labels_are_clear(fig)
    finally:
        plt.close(fig)


def test_only_the_showcase_dots_have_their_bootstrap_interval_as_an_error_bar(figure):
    (ax,) = figure.axes
    bars = [
        segment
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
        for segment in c.get_segments()
    ]
    found = sorted((s[0][0], s[0][1], s[1][1]) for s in bars)
    expected = sorted((SHARES[o], GAPS[o] - BELOW, GAPS[o] + ABOVE) for o in SHOWCASE)
    assert np.array(found) == pytest.approx(np.array(expected))
    bar_colours = {
        matplotlib.colors.to_hex(c.get_color()[0]).upper()
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
    }
    assert bar_colours == {"#999999"}


def test_a_long_label_beside_the_rightmost_dot_stays_inside_the_axes(results):
    long_name = "dietitian"
    for name in ("metrics_deberta.json", "gap_statistics.json", "showcase_occupations.json"):
        path = results / name
        path.write_text(path.read_text().replace('"o8"', f'"{long_name}"'))
    fig = _render(results, examples=(long_name,))
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        (ax,) = fig.axes
        label = next(t for t in ax.texts if t.get_text().startswith(long_name))
        box = matplotlib.text.Text.get_window_extent(label, renderer)
        assert box.x1 <= ax.get_window_extent(renderer).x1
    finally:
        plt.close(fig)


def test_the_wrapped_title_fits_inside_the_figure(figure):
    figure.canvas.draw()
    box = figure._suptitle.get_window_extent(figure.canvas.get_renderer())
    assert 0 <= box.x0 and box.x1 <= figure.bbox.x1


def _label_boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (ax,) = fig.axes
    return {
        t.get_text(): matplotlib.text.Text.get_window_extent(t, renderer)
        for t in ax.texts
        if hasattr(t, "xyann")
    }


def _assert_labels_are_clear(fig):
    (ax,) = fig.axes
    boxes = _label_boxes(fig)
    fitted = next(line for line in ax.get_lines() if not np.all(np.asarray(line.get_ydata()) == 0))
    path = fitted.get_path().transformed(fitted.get_transform())
    dots = np.concatenate(
        [
            ax.transData.transform(c.get_offsets())
            for c in ax.collections
            if isinstance(c, matplotlib.collections.PathCollection)
        ]
    )
    renderer = fig.canvas.get_renderer()
    fixed = {
        t.get_text(): t.get_window_extent(renderer) for t in ax.texts if not hasattr(t, "xyann")
    }
    for label in ax.texts:
        arrow = getattr(label, "arrow_patch", None)
        if arrow is None:
            continue
        leader = arrow.get_path().transformed(arrow.get_transform())
        for name, box in (boxes | fixed).items():
            if name != label.get_text():
                assert not leader.intersects_bbox(box, filled=False), (
                    f"{label.get_text()} leader crosses {name}"
                )
    names = list(boxes)
    for i, name in enumerate(names):
        box = boxes[name]
        assert not path.intersects_bbox(box, filled=False), f"{name} crosses the fitted line"
        for text, extent in fixed.items():
            assert not box.overlaps(extent), f"{name} overlaps {text!r}"
        for other in names[i + 1 :]:
            assert not box.overlaps(boxes[other]), f"{name} overlaps {other}"
        inside = (
            (dots[:, 0] > box.x0 - 3)
            & (dots[:, 0] < box.x1 + 3)
            & (dots[:, 1] > box.y0 - 3)
            & (dots[:, 1] < box.y1 + 3)
        )
        assert not inside.any(), f"{name} covers a dot"


def test_labels_do_not_cross_the_fitted_line_or_each_other_or_cover_a_dot(figure):
    _assert_labels_are_clear(figure)


def test_the_committed_f1_has_clear_labels():
    spec = importlib.util.spec_from_file_location("figure_f1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = module.build(SCRIPT.parents[1] / "results")
    try:
        _assert_labels_are_clear(fig)
        _assert_labels_are_clear(fig)
    finally:
        plt.close(fig)
