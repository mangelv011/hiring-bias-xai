import importlib.util
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f1.py"

# Eight occupations on the line gap = 1.2 * share - 0.6, each with a bootstrap interval of
# gap -+ 0.05. The Pearson r and its Fisher interval are written in as the statistics file would.
SHARES = {"o1": 0.1, "o2": 0.2, "o3": 0.3, "o4": 0.4, "o5": 0.6, "o6": 0.7, "o7": 0.8, "o8": 0.9}
GAPS = {name: 1.2 * share - 0.6 for name, share in SHARES.items()}
R, R_LOW, R_HIGH = 0.8547, 0.7071, 0.9309
SHOWCASE = ["o1", "o2", "o3", "o6", "o7", "o8"]


def _write(results, name, content):
    (results / name).write_text(json.dumps(content))


@pytest.fixture
def results(tmp_path):
    per_occupation = {
        name: {"tpr_gap": {"value": GAPS[name], "ci_low": GAPS[name] - 0.05, "ci_high": GAPS[name] + 0.05}}
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


@pytest.fixture
def figure(results):
    spec = importlib.util.spec_from_file_location("figure_f1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = module.main(results)
    yield fig
    plt.close(fig)


def test_the_figure_is_saved_as_pdf_and_png_under_the_results_directory(figure, results):
    saved = sorted(path.name for path in (results / "figures").iterdir())
    assert saved == ["f1_gap_vs_share.pdf", "f1_gap_vs_share.png"]


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    figure.canvas.draw()
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert texts
    assert min(t.get_fontsize() for t in texts) >= 7


def test_there_is_one_dot_per_occupation_at_its_share_and_gap(figure):
    (ax,) = figure.axes
    offsets = np.concatenate([c.get_offsets() for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)])
    expected = sorted((SHARES[o], GAPS[o]) for o in SHARES)
    assert np.array(sorted(map(tuple, offsets))) == pytest.approx(np.array(expected))


def _lines(ax):
    return [(line.get_xdata(), line.get_ydata()) for line in ax.get_lines()]


def test_there_is_a_zero_reference_line(figure):
    (ax,) = figure.axes
    assert any(np.all(np.asarray(y) == 0) for _, y in _lines(ax))


def test_the_fitted_line_is_the_least_squares_line_of_the_dots(figure):
    # The dots lie on gap = 1.2 * share - 0.6, so the OLS line has slope 1.2 and intercept -0.6.
    (ax,) = figure.axes
    fitted = [(x, y) for x, y in _lines(ax) if not np.all(np.asarray(y) == 0)]
    assert len(fitted) == 1
    x, y = fitted[0]
    assert np.polyfit(x, y, 1) == pytest.approx([1.2, -0.6])


def test_the_figure_states_pearson_r_with_its_fisher_interval(figure):
    (ax,) = figure.axes
    assert "r = 0.85 [0.71, 0.93]" in [t.get_text() for t in ax.texts]


def test_only_the_showcase_occupations_are_labelled_next_to_their_dots(figure):
    (ax,) = figure.axes
    labels = {t.get_text(): t for t in ax.texts if not t.get_text().startswith("r =")}
    assert sorted(labels) == sorted(o.replace("_", " ") for o in ["o1", "o2", "o3", "o6", "o7", "o8"])
    for name in ("o1", "o6"):
        x, y = labels[name].xy
        assert (x, y) == pytest.approx((SHARES[name], GAPS[name]))


def test_showcase_dots_have_the_highlight_colour_and_the_others_are_grey(figure):
    (ax,) = figure.axes
    colour_of = {}
    for c in ax.collections:
        if isinstance(c, matplotlib.collections.PathCollection):
            hex_colour = matplotlib.colors.to_hex(c.get_facecolor()[0]).upper()
            for x, y in c.get_offsets():
                colour_of[round(float(x), 3)] = hex_colour
    highlighted = {x for x, colour in colour_of.items() if colour == "#D55E00"}
    assert highlighted == {SHARES[o] for o in ["o1", "o2", "o3", "o6", "o7", "o8"]}
    assert {colour_of[SHARES["o4"]], colour_of[SHARES["o5"]]} == {"#999999"}


def test_labels_of_showcase_dots_that_nearly_coincide_do_not_overlap(results):
    # o1 and o2 get almost the same share and gap, so labels placed beside their dots would collide.
    metrics = json.loads((results / "metrics_deberta.json").read_text())
    gap = GAPS["o1"] + 0.004
    metrics["test"]["per_occupation"]["o2"]["tpr_gap"] = {
        "value": gap,
        "ci_low": gap - 0.05,
        "ci_high": gap + 0.05,
    }
    (results / "metrics_deberta.json").write_text(json.dumps(metrics))
    stats = json.loads((results / "gap_statistics.json").read_text())
    stats["main_model"]["occupations"]["o2"]["share_women"] = SHARES["o1"] + 0.01
    (results / "gap_statistics.json").write_text(json.dumps(stats))
    spec = importlib.util.spec_from_file_location("figure_f1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = module.main(results)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            matplotlib.text.Text.get_window_extent(t, renderer)  # the text, not its leader line
            for t in fig.axes[0].texts
            if t.get_text() in ("o1", "o2")
        ]
        assert len(boxes) == 2
        assert not boxes[0].overlaps(boxes[1])
    finally:
        plt.close(fig)


def test_the_title_is_the_finding_as_a_sentence_and_its_number_is_the_stored_r(figure, results):
    (ax,) = figure.axes
    title = figure.get_suptitle()
    assert title == "In male-dominated occupations the main model misses more women (r = 0.85)."
    stored = json.loads((results / "gap_statistics.json").read_text())["main_model"]["pearson"]
    numbers = [float(n) for n in re.findall(r"\d+\.\d+", title)]
    assert numbers == [round(stored["value"], 2)]
    assert ax.get_title() == ""


def test_the_title_changes_with_the_stored_r(results):
    stats = json.loads((results / "gap_statistics.json").read_text())
    stats["main_model"]["pearson"]["value"] = 0.5
    (results / "gap_statistics.json").write_text(json.dumps(stats))
    spec = importlib.util.spec_from_file_location("figure_f1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fig = module.main(results)
    try:
        assert "r = 0.50)" in fig.get_suptitle()
    finally:
        plt.close(fig)


def test_every_dot_has_its_bootstrap_interval_as_an_error_bar(figure):
    (ax,) = figure.axes
    bars = [
        segment
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
        for segment in c.get_segments()
    ]
    found = sorted((s[0][0], s[0][1], s[1][1]) for s in bars)
    expected = sorted((SHARES[o], GAPS[o] - 0.05, GAPS[o] + 0.05) for o in SHARES)
    assert np.array(found) == pytest.approx(np.array(expected))
