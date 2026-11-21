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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f8.py"


SAME = "With a woman's or a man's name, qualified candidates get yes about equally often."


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f8", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f8():
    return _load_script()


def _stat(value, half):
    return {"value": value, "ci_low": value - half, "ci_high": value + half, "p": 0.5}


# change in TPR from the none arm, in yes-rate (0.01 is one point)
ARMS = {
    "neutral": -0.010,
    "swap": 0.004,
    "name|Spain + Hispanic America|female": 0.020,
    "name|Spain + Hispanic America|male": -0.002,
    "name|Morocco|female": -0.015,
    "name|Morocco|male": 0.001,
}
HALF = 0.012


@pytest.fixture
def report():
    statistics = {f"diff_tpr|{arm}": _stat(v, HALF) for arm, v in ARMS.items()}
    statistics["primary"] = _stat(0.012, 0.015)
    return {
        "statistics": statistics,
        "noise_floor": {
            "rerun": {"tpr_diff": _stat(0.0, 0.0)},
            "edit": {"tpr_diff": _stat(-0.004, 0.006)},
        },
        "rationalization": {
            "name_arms_above_noise": 0,
            "beyond_noise": None,
            "all_name_and_swap_flips": {"n_flips": 90},
        },
    }


@pytest.fixture
def figure(f8, report):
    fig = f8.build(report)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _ax(fig):
    (ax,) = [a for a in fig.axes if a.collections]
    return ax


def _dots(ax):
    dots = [c for c in ax.collections if isinstance(c, matplotlib.collections.PathCollection)]
    return [
        (x, y, matplotlib.colors.to_hex(d.get_facecolor()[0]).upper())
        for d in dots
        for x, y in d.get_offsets().tolist()
    ]


def test_the_title_says_in_plain_words_whether_women_s_and_men_s_names_get_the_same_answers(
    f8, report
):
    assert f8.title(report) == SAME
    report["statistics"]["primary"] = _stat(0.03, 0.01)
    assert f8.title(report) == (
        "With a woman's name, 3 more qualified candidates per 100 get yes than with a man's."
    )
    report["statistics"]["primary"] = _stat(-0.03, 0.01)
    assert f8.title(report) == (
        "With a woman's name, 3 fewer qualified candidates per 100 get yes than with a man's."
    )


def test_the_title_fits_in_two_lines(figure):
    title = figure._suptitle
    line = title.get_fontsize() * 1.5 * figure.dpi / 72
    assert title.get_window_extent().height <= 2 * line


def test_each_arm_and_the_primary_contrast_is_a_dot_in_points_sorted_by_value(figure):
    ax = _ax(figure)
    dots = _dots(ax)
    assert len(dots) == len(ARMS) + 1
    primary = max(dots, key=lambda d: d[1])
    assert primary[0] == pytest.approx(1.2) and primary[2] == plotting.HIGHLIGHT
    arms = sorted((d for d in dots if d is not primary), key=lambda d: d[1])
    assert [d[2] for d in arms] == [plotting.NEUTRAL] * len(ARMS)
    values = [d[0] for d in arms]
    assert values == pytest.approx(sorted(100 * v for v in ARMS.values()))
    assert primary[1] > arms[-1][1]


def test_every_dot_has_a_95_percent_interval_and_there_is_a_line_at_zero(figure):
    ax = _ax(figure)
    segments = [
        seg.tolist()
        for c in ax.collections
        if isinstance(c, matplotlib.collections.LineCollection)
        for seg in c.get_segments()
    ]
    assert len(segments) == len(ARMS) + 1
    assert all(s[0][1] == s[1][1] for s in segments)
    widths = sorted(round(s[1][0] - s[0][0], 6) for s in segments)
    assert widths == [2.4] * 6 + [3.0]  # arms 2 x 1.2 points, the primary contrast 2 x 1.5
    assert any(line.get_xdata()[0] == 0 for line in ax.lines)


def test_the_band_is_the_range_of_the_noise_floor_intervals_in_points(figure):
    ax = _ax(figure)
    (band,) = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
    assert (band.get_x(), band.get_x() + band.get_width()) == pytest.approx((-1.0, 0.2))
    assert any("noise" in t.get_text() for t in ax.texts)


def test_the_annotation_says_the_decision_barely_depends_on_the_name_when_no_arm_is_above_noise(
    figure,
):
    text = " ".join(" ".join(t.get_text().split()) for t in _ax(figure).texts)
    assert "barely depends on the name" in text


def test_the_annotation_gives_the_share_without_a_mention_when_arms_are_above_noise(f8, report):
    share = {"value": 0.9, "ci_low": 0.8, "ci_high": 0.95}
    report["rationalization"]["beyond_noise"] = {"n_flips": 40, "share_no_mention": share}
    report["rationalization"]["name_arms_above_noise"] = 2
    fig = f8.build(report)
    fig.canvas.draw()
    text = " ".join(" ".join(t.get_text().split()) for t in _ax(fig).texts)
    assert "Of 40 changed decisions, 90% (80 to 95) give a reason that does not mention" in text
    plt.close(fig)


def test_arms_are_labelled_directly_without_a_legend_and_the_axis_is_per_100_qualified(figure):
    ax = _ax(figure)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for name in ("neutral", "pronouns", "Morocco", "woman", "man"):
        assert any(name in label for label in labels)
    assert ax.get_legend() is None
    assert " ".join(ax.get_xlabel().split()) == (
        "Change in yes per 100 qualified candidates, against no name"
    )


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    ax = _ax(figure)
    low, high = sorted(ax.get_xlim())
    hidden = {
        id(label)
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False)
        if not low <= tick <= high
    }
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible() and id(t) not in hidden:
            extent = t.get_window_extent()
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def test_a_gap_is_claimed_only_when_the_interval_excludes_zero_not_when_it_touches_it(f8, report):
    report["statistics"]["primary"] = _stat(0.01, 0.01)
    assert f8.title(report) == SAME
    report["statistics"]["primary"] = _stat(-0.01, 0.01)
    assert f8.title(report) == SAME


def test_each_row_is_labelled_with_the_origin_and_gender_of_its_arm(figure):
    ax = _ax(figure)
    dots = _dots(ax)
    labels = {t.get_text(): y for y, t in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)}
    for label, arm in (
        ("Morocco, woman", "name|Morocco|female"),
        ("Morocco, man", "name|Morocco|male"),
        ("Spain + Hispanic America, woman", "name|Spain + Hispanic America|female"),
        ("Spain + Hispanic America, man", "name|Spain + Hispanic America|male"),
        ("neutral ID header", "neutral"),
        ("pronouns swapped", "swap"),
    ):
        (x,) = [d[0] for d in dots if d[1] == labels[label]]
        assert x == pytest.approx(100 * ARMS[arm]), label


def test_the_axis_shows_the_whole_noise_band_and_every_interval_even_when_the_band_is_wider(
    f8, report
):
    report["noise_floor"]["rerun"]["tpr_diff"] = _stat(0.0, 0.08)
    fig = f8.build(report)
    fig.canvas.draw()
    low, high = _ax(fig).get_xlim()
    assert low <= -8.0 and high >= 8.0
    plt.close(fig)


def test_the_noise_floor_label_is_centred_over_the_band(figure):
    (label,) = [t for t in _ax(figure).texts if t.get_text() == "noise floor"]
    assert label.get_position()[0] == pytest.approx(-0.4)  # the band runs from -1.0 to 0.2


def test_main_reads_the_report_and_saves_the_figure_as_pdf_and_png(f8, report, tmp_path):
    (tmp_path / "llm_audit.json").write_text(json.dumps(report))
    fig = f8.main(tmp_path)
    plt.close(fig)
    assert (tmp_path / "figures" / "f8_llm_screener.pdf").is_file()
    assert (tmp_path / "figures" / "f8_llm_screener.png").is_file()
