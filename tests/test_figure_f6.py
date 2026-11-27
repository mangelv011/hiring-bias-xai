import importlib.util
import re
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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f6.py"


@pytest.fixture(scope="module")
def f6():
    spec = importlib.util.spec_from_file_location("figure_f6", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# accuracy of the probe at layers 0, 1, 2; the interval is accuracy -/+ 0.02
ACCURACY = {"pretrained": [0.60, 0.64, 0.66], "deberta": [0.60, 0.59, 0.57]}
REFERENCES = {"majority": 0.48, "occupation_only": 0.52, "tfidf": 0.58}
SHOWN_REFERENCES = ("tfidf", "occupation_only")  # the figure leaves the majority class out


def _row(model, layer, subset, accuracy):
    shift = 0.0 if subset == "all" else 0.3  # unflagged rows differ, so a mix-up shows
    return {
        "model": model,
        "layer": layer,
        "subset": subset,
        "accuracy": accuracy + shift,
        "ci_low": accuracy + shift - 0.02,
        "ci_high": accuracy + shift + 0.02,
    }


@pytest.fixture
def probing():
    """A csv as `scripts/probing.py` writes it, with rows the figure must ignore: the unflagged
    subset, the pretrained encoder and the majority class."""
    rows = []
    for model, values in ACCURACY.items():
        for layer, accuracy in enumerate(values):
            rows += [_row(model, layer, subset, accuracy) for subset in ("all", "unflagged")]
    for name, accuracy in REFERENCES.items():
        rows += [_row(name, None, subset, accuracy) for subset in ("all", "unflagged")]
    return pd.DataFrame(rows)


@pytest.fixture
def figure(f6, probing):
    fig = f6.build(probing)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _panel(fig):
    (ax,) = fig.axes
    return ax


def _line(ax, label):
    (line,) = [line for line in ax.lines if line.get_label() == label]
    return line


def _band(ax, colour):
    (band,) = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PolyCollection)
        and matplotlib.colors.to_hex(c.get_facecolor()[0]).upper() == colour
    ]
    return band


def test_the_title_claims_only_that_gender_is_still_readable_without_pronouns(f6, probing):
    # The occupation-only interval is 0.50 to 0.54. Every probe interval of the main model starts
    # above it: the lowest is layer 2 (0.55).
    assert f6.title(probing) == "Gender is still readable from bios without pronouns."


def test_the_title_says_in_only_some_layers_when_not_all_are_above_the_reference(f6, probing):
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), ["ci_low"]] = 0.50
    assert f6.title(probing) == (
        "Gender is readable from bios without pronouns in only some layers."
    )


def test_the_title_ignores_the_unflagged_rows(f6, probing):
    probing.loc[probing["subset"] == "unflagged", "ci_low"] = 0.0
    assert f6.title(probing) == "Gender is still readable from bios without pronouns."


def test_the_title_ignores_the_pretrained_encoder_which_the_figure_no_longer_shows(f6, probing):
    probing.loc[probing["model"] == "pretrained", "ci_low"] = 0.0
    assert f6.title(probing) == "Gender is still readable from bios without pronouns."


def test_the_main_model_is_drawn_per_layer_with_its_interval_in_the_highlight(figure):
    ax = _panel(figure)
    line = _line(ax, "deberta")
    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == pytest.approx(ACCURACY["deberta"])
    assert matplotlib.colors.to_hex(line.get_color()).upper() == plotting.HIGHLIGHT
    vertices = _band(ax, plotting.HIGHLIGHT).get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(min(ACCURACY["deberta"]) - 0.02)
    assert vertices[:, 1].max() == pytest.approx(max(ACCURACY["deberta"]) + 0.02)


def test_exactly_three_lines_are_drawn_without_the_pretrained_encoder_or_the_majority_class(figure):
    ax = _panel(figure)
    assert sorted(line.get_label() for line in ax.lines) == ["deberta", "occupation_only", "tfidf"]


def test_the_two_references_are_horizontal_lines_across_the_layers(figure):
    ax = _panel(figure)
    for name in SHOWN_REFERENCES:
        line = _line(ax, name)
        assert list(line.get_ydata()) == pytest.approx([REFERENCES[name]] * 2)
        assert list(line.get_xdata()) == [0, 2]


def test_every_line_is_labelled_directly_and_there_is_no_legend(figure):
    ax = _panel(figure)
    texts = {t.get_text() for t in ax.texts}
    assert texts == {"main model", "TF-IDF", "occupation only"}
    assert ax.get_legend() is None


def test_the_labels_do_not_overlap(figure):
    ax = _panel(figure)
    boxes = sorted(
        (t.get_window_extent() for t in ax.texts if t.get_text().strip()), key=lambda b: b.y0
    )
    for lower, upper in zip(boxes, boxes[1:], strict=False):
        assert upper.y0 >= lower.y1 - 1


def test_the_layers_are_on_the_x_axis_with_the_embeddings_first(figure):
    ax = _panel(figure)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["emb.", "1", "2"]
    assert ax.get_xlabel() == "Layer output"
    assert ax.get_ylabel() == "Probe accuracy"


def test_the_figure_is_3_03_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == plotting.WIDTH_INCHES == 3.03
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 7


def test_the_figure_has_one_panel_and_no_patching_text(figure):
    assert len(figure.axes) == 1
    texts = " ".join(t.get_text() for t in figure.findobj(matplotlib.text.Text)).lower()
    assert "patch" not in texts and "restored" not in texts


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    for t in figure.findobj(matplotlib.text.Text):
        if t.get_text().strip() and t.get_visible():
            extent = t.get_window_extent()
            assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
            assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def test_main_saves_the_figure_from_probing_csv_alone(f6, probing, tmp_path):
    probing.to_csv(tmp_path / "probing.csv", index=False)
    fig = f6.main(tmp_path)
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "f6_probing.pdf",
        "f6_probing.png",
    ]
    assert fig._suptitle.get_text() == f6.title(probing)
    plt.close(fig)


def _text(ax, label):
    (text,) = [t for t in ax.texts if t.get_text() == label]
    return text


def _occupation_reference(probing, column):
    rows = probing[(probing["model"] == "occupation_only") & (probing["subset"] == "all")]
    return rows[column].iloc[0]


def test_a_probe_whose_interval_starts_exactly_at_the_reference_interval_end_does_not_count(
    f6, probing
):
    ends = _occupation_reference(probing, "ci_high")
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), "ci_low"] = ends
    assert "in only some layers" in f6.title(probing)


def test_a_probe_counts_only_when_its_interval_starts_above_the_end_of_the_reference_interval(
    f6, probing
):
    inside = (
        _occupation_reference(probing, "accuracy") + 0.01
    )  # above the accuracy, in the interval
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), "ci_low"] = inside
    assert "in only some layers" in f6.title(probing)


def test_the_label_of_the_main_model_sits_at_its_last_layer_accuracy(figure):
    ax = _panel(figure)
    # TF-IDF is 0.58, so the labels are moved apart by a hair.
    assert _text(ax, "main model").get_position()[1] == pytest.approx(0.57, abs=0.01)
    assert _text(ax, "occupation only").get_position()[1] == pytest.approx(0.52, abs=0.01)


def test_the_axis_spans_the_intervals_of_the_three_lines_with_a_margin_of_two_points(figure):
    ax = _panel(figure)
    assert ax.get_ylim() == pytest.approx((0.50 - 0.02, 0.62 + 0.02))
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert labels and all(re.fullmatch(r"[\d.]+%", label) for label in labels)


def test_the_band_is_translucent_and_the_x_axis_line_spans_the_layers(figure):
    ax = _panel(figure)
    assert _band(ax, plotting.HIGHLIGHT).get_facecolor()[0][3] == pytest.approx(0.25)
    assert ax.spines["bottom"].get_bounds() == (0, 2)


def test_the_tfidf_and_occupation_only_lines_have_their_interval_as_a_faint_band(figure):
    ax = _panel(figure)
    bands = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PolyCollection)
        and matplotlib.colors.to_hex(c.get_facecolor()[0]).upper() == "#333333"
    ]
    assert len(bands) == 2
    spans = sorted(
        (
            round(b.get_paths()[0].vertices[:, 1].min(), 2),
            round(b.get_paths()[0].vertices[:, 1].max(), 2),
        )
        for b in bands
    )
    expected = [REFERENCES["tfidf"], REFERENCES["occupation_only"]]
    assert spans == sorted((round(a - 0.02, 2), round(a + 0.02, 2)) for a in expected)
    assert all(b.get_facecolor()[0][3] < 0.2 for b in bands)


def test_the_two_references_are_told_apart_by_line_style(figure):
    ax = _panel(figure)
    styles = [_line(ax, name).get_linestyle() for name in SHOWN_REFERENCES]
    assert len(set(styles)) == 2


def test_the_title_is_left_aligned_and_the_figure_is_2_9_inches_high(figure):
    assert figure._suptitle.get_position()[0] == 0
    assert figure.get_size_inches()[1] == 2.9


def test_the_labels_fit_inside_the_axes_to_the_right_of_the_last_layer(figure):
    ax = _panel(figure)
    box = ax.get_window_extent()
    for label in ("main model", "TF-IDF", "occupation only"):
        assert _text(ax, label).get_window_extent().x1 <= box.x1 + 1, label


def test_the_title_says_some_layers_when_only_one_probe_is_above_the_reference(f6, probing):
    rows = (probing["model"] == "deberta") & (probing["subset"] == "all")
    probing.loc[rows, "ci_low"] = 0.0
    probing.loc[rows & (probing["layer"] == 0), "ci_low"] = 0.9
    assert "in only some layers" in f6.title(probing)


def test_the_title_says_not_readable_when_no_probe_is_above_the_reference(f6, probing):
    rows = (probing["model"] == "deberta") & (probing["subset"] == "all")
    probing.loc[rows, "ci_low"] = 0.0
    assert f6.title(probing) == (
        "Gender is not readable beyond the occupation from bios without pronouns."
    )
