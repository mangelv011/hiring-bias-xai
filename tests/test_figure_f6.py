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
# restored share of the differing positions at layers 0, 1, 2 (mean, interval -/+ 0.03)
SHARE = [1.0, 0.7, 0.0]


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
    """A csv as `scripts/probing.py` writes it, with unflagged rows the figure must ignore."""
    rows = []
    for model, values in ACCURACY.items():
        for layer, accuracy in enumerate(values):
            rows += [_row(model, layer, subset, accuracy) for subset in ("all", "unflagged")]
    for name, accuracy in REFERENCES.items():
        rows += [_row(name, None, subset, accuracy) for subset in ("all", "unflagged")]
    return pd.DataFrame(rows)


@pytest.fixture
def patching():
    rows = []
    for layer, share in enumerate(SHARE):
        rows.append(
            {
                "layer": layer,
                "position_set": "differing",
                "mean": share,
                "mean_ci_low": share - 0.03,
                "mean_ci_high": share + 0.03,
            }
        )
        rows.append(
            {
                "layer": layer,
                "position_set": "all",
                "mean": 1.0,
                "mean_ci_low": 1.0,
                "mean_ci_high": 1.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def figure(f6, probing, patching):
    fig = f6.build(probing, patching)
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _panels(fig):
    top, bottom = [a for a in fig.axes if a.lines]
    return top, bottom


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


def test_the_title_counts_layers_above_the_occupation_reference_and_names_the_layer_effect_leaves(
    f6, probing, patching
):
    # The occupation-only interval is 0.50 to 0.54. Every probe interval starts above it: the
    # lowest is the main model at layer 2 (0.55). The differing positions have an interval above
    # one half at layer 1 (0.67 to 0.73) and below it at layer 2.
    assert f6.title(probing, patching) == (
        "Gender is readable from scrubbed bios in all 6 layers (57% to 66%, against 52% from the "
        "occupation alone), and the swap's effect leaves the edited words by layer 2."
    )


def test_the_title_says_how_many_layers_when_not_all_are_above_the_reference(f6, probing, patching):
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), ["ci_low"]] = 0.50
    assert f6.title(probing, patching).startswith(
        "Gender is readable from scrubbed bios in 5 of 6 layers (57% to 66%,"
    )


def test_the_title_names_no_layer_when_the_effect_stays_above_one_half(f6, probing, patching):
    patching.loc[patching["position_set"] == "differing", ["mean", "mean_ci_high"]] = (0.9, 0.95)
    assert f6.title(probing, patching).endswith(
        "and the swap's effect stays in the edited words in every layer."
    )


def test_the_title_ignores_the_unflagged_rows(f6, probing, patching):
    probing.loc[probing["subset"] == "unflagged", "ci_low"] = 0.0
    assert "in all 6 layers" in f6.title(probing, patching)


def test_panel_a_draws_the_two_models_per_layer_with_their_intervals_in_grey_and_highlight(figure):
    top, _ = _panels(figure)
    for name, colour in (("pretrained", plotting.NEUTRAL), ("deberta", plotting.HIGHLIGHT)):
        line = _line(top, name)
        assert list(line.get_xdata()) == [0, 1, 2]
        assert list(line.get_ydata()) == pytest.approx(ACCURACY[name])
        assert matplotlib.colors.to_hex(line.get_color()).upper() == colour
        vertices = _band(top, colour).get_paths()[0].vertices
        assert vertices[:, 1].min() == pytest.approx(min(ACCURACY[name]) - 0.02)
        assert vertices[:, 1].max() == pytest.approx(max(ACCURACY[name]) + 0.02)


def test_panel_a_draws_the_three_references_as_horizontal_lines_across_the_layers(figure):
    top, _ = _panels(figure)
    for name, accuracy in REFERENCES.items():
        line = _line(top, name)
        assert list(line.get_ydata()) == pytest.approx([accuracy, accuracy])
        assert list(line.get_xdata()) == [0, 2]


def test_panel_a_labels_every_line_directly_and_has_no_legend(figure):
    top, _ = _panels(figure)
    texts = {t.get_text() for t in top.texts}
    assert {"main model", "pretrained", "TF-IDF", "occupation only", "majority class"} <= texts
    assert top.get_legend() is None


def test_panel_a_labels_do_not_overlap(figure):
    top, _ = _panels(figure)
    boxes = sorted(
        (t.get_window_extent() for t in top.texts if t.get_text().strip()), key=lambda b: b.y0
    )
    for lower, upper in zip(boxes, boxes[1:], strict=False):
        assert upper.y0 >= lower.y1 - 1


def test_panel_b_draws_the_restored_share_at_the_differing_positions_in_the_highlight(figure):
    _, bottom = _panels(figure)
    line = _line(bottom, "differing")
    assert list(line.get_ydata()) == pytest.approx(SHARE)
    assert matplotlib.colors.to_hex(line.get_color()).upper() == plotting.HIGHLIGHT
    vertices = _band(bottom, plotting.HIGHLIGHT).get_paths()[0].vertices
    assert vertices[:, 1].min() == pytest.approx(-0.03)
    assert vertices[:, 1].max() == pytest.approx(1.03)


def test_panel_b_draws_all_positions_in_grey_and_the_reference_lines_at_zero_and_one(figure):
    _, bottom = _panels(figure)
    everywhere = _line(bottom, "all")
    assert list(everywhere.get_ydata()) == [1.0, 1.0, 1.0]
    assert matplotlib.colors.to_hex(everywhere.get_color()).upper() == plotting.NEUTRAL
    references = [line for line in bottom.lines if line.get_label() in ("zero", "one")]
    assert sorted(line.get_ydata()[0] for line in references) == [0, 1]


def test_panel_b_labels_the_lines_and_says_the_flat_line_holds_by_construction(figure):
    _, bottom = _panels(figure)
    texts = {t.get_text() for t in bottom.texts}
    assert "edited words only" in texts
    assert any(t.startswith("all positions") and "by construction" in t for t in texts)
    assert bottom.get_legend() is None


def test_both_panels_put_the_layers_on_the_x_axis_with_the_embeddings_first(figure):
    top, bottom = _panels(figure)
    for ax in (top, bottom):
        assert [t.get_text() for t in ax.get_xticklabels()] == ["emb.", "1", "2"]
        assert ax.get_xlabel() == "Layer output"
    assert top.get_ylabel() == "Probe accuracy"
    assert bottom.get_ylabel() == "Restored share of the effect"


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


def test_main_saves_the_figure_from_the_two_csv_files(f6, probing, patching, tmp_path):
    probing.to_csv(tmp_path / "probing.csv", index=False)
    patching.to_csv(tmp_path / "patching.csv", index=False)
    fig = f6.main(tmp_path)
    assert sorted(p.name for p in (tmp_path / "figures").iterdir()) == [
        "f6_probing_patching.pdf",
        "f6_probing_patching.png",
    ]
    assert fig._suptitle.get_text() == f6.title(probing, patching)
    plt.close(fig)


def test_the_effect_has_left_the_edited_words_only_when_the_whole_interval_is_below_one_half(
    f6, probing, patching
):
    differing = patching["position_set"] == "differing"
    patching.loc[differing & (patching["layer"] == 1), ["mean", "mean_ci_high"]] = (0.45, 0.55)
    assert f6.title(probing, patching).endswith("leaves the edited words by layer 2.")


def _text(ax, label):
    (text,) = [t for t in ax.texts if t.get_text() == label]
    return text


def test_an_interval_that_ends_exactly_at_one_half_has_not_left_the_edited_words_yet(
    f6, probing, patching
):
    patching.loc[
        (patching["position_set"] == "differing") & (patching["layer"] == 1), "mean_ci_high"
    ] = 0.5
    assert f6.title(probing, patching).endswith("leaves the edited words by layer 2.")


def test_the_title_names_the_first_layer_below_one_half_whatever_the_order_of_the_rows(
    f6, probing, patching
):
    differing = patching["position_set"] == "differing"
    patching.loc[differing & (patching["layer"] == 1), ["mean", "mean_ci_high"]] = (0.4, 0.45)
    assert f6.title(probing, patching.iloc[::-1]).endswith("leaves the edited words by layer 1.")


def _occupation_reference(probing, column):
    rows = probing[(probing["model"] == "occupation_only") & (probing["subset"] == "all")]
    return rows[column].iloc[0]


def test_a_probe_whose_interval_starts_exactly_at_the_reference_interval_end_does_not_count(
    f6, probing, patching
):
    ends = _occupation_reference(probing, "ci_high")
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), "ci_low"] = ends
    assert "in 5 of 6 layers" in f6.title(probing, patching)


def test_a_probe_counts_only_when_its_interval_starts_above_the_end_of_the_reference_interval(
    f6, probing, patching
):
    inside = (
        _occupation_reference(probing, "accuracy") + 0.01
    )  # above the accuracy, in the interval
    probing.loc[(probing["model"] == "deberta") & (probing["layer"] == 2), "ci_low"] = inside
    assert "in 5 of 6 layers" in f6.title(probing, patching)


def test_spread_moves_labels_up_to_the_gap_and_keeps_their_order(f6):
    assert f6._spread([0.5, 0.51, 0.52], 0.1) == pytest.approx([0.5, 0.6, 0.7])
    assert f6._spread([0.52, 0.5], 0.1) == pytest.approx([0.6, 0.5])
    assert f6._spread([0.1, 0.9], 0.1) == [0.1, 0.9]


def test_panel_a_labels_sit_at_the_last_layer_accuracy_moved_apart_only_where_they_touch(figure):
    top, _ = _panels(figure)
    assert _text(top, "pretrained").get_position()[1] == pytest.approx(0.66)
    assert _text(top, "main model").get_position()[1] == pytest.approx(0.57)  # TF-IDF is 0.58


def test_panel_a_axis_spans_the_intervals_with_a_margin_of_two_points(figure):
    top, _ = _panels(figure)
    assert top.get_ylim() == pytest.approx((0.46 - 0.02, 0.68 + 0.02))
    labels = [t.get_text() for t in top.get_yticklabels() if t.get_text()]
    assert labels and all(re.fullmatch(r"\d+%", label) for label in labels)


def test_panel_b_ticks_zero_half_and_one_and_labels_its_lines_around_their_last_value(figure):
    _, bottom = _panels(figure)
    assert list(bottom.get_yticks()) == [0, 0.5, 1]
    assert _text(bottom, "edited words only").get_position()[1] == pytest.approx(0.0 + 0.12)
    label = next(t for t in bottom.texts if t.get_text().startswith("all positions"))
    assert label.get_position()[1] == pytest.approx(1.0 - 0.1)


def test_the_bands_are_translucent_and_the_x_axis_line_spans_the_layers(figure):
    top, bottom = _panels(figure)
    for ax, colour in ((top, plotting.NEUTRAL), (bottom, plotting.HIGHLIGHT)):
        assert _band(ax, colour).get_facecolor()[0][3] == pytest.approx(0.25)
        assert ax.spines["bottom"].get_bounds() == (0, 2)


def test_the_three_references_are_told_apart_by_line_style(figure):
    top, _ = _panels(figure)
    styles = [_line(top, name).get_linestyle() for name in REFERENCES]
    assert len(set(styles)) == 3


def test_the_title_is_left_aligned_and_the_top_panel_is_taller_than_the_bottom_one(figure):
    top, bottom = _panels(figure)
    assert figure._suptitle.get_position()[0] == 0
    assert top.get_position().height > bottom.get_position().height
    assert figure.get_size_inches()[1] == 5.6


def test_the_labels_of_panel_a_fit_inside_the_axes_to_the_right_of_the_last_layer(figure):
    top, _ = _panels(figure)
    box = top.get_window_extent()
    for label in ("main model", "pretrained", "TF-IDF", "occupation only", "majority class"):
        assert _text(top, label).get_window_extent().x1 <= box.x1 + 1, label
