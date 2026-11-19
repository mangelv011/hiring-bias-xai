import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from hiring_bias_xai import plotting


@pytest.fixture
def fig():
    figure, ax = plt.subplots(figsize=(3.3, 2))
    ax.plot([0, 1], [0, 1])
    yield figure
    plt.close(figure)


def test_save_writes_a_pdf_and_a_png_named_after_the_figure(fig, tmp_path):
    plotting.save(fig, "demo", tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["demo.pdf", "demo.png"]
    assert (tmp_path / "demo.pdf").read_bytes().startswith(b"%PDF")


def test_the_png_is_300_dpi(fig, tmp_path):
    plotting.save(fig, "demo", tmp_path)
    with Image.open(tmp_path / "demo.png") as png:
        assert png.size == (990, 600)  # 3.3 x 2 inches at 300 dpi
        assert png.info["dpi"] == pytest.approx((300, 300), abs=0.1)


def test_semantic_colours_are_the_okabe_ito_values():
    assert plotting.WOMEN == "#CC79A7"  # reddish purple
    assert plotting.MEN == "#0072B2"  # blue
    assert plotting.HIGHLIGHT == "#D55E00"  # vermillion
    assert plotting.NEUTRAL == "#999999"  # grey


def test_new_figure_is_single_column_wide_with_readable_text_and_open_spines():
    fig, ax = plotting.new_figure(height=2.5)
    try:
        ax.set_title("title")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.text(0.5, 0.5, "note")
        fig.canvas.draw()
        assert tuple(fig.get_size_inches()) == (3.3, 2.5)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert not ax.xaxis.get_gridlines()[0].get_visible()
        sizes = [text.get_fontsize() for text in fig.findobj(matplotlib.text.Text)]
        sizes += [tick.get_fontsize() for tick in ax.get_xticklabels()]
        assert min(sizes) >= 7
    finally:
        plt.close(fig)


def test_deuteranopia_simulation_keeps_greys_and_maps_pure_red_by_the_machado_matrix():
    # Machado et al. (2009), severity 1.0, on linear RGB: red (1, 0, 0) becomes the first matrix
    # column (0.367322, 0.280085, -0.01182), whose blue part is clipped to 0; sRGB-encoded that
    # is (163, 144, 0). The rows of the matrix sum to 1, so greys do not move.
    image = np.array([[[255, 0, 0, 200], [128, 128, 128, 255], [0, 0, 0, 255]]], dtype=np.uint8)
    out = plotting.simulate_deuteranopia(image)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert out[0, 0].tolist() == [163, 144, 0, 200]
    assert out[0, 1].tolist() == [128, 128, 128, 255]
    assert out[0, 2].tolist() == [0, 0, 0, 255]


def test_deuteranopia_simulation_maps_pure_green_and_blue_by_the_other_matrix_columns():
    # Green (0, 1, 0) becomes the second column (0.860646, 0.672501, 0.04294) and blue the third
    # (-0.227968 clipped to 0, 0.047413, 0.968881); sRGB-encoded these are the values below.
    image = np.array([[[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    out = plotting.simulate_deuteranopia(image)
    assert out.tolist() == [[[239, 214, 58], [0, 61, 251]]]


def test_deuteranopia_simulation_keeps_dark_greys_in_both_sRGB_segments():
    greys = np.array([[[10, 10, 10], [30, 30, 30]]], dtype=np.uint8)
    assert plotting.simulate_deuteranopia(greys).tolist() == greys.tolist()


def test_deuteranopia_simulation_leaves_its_input_unchanged():
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)
    plotting.simulate_deuteranopia(image)
    assert image.tolist() == [[[255, 0, 0]]]


def test_the_pdf_embeds_text_as_truetype_not_type_3(tmp_path):
    fig, ax = plotting.new_figure(height=2)
    try:
        ax.set_title("title")
        plotting.save(fig, "demo", tmp_path)
    finally:
        plt.close(fig)
    assert b"/Type3" not in (tmp_path / "demo.pdf").read_bytes()


def test_axis_labels_stay_inside_the_figure():
    fig, ax = plotting.new_figure(height=2)
    try:
        ax.set_xlabel("Share of women in the training subsample")
        ax.set_ylabel("TPR gap (women minus men)")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for label in (ax.xaxis.label, ax.yaxis.label):
            box = label.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0
            assert box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
    finally:
        plt.close(fig)


def test_the_pdf_carries_no_creation_date_so_reruns_are_byte_identical(fig, tmp_path):
    plotting.save(fig, "demo", tmp_path)
    assert b"/CreationDate" not in (tmp_path / "demo.pdf").read_bytes()


def test_deuteranopia_simulation_of_colours_that_saturate_or_sit_in_the_dark_segment():
    # Yellow saturates the red channel before clipping (1.228); (40, 20, 90) is dark enough to use
    # the linear sRGB segment; the third is an ordinary orange. Expected values were worked out
    # with the Machado matrix in plain Python.
    image = np.array([[[255, 255, 0], [40, 20, 90], [200, 100, 50]]], dtype=np.uint8)
    assert plotting.simulate_deuteranopia(image).tolist() == [
        [[255, 250, 49], [0, 33, 89], [152, 137, 48]]
    ]


def test_no_font_size_in_the_style_is_below_7_pt():
    sizes = {key: value for key, value in plotting.STYLE.items() if "size" in key}
    assert sizes
    assert all(value >= 7 for value in sizes.values())
