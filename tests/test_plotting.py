import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
