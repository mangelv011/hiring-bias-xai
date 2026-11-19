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
