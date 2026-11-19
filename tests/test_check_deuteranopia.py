import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_deuteranopia.py"


def test_the_script_writes_the_deuteranope_view_of_a_png(tmp_path):
    source = tmp_path / "in.png"
    Image.fromarray(np.array([[[255, 0, 0, 255], [128, 128, 128, 255]]], dtype=np.uint8)).save(
        source
    )
    spec = importlib.util.spec_from_file_location("check_deuteranopia", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "nested" / "out.png"
    module.main(source, target)
    with Image.open(target) as seen:
        assert np.array(seen).tolist() == [[[163, 144, 0, 255], [128, 128, 128, 255]]]


def test_the_script_overwrites_an_existing_target(tmp_path):
    source = tmp_path / "in.png"
    Image.fromarray(np.array([[[255, 0, 0, 255]]], dtype=np.uint8)).save(source)
    spec = importlib.util.spec_from_file_location("check_deuteranopia", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(source, tmp_path / "out.png")
    module.main(source, tmp_path / "out.png")
    with Image.open(tmp_path / "out.png") as seen:
        assert np.array(seen).tolist() == [[[163, 144, 0, 255]]]
