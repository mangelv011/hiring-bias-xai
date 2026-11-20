import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f3.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def f3():
    return _load_script()


def _summary(gap):
    return {"score_gap": {"value": gap}}


def test_each_panel_is_the_showcase_occupation_of_its_kind_with_the_largest_absolute_score_gap(f3):
    showcase = {
        "male_dominated": [{"occupation": o} for o in ("surgeon", "composer", "engineer")],
        "female_dominated": [{"occupation": o} for o in ("nurse", "model", "dietitian")],
    }
    summaries = {  # the main model's; a negative gap counts by its size
        "surgeon": _summary(-0.8),
        "composer": _summary(0.5),
        "engineer": _summary(-0.2),
        "nurse": _summary(1.3),
        "model": _summary(-5.4),
        "dietitian": _summary(2.3),
    }
    assert f3.pick(summaries, showcase) == {"male_dominated": "surgeon", "female_dominated": "model"}
