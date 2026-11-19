import importlib.util
import json
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
    offsets = np.concatenate([c.get_offsets() for c in ax.collections if len(c.get_offsets())])
    expected = sorted((SHARES[o], GAPS[o]) for o in SHARES)
    assert sorted(map(tuple, offsets)) == pytest.approx(expected)


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
    assert found == pytest.approx(expected)
