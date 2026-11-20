import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data, plotting, results

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "figure_f2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("figure_f2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 36 words in the error bio; the 8 largest |value| are marked below. Two of them are gendered
# ("She", "her"), and one word is longer than a line of the sentence block.
ERROR_WORDS = (
    "She leads a team of surgeons and her patients praise the care; contact "
    "https://example.org/a-very-long-address-that-does-not-fit-in-one-line-of-text for more "
    "about work at the hospital in the city where she trained as a nurse for years"
).split()
ERROR_VALUES = np.zeros(len(ERROR_WORDS))
for word, value in {
    "She": 2.0, "her": 1.2, "surgeons": -1.5, "patients": 0.9, "care;": -0.7, "hospital": 0.6,
    "nurse": 3.0, "trained": -0.4,
}.items():  # fmt: skip
    ERROR_VALUES[ERROR_WORDS.index(word)] = value
CORRECT_WORDS = "He is a nurse and his ward is busy".split()
CORRECT_VALUES = np.array([0.5, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0])


@pytest.fixture(scope="module")
def f2():
    return _load_script()


@pytest.fixture
def cases(f2):
    error = f2.Case("test-1", ERROR_WORDS, "surgeon", "nurse", ERROR_VALUES)
    correct = f2.Case("test-2", CORRECT_WORDS, "nurse", "nurse", CORRECT_VALUES)
    return error, correct


@pytest.fixture
def figure(f2, cases):
    fig = f2.build(*cases)
    fig.set_dpi(plotting.DPI)  # the resolution the figure is saved at
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _bar_axes(fig):
    return [ax for ax in fig.axes if ax.patches]


def _text_axes(fig):
    return [ax for ax in fig.axes if not ax.patches]


def test_the_figure_is_3_3_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == 3.3
    texts = [t for t in figure.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert texts
    assert min(t.get_fontsize() for t in texts) >= 7


def _drawn_texts(fig):
    """Texts with content, without the tick labels of ticks outside the axis limits (not drawn)."""
    hidden = set()
    for ax in fig.axes:
        low, high = sorted(ax.get_xlim())
        for tick, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=False):
            if not low <= tick <= high:
                hidden.add(id(label))
    return [
        t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and id(t) not in hidden
    ]


def test_nothing_is_drawn_outside_the_figure(figure):
    box = figure.bbox
    for t in _drawn_texts(figure):
        extent = t.get_window_extent()
        assert extent.x0 >= box.x0 - 1 and extent.x1 <= box.x1 + 1, t.get_text()
        assert extent.y0 >= box.y0 - 1 and extent.y1 <= box.y1 + 1, t.get_text()


def test_each_case_has_horizontal_bars_for_its_eight_largest_values_sorted_by_size(figure):
    error_ax, correct_ax = _bar_axes(figure)
    expected = ["nurse", "She", "surgeons", "her", "patients", "care;", "hospital", "trained"]
    labels = [t.get_text() for t in error_ax.get_yticklabels()]
    assert labels == expected  # top to bottom
    widths = [p.get_width() for p in error_ax.patches]
    assert widths == pytest.approx([3.0, 2.0, -1.5, 1.2, 0.9, -0.7, 0.6, -0.4])
    assert [p.get_y() + p.get_height() / 2 for p in error_ax.patches] == pytest.approx(range(8))
    assert error_ax.yaxis_inverted()  # the largest value is the first bar, at the top
    assert len(correct_ax.patches) == 8
    # Ties (here the zeros and the two 0.5) keep the order of the words in the bio.
    assert [t.get_text() for t in correct_ax.get_yticklabels()] == [
        "nurse", "his", "He", "is", "is", "a", "and", "ward"
    ]  # fmt: skip


def test_gendered_words_are_in_the_highlight_colour_and_the_others_in_grey(figure):
    error_ax, _ = _bar_axes(figure)
    labels = [t.get_text() for t in error_ax.get_yticklabels()]
    colours = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in error_ax.patches]
    gendered = {"She", "her"}
    for label, colour in zip(labels, colours, strict=True):
        assert colour == (plotting.HIGHLIGHT if label in gendered else plotting.NEUTRAL)
    for tick in error_ax.get_yticklabels():
        expected = plotting.HIGHLIGHT if tick.get_text() in gendered else "#333333"
        assert matplotlib.colors.to_hex(tick.get_color()).upper() == expected


def test_every_bar_has_its_value_written_at_its_end(figure):
    error_ax, _ = _bar_axes(figure)
    assert [t.get_text() for t in error_ax.texts] == [
        "+3.0", "+2.0", "-1.5", "+1.2", "+0.9", "-0.7", "+0.6", "-0.4"
    ]  # fmt: skip


def test_there_is_a_zero_line_and_an_axis_label_naming_the_predicted_occupation(figure):
    error_ax, correct_ax = _bar_axes(figure)
    for ax in (error_ax, correct_ax):
        assert any(np.all(np.asarray(line.get_xdata()) == 0) for line in ax.get_lines())
        assert "nurse" in ax.get_xlabel()
        assert "log-odds" in ax.get_xlabel()


def test_the_whole_bio_is_written_above_its_bars_in_a_monospaced_font(figure):
    error_text, correct_text = _text_axes(figure)
    for ax, words in ((error_text, ERROR_WORDS), (correct_text, CORRECT_WORDS)):
        shown = [t.get_text() for t in ax.texts]
        assert "".join(shown) == "".join(words)
        assert {t.get_fontfamily()[0] for t in ax.texts} == {"monospace"}


def _underlines(ax):
    return [line for line in ax.get_lines() if line.get_gid() == "underline"]


def test_the_eight_words_of_the_bars_are_underlined_under_their_own_text(figure):
    error_text, _ = _text_axes(figure)
    underlined = _underlines(error_text)
    # The 8 words fit on one line each; no underlined word is a chunk of the long address.
    assert len(underlined) == 8
    pieces = {t.get_text(): t for t in error_text.texts}
    for line in underlined:
        (x0, x1), (y0, _) = line.get_xdata(), line.get_ydata()
        start, end = error_text.transData.transform([(x0, y0), (x1, y0)])
        match = [
            t
            for t in pieces.values()
            if abs(t.get_window_extent().x0 - start[0]) < 1.5
            and abs(t.get_window_extent().x1 - end[0]) < 1.5
            and start[1] < t.get_window_extent().y0 + 3
            and start[1] > t.get_window_extent().y0 - 6
        ]
        assert len(match) == 1


def test_underlines_of_gendered_words_are_highlighted_and_the_rest_dark(figure):
    error_text, _ = _text_axes(figure)
    colours = sorted(
        matplotlib.colors.to_hex(line.get_color()).upper() for line in _underlines(error_text)
    )
    assert colours == sorted([plotting.HIGHLIGHT] * 2 + ["#333333"] * 6)


def test_a_word_longer_than_a_line_is_split_over_lines_and_each_piece_underlined_if_it_is_top(f2):
    from_long = f2.Case("test-9", ["x" * 120, "she"], "a", "b", np.array([5.0, 1.0]))
    correct = f2.Case("test-8", ["a", "she"], "b", "b", np.array([1.0, 1.0]))
    fig = f2.build(from_long, correct)
    fig.canvas.draw()
    error_text, _ = _text_axes(fig)
    assert "".join(t.get_text() for t in error_text.texts) == "x" * 120 + "she"
    assert (
        len(_underlines(error_text))
        == len([t for t in error_text.texts if "x" in t.get_text()]) + 1
    )
    plt.close(fig)


def test_the_title_states_the_pushes_and_shares_it_measured(figure):
    title = " ".join(t.get_text() for t in figure.texts if "Gendered" in t.get_text())
    # Gendered sum in the error bio: 2.0 + 1.2 = 3.2. Share in the correct bio: (0.5 + 1.0) / 4.0.
    assert "+3.2" in title
    assert "nurse" in title
    assert "38%" in title


def test_the_figure_says_that_per_word_values_carry_sampling_noise(figure):
    assert any("sampling noise" in t.get_text() for t in figure.texts)


def test_the_error_and_the_correct_case_are_named_above_their_sentences(figure):
    headers = [t.get_text() for t in figure.texts]
    assert any(h.startswith("Error:") and "surgeon" in h and "nurse" in h for h in headers)
    assert any(h.startswith("Correct:") and "nurse" in h for h in headers)


def _subset(ids, texts, occupations):
    return pd.DataFrame({"id": ids, "text": texts, "occupation": occupations})


@pytest.fixture
def run(f2, monkeypatch, tmp_path):
    """Return go(): main() on 8 synthetic bios and a fake classifier; gives the results dir."""
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    model = Hashed()
    texts = [
        "She trains nurses and she writes her notes",
        "He works as a surgeon and his team operates",
        "She is a nurse here and her ward is busy",
        "He is a dentist and his clinic is open today",
    ] * 2
    ids = [f"test-{i}" for i in range(8)]
    guesses = model.predict(texts)
    # Bios 1, 3 and 5 are called right, the other five wrong.
    occupations = [
        g if i in (1, 3, 5) else ("surgeon" if g != "surgeon" else "nurse")
        for i, g in enumerate(guesses)
    ]
    subset = _subset(ids, texts, occupations)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_bytes(
        ("id\n" + "".join(f"{i}\n" for i in ids)).encode()
    )
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    for bio_id, text, occupation in zip(ids, texts, occupations, strict=True):
        (saved,) = attribution.shapley(model, [text], [occupation], 3, "delete", seed=1)
        attribution.save(attribution.path("deberta", bio_id), saved)
    results.save_predictions("deberta", subset, model.predict(texts))

    def go():
        f2.main(results_dir)
        return results_dir

    go.subset, go.model = subset, model
    return go


def test_main_saves_the_figure_and_the_record_of_the_two_picked_bios(run):
    results_dir = run()
    names = sorted(p.name for p in (results_dir / "figures").iterdir())
    assert names == ["f2_shapley_cases.pdf", "f2_shapley_cases.png"]
    record = json.loads((results_dir / "f2_cases.json").read_text())
    assert "gendered" in record["rule_error"]
    assert "share" in record["rule_correct"]
    assert record["n_bios"] == 8
    assert record["n_skipped"] == 0
    assert (record["n_errors"], record["n_correct"]) == (5, 3)
    for key in ("error", "correct"):
        assert record[key]["bio_id"].startswith("test-")
        assert record[key]["predicted"] in data.OCCUPATIONS
    assert record["error"]["true"] != record["error"]["predicted"]
    assert record["correct"]["true"] == record["correct"]["predicted"]


def test_a_label_longer_than_14_characters_is_cut_with_an_ellipsis(f2):
    words = ["x" * 30, "she"]
    error = f2.Case("test-1", words, "a", "b", np.array([5.0, 1.0]))
    correct = f2.Case("test-2", ["a", "she"], "b", "b", np.array([1.0, 1.0]))
    fig = f2.build(error, correct)
    (error_ax, _) = _bar_axes(fig)
    assert [t.get_text() for t in error_ax.get_yticklabels()] == ["x" * 13 + "\u2026", "she"]
    plt.close(fig)


def test_each_value_is_written_beyond_the_end_of_its_bar_on_the_side_it_points_to(figure):
    error_ax, _ = _bar_axes(figure)
    for bar, label in zip(error_ax.patches, error_ax.texts, strict=True):
        end = error_ax.transData.transform([(bar.get_x() + bar.get_width(), 0)])[0][0]
        extent = label.get_window_extent()
        if bar.get_width() >= 0:
            assert extent.x0 > end
        else:
            assert extent.x1 < end
