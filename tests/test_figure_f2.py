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


# 32 words in the error bio; the 8 largest |value| are marked below. Two of them are gendered
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


@pytest.fixture(scope="module")
def f2():
    return _load_script()


EXCERPT = 25  # words of the bio that the figure prints; the real figure prints 40


@pytest.fixture
def case(f2, monkeypatch):
    monkeypatch.setattr(f2, "MAX_WORDS", EXCERPT)
    return f2.Case("test-1", ERROR_WORDS, "surgeon", "nurse", "female", ERROR_VALUES)


@pytest.fixture
def figure(f2, case):
    fig = f2.build(case)
    fig.set_dpi(plotting.DPI)  # the resolution the figure is saved at
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _bar_axes(fig):
    return [ax for ax in fig.axes if ax.patches]


def _text_axes(fig):
    return [ax for ax in fig.axes if not ax.patches]


def test_the_figure_is_3_03_inches_wide_and_no_text_is_below_7_pt(figure):
    assert figure.get_size_inches()[0] == plotting.WIDTH_INCHES == 3.03
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


def test_the_case_has_horizontal_bars_for_its_eight_largest_values_sorted_by_size(figure):
    (error_ax,) = _bar_axes(figure)
    expected = ["nurse", "She", "surgeons", "her", "patients", "care;", "hospital", "trained"]
    labels = [t.get_text() for t in error_ax.get_yticklabels()]
    assert labels == expected  # top to bottom
    widths = [p.get_width() for p in error_ax.patches]
    assert widths == pytest.approx([3.0, 2.0, -1.5, 1.2, 0.9, -0.7, 0.6, -0.4])
    assert [p.get_y() + p.get_height() / 2 for p in error_ax.patches] == pytest.approx(range(8))
    assert error_ax.yaxis_inverted()  # the largest value is the first bar, at the top


def test_gendered_words_are_in_the_highlight_colour_and_the_others_in_grey(figure):
    (error_ax,) = _bar_axes(figure)
    labels = [t.get_text() for t in error_ax.get_yticklabels()]
    colours = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in error_ax.patches]
    gendered = {"She", "her"}
    for label, colour in zip(labels, colours, strict=True):
        assert colour == (plotting.HIGHLIGHT if label in gendered else plotting.NEUTRAL)
    for tick in error_ax.get_yticklabels():
        expected = plotting.HIGHLIGHT if tick.get_text() in gendered else "#333333"
        assert matplotlib.colors.to_hex(tick.get_color()).upper() == expected


def test_every_bar_has_its_value_written_at_its_end(figure):
    (error_ax,) = _bar_axes(figure)
    assert [t.get_text() for t in error_ax.texts] == [
        "+3.0", "+2.0", "-1.5", "+1.2", "+0.9", "-0.7", "+0.6", "-0.4"
    ]  # fmt: skip


def test_there_is_a_zero_line_and_an_axis_label_naming_the_predicted_occupation(figure):
    (ax,) = _bar_axes(figure)
    assert any(np.all(np.asarray(line.get_xdata()) == 0) for line in ax.get_lines())
    assert "nurse" in ax.get_xlabel()
    assert "log-odds" not in ax.get_xlabel()
    assert "score points" in ax.get_xlabel()


def test_the_bio_is_written_in_monospace_and_cut_after_the_excerpt_with_an_ellipsis(f2, case):
    fig = f2.build(case)
    fig.canvas.draw()
    (ax,) = _text_axes(fig)
    assert "".join(t.get_text() for t in ax.texts) == "".join(ERROR_WORDS[:EXCERPT]) + "\u2026"
    assert {t.get_fontfamily()[0] for t in ax.texts} == {"monospace"}
    short = f2.build(_case(f2, ERROR_WORDS[:20], ERROR_VALUES[:20]))
    short.canvas.draw()
    (short_ax,) = _text_axes(short)
    assert "".join(t.get_text() for t in short_ax.texts) == "".join(ERROR_WORDS[:20])
    plt.close(fig)
    plt.close(short)


def _underlines(ax):
    return [line for line in ax.get_lines() if line.get_gid() == "underline"]


def test_the_words_of_the_bars_inside_the_excerpt_are_underlined_under_their_own_text(figure):
    (error_text,) = _text_axes(figure)
    underlined = _underlines(error_text)
    # She, surgeons, her, patients, care; and hospital lie in the first 25 words; trained and
    # nurse come later. No underlined word is a chunk of the long address.
    assert len(underlined) == 6
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
    (error_text,) = _text_axes(figure)
    colours = sorted(
        matplotlib.colors.to_hex(line.get_color()).upper() for line in _underlines(error_text)
    )
    assert colours == sorted([plotting.HIGHLIGHT] * 2 + ["#333333"] * 4)


def _case(f2, words, values, gender="female", bio_id="test-9", true="a", predicted="b"):
    return f2.Case(bio_id, words, true, predicted, gender, np.array(values, dtype=float))


def test_a_word_longer_than_a_line_is_split_over_lines_and_each_piece_underlined_if_it_is_top(f2):
    fig = f2.build(_case(f2, ["x" * 120, "she"], [5.0, 1.0]))
    fig.canvas.draw()
    (error_text,) = _text_axes(fig)
    assert "".join(t.get_text() for t in error_text.texts) == "x" * 120 + "she"
    pieces = [t for t in error_text.texts if "x" in t.get_text()]
    assert len(_underlines(error_text)) == len(pieces) + 1
    plt.close(fig)


def test_a_unit_that_only_contains_a_gendered_word_is_not_highlighted(f2):
    fig = f2.build(_case(f2, ["shoot...she", "She,", "sheriff"], [5.0, 1.0, 0.5]))
    fig.canvas.draw()
    (ax,) = _bar_axes(fig)
    colours = [matplotlib.colors.to_hex(p.get_facecolor()).upper() for p in ax.patches]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["shoot...she", "She,", "sheriff"]
    assert colours == [plotting.NEUTRAL, plotting.HIGHLIGHT, plotting.NEUTRAL]
    (text_ax,) = _text_axes(fig)
    underline = [matplotlib.colors.to_hex(u.get_color()).upper() for u in _underlines(text_ax)]
    assert underline == ["#333333", plotting.HIGHLIGHT, "#333333"]
    plt.close(fig)


def _title_lines(fig):
    fig.canvas.draw()
    (title,) = [t for t in fig.texts if t.get_fontweight() == "bold"]
    return title.get_text(), title._get_wrapped_text().count("\n") + 1


def test_the_title_is_one_sentence_of_at_most_two_lines_with_the_push(figure):
    # Gendered sum of the case: She 2.0 + her 1.2 = 3.2.
    text, lines = _title_lines(figure)
    assert text == "Gender words added 3.2 score points to a wrong job for a woman."
    assert lines <= 2


def test_the_title_and_panel_say_when_there_is_no_case(f2):
    none, lines = _title_lines(f2.build(None))
    assert none == "No misclassified woman met the rules, so there is no case to show."
    assert lines <= 2
    empty = f2.build(None)
    empty.canvas.draw()
    assert _bar_axes(empty) == []
    texts = [t.get_text() for t in empty.texts]
    assert "Woman in a male-dominated occupation: no bio met the rules." in texts
    plt.close(empty)


def test_no_text_of_the_figure_says_log_odds(figure):
    figure.canvas.draw()
    assert not [
        t.get_text() for t in figure.findobj(matplotlib.text.Text) if "log-odds" in t.get_text()
    ]


def test_the_figure_says_that_per_word_values_carry_sampling_noise(figure):
    assert any("sampling noise" in t.get_text() for t in figure.texts)


def test_the_case_is_named_above_its_sentence_by_gender_true_and_predicted_occupation(figure):
    headers = [t.get_text() for t in figure.texts]
    assert "Woman, true surgeon, predicted nurse (test-1), first 25 of 32 words." in headers


def test_a_short_note_gives_the_gender_words_total_and_how_extreme_the_case_is(f2, case):
    fig = f2.build(case, wrong=114)
    notes = [" ".join(t.get_text().split()) for t in fig.texts]
    # Gendered units of the bio: She, her and she (2.0 + 1.2 + 0.0).
    assert (
        "Gender words: +3.2 from 3 units, the largest of 114 wrong decisions. "
        "Per-word values carry sampling noise." in notes
    )
    assert not any("A score point is" in note for note in notes)
    plt.close(fig)


MALE_DOMINATED = ["surgeon", "composer", "software_engineer"]
FEMALE_DOMINATED = ["dietitian", "nurse", "model"]
TEXTS = {
    "female": [
        "She trains new surgeons every year and she writes her notes late at night in the ward",
        "As a woman in surgery she often explains her methods to students who ask about her career",
    ],
    "male": [
        "He teaches nursing students every spring and he reviews his notes before his shifts start",
        "His ward is busy and he supports the patients while his colleagues cover the night rounds",
    ],
}


@pytest.fixture
def run(f2, monkeypatch, tmp_path):
    """Return go(): main() on 8 synthetic bios and a fake classifier; gives the results dir.

    Four women are surgeons and four men are nurses. `go.subset` can be changed before the call.
    """
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    model = Hashed()
    genders = ["female"] * 4 + ["male"] * 4
    texts = [TEXTS[g][i % 2] for i, g in enumerate(genders)]
    ids = [f"test-{i}" for i in range(8)]
    occupations = ["surgeon"] * 4 + ["nurse"] * 4
    subset = pd.DataFrame({"id": ids, "text": texts, "occupation": occupations, "gender": genders})
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_bytes(
        ("id\n" + "".join(f"{i}\n" for i in ids)).encode()
    )
    rows = lambda names: [{"occupation": n, "share_women": 0.5} for n in names]  # noqa: E731
    results.write_json(
        results_dir / "showcase_occupations.json",
        {"male_dominated": rows(MALE_DOMINATED), "female_dominated": rows(FEMALE_DOMINATED)},
    )
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    for bio_id, text, occupation in zip(ids, texts, occupations, strict=True):
        (saved,) = attribution.shapley(model, [text], [occupation], 3, "delete", seed=1)
        attribution.save(attribution.path("deberta", bio_id), saved)
    results.save_predictions("deberta", subset, model.predict(texts))

    def go():
        go.figure = f2.main(results_dir)
        return results_dir

    go.subset, go.model, go.results_dir = subset, model, results_dir
    return go


def test_main_saves_the_figure_and_the_record_of_the_picked_bios(run):
    results_dir = run()
    names = sorted(p.name for p in (results_dir / "figures").iterdir())
    assert names == ["f2_shapley_cases.pdf", "f2_shapley_cases.png"]
    record = json.loads((results_dir / "f2_cases.json").read_text())
    assert record["n_bios"] == 8
    assert record["n_skipped"] == 0
    assert record["min_units"] == 15
    assert "male-dominated" in record["rule"] and "female-dominated" in record["rule"]
    assert "15 word units" in record["rule"]
    assert any("the largest of" in t.get_text() for t in run.figure.texts)
    assert len(_bar_axes(run.figure)) == 1
    assert "six or more digits" in record["rule"]


def test_a_label_longer_than_14_characters_is_cut_with_an_ellipsis(f2):
    fig = f2.build(_case(f2, ["x" * 30, "she"], [5.0, 1.0]))
    (error_ax,) = _bar_axes(fig)
    assert [t.get_text() for t in error_ax.get_yticklabels()] == ["x" * 13 + "\u2026", "she"]
    plt.close(fig)


def test_each_value_is_written_beyond_the_end_of_its_bar_on_the_side_it_points_to(figure):
    (error_ax,) = _bar_axes(figure)
    for bar, label in zip(error_ax.patches, error_ax.texts, strict=True):
        end = error_ax.transData.transform([(bar.get_x() + bar.get_width(), 0)])[0][0]
        extent = label.get_window_extent()
        if bar.get_width() >= 0:
            assert extent.x0 > end
        else:
            assert extent.x1 < end


def _expected_groups(run):
    """Per group: the misclassified bios, and the best (bio ID, push) among those that qualify.

    Worked out here from the stored attributions with a plain set of gendered words.
    """
    gendered = {"she", "her", "he", "his"}
    guesses = run.model.predict(list(run.subset["text"]))
    groups = {"female": ("surgeon", [], []), "male": ("nurse", [], [])}
    for bio, guess in zip(run.subset.itertuples(), guesses, strict=True):
        if bio.occupation == guess:
            continue
        saved = attribution.load(attribution.path("deberta", bio.id))
        values = saved.values_for(data.OCCUPATIONS.index(guess))
        mask = np.array([word.lower() in gendered for word in saved.words])
        _, wrong, qualified = groups[bio.gender]
        wrong.append(bio.id)
        push = values[mask].sum()
        if len(saved.words) >= 15 and push > 0:
            qualified.append((bio.id, push))
    return {
        gender: (len(wrong), max(qualified, key=lambda pair: pair[1]) if qualified else None)
        for gender, (_, wrong, qualified) in groups.items()
    }


def test_the_record_holds_the_pick_of_each_group_with_its_score_and_size(run):
    record = json.loads((run() / "f2_cases.json").read_text())
    expected = _expected_groups(run)
    for gender, key in (("female", "women_in_male_dominated"), ("male", "men_in_female_dominated")):
        n_wrong, best = expected[gender]
        assert record[key]["n_wrong"] == n_wrong
        assert best is not None, "the fixture must give both groups a qualifying bio"
        pick = record[key]["pick"]
        assert pick["bio_id"] == best[0]
        assert pick["gendered_push"] == pytest.approx(best[1])
        assert pick["predicted"] != pick["true"]
        assert pick["true"] == {"female": "surgeon", "male": "nurse"}[gender]
        text = run.subset.set_index("id").loc[pick["bio_id"], "text"]
        assert pick["words"] == len(text.split())


def test_a_group_without_a_qualifying_bio_is_recorded_as_empty_and_the_figure_is_still_saved(run):
    run.subset["gender"] = "female"  # no man in the female-dominated occupations
    results_dir = run()
    record = json.loads((results_dir / "f2_cases.json").read_text())
    assert record["men_in_female_dominated"] == {"n_wrong": 0, "pick": None}
    assert record["women_in_male_dominated"]["pick"] is not None
    assert (results_dir / "figures" / "f2_shapley_cases.png").is_file()


def test_a_bio_whose_stored_prediction_differs_from_the_saved_one_is_counted_as_skipped(run):
    guesses = run.model.predict(list(run.subset["text"]))
    changed = ["surgeon" if guesses[0] != "surgeon" else "nurse", *guesses[1:]]
    results.save_predictions("deberta", run.subset, changed)
    record = json.loads((run() / "f2_cases.json").read_text())
    assert record["n_skipped"] == 1
    assert record["n_bios"] == 8  # the skipped bio is still one of the subset


def test_every_value_label_stays_inside_its_axes_so_it_does_not_run_into_the_word_labels(figure):
    for ax in _bar_axes(figure):
        box = ax.get_window_extent()
        for label in ax.texts:
            extent = label.get_window_extent()
            assert box.x0 <= extent.x0 and extent.x1 <= box.x1, label.get_text()


def test_the_count_of_wrong_calls_of_a_group_leaves_out_bios_of_other_occupations(run):
    wrong_women = _expected_groups(run)["female"][0]
    guesses = run.model.predict(list(run.subset["text"]))
    first_wrong = next(i for i in range(4) if run.subset.loc[i, "occupation"] != guesses[i])
    run.subset.loc[first_wrong, "occupation"] = "dentist"  # not a male-dominated occupation
    record = json.loads((run() / "f2_cases.json").read_text())
    assert record["women_in_male_dominated"]["n_wrong"] == wrong_women - 1


def test_the_count_of_wrong_calls_of_a_group_leaves_out_the_right_calls(run):
    wrong_women = _expected_groups(run)["female"][0]
    guesses = run.model.predict(list(run.subset["text"]))
    first_wrong = next(i for i in range(4) if run.subset.loc[i, "occupation"] != guesses[i])
    # The guess becomes a male-dominated occupation and this woman's true one: a right call.
    run.subset.loc[first_wrong, "occupation"] = guesses[first_wrong]
    rows = [{"occupation": o, "share_women": 0.1} for o in [*MALE_DOMINATED, guesses[first_wrong]]]
    results.write_json(
        run.results_dir / "showcase_occupations.json",
        {"male_dominated": rows, "female_dominated": []},
    )
    record = json.loads((run() / "f2_cases.json").read_text())
    assert record["women_in_male_dominated"]["n_wrong"] == wrong_women - 1
