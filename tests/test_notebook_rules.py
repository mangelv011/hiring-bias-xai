import pytest
from notebook_rules import hand_typed_numbers, loose_numbers, prints, result_reads


def _nb(*cells):
    return {"cells": [{"cell_type": kind, "source": source} for kind, source in cells]}


def md(source):
    return ("markdown", source)


def code(source):
    return ("code", source)


@pytest.mark.parametrize(
    "text",
    [
        "Figure F1 and F8 come first.",
        "See ADR 0003 for the design.",
        "Weight randomization (Adebayo et al., 2018) was not run.",
        "Hooker et al. (2019) needs retraining.",
        "The script `scripts/figure_f3.py` writes `results/f3_panels.json`.",
        "The macro-F1 averages over jobs.",
        "Macro-F1 heads the column.",
        "The main model is DeBERTa-v3-small.",
        "One reader labelled them, and often none of them agreed.",
        "A tenant and a tentative answer.",
        "No digits at all.",
    ],
)
def test_allowed_text_has_no_numbers(text):
    assert loose_numbers(text) == []


@pytest.mark.parametrize(
    ("text", "token"),
    [
        ("The model passes on 76 of 100 men.", "76"),
        ("A gap of 0.14.", "0.14."),
        ("Shapley uses 40 orders.", "40"),
        ("In 2018 the gap was wide.", "2018"),
        ("It happened in the 2000s.", "2000s."),
        ("Figure F9 shows it.", "F9"),
        ("Figure F10 shows it.", "F10"),
        ("See ADR 3.", "3."),
        ("The F1 of 0.7.", "0.7."),
        ("The micro-F1 of the model.", "micro-F1"),
        ("DeBERTa-v4 is another model.", "DeBERTa-v4"),
        ("I compare two screeners.", "two"),
        ("Three guesses to compare.", "Three"),
        ("About a dozen words.", "dozen"),
        ("Of every hundred women.", "hundred"),
        ("Fifty of them.", "Fifty"),
        ("The file notes 5.txt has it.", "5.txt"),
    ],
)
def test_a_typed_number_is_found(text, token):
    assert token in loose_numbers(text)


def test_numbers_written_in_words_are_listed_in_reading_order():
    assert loose_numbers("Two of 76 jobs, then three.") == ["Two", "76", "three"]


def test_a_number_beside_an_allowed_token_is_still_found():
    assert loose_numbers("F1 shows 76 of 100, see ADR 0003.") == ["76", "100,"]


def test_static_markdown_numbers_are_reported_with_their_cell():
    nb = _nb(md("# Title"), md("I found 73 jobs."))
    assert hand_typed_numbers(nb) == [(1, "73")]


def test_a_list_of_source_lines_is_joined():
    nb = _nb(md(["I found\n", "73 jobs."]))
    assert hand_typed_numbers(nb) == [(0, "73")]


def test_the_literal_text_of_a_markdown_f_string_is_checked():
    nb = _nb(code('Markdown(f"{share:.0f} of 100 men, 76 passed")'))
    assert hand_typed_numbers(nb) == [(0, "100"), (0, "76")]


def test_format_specs_and_names_in_f_strings_are_not_numbers():
    nb = _nb(code('Markdown(f"{share:.1f} and {gap:+.2f} and {count:,}")'))
    assert hand_typed_numbers(nb) == []


def test_a_typed_number_inside_an_f_string_expression_is_found():
    nb = _nb(code('Markdown(f"about {share + 5} per group")'))
    assert hand_typed_numbers(nb) == [(0, "5")]


def test_scaling_by_a_hundred_is_allowed_in_an_expression():
    nb = _nb(code('Markdown(f"{100 * share:.0f} per group, {1 - share:.2f} left")'))
    assert hand_typed_numbers(nb) == []


def test_a_prose_string_built_outside_markdown_is_checked():
    nb = _nb(code('sentence = "Most of the 54 bios"\nMarkdown(sentence)'))
    assert hand_typed_numbers(nb) == [(0, "54")]


def test_a_concatenated_markdown_string_is_checked():
    nb = _nb(code('Markdown("first " + f"{x} and 7 more")'))
    assert hand_typed_numbers(nb) == [(0, "7")]


def test_keys_and_paths_with_digits_are_not_prose():
    nb = _nb(code('data["v2"]["flip_rate_against_v1"]\nread_json("f3_panels.json")'))
    assert hand_typed_numbers(nb) == []


def test_numbers_in_ordinary_code_are_not_prose():
    nb = _nb(code("scale = 100\nrows = frame.head(3)\nvalue = 0.5 * rows"))
    assert hand_typed_numbers(nb) == []


def test_every_offence_across_cells_is_listed_in_order():
    nb = _nb(md("About 12 bios."), code('Markdown("and 3 more")'), md("Fine."))
    assert hand_typed_numbers(nb) == [(0, "12"), (1, "3")]


def test_result_reads_lists_the_literal_file_names_in_order():
    nb = _nb(
        code('a = read_json("one.json")\nb = read_csv("two.csv")'),
        md("not code"),
        code("show_figure('f1_gap_vs_share.png')"),
    )
    assert result_reads(nb) == ["one.json", "two.csv", "f1_gap_vs_share.png"]


def test_result_reads_ignores_other_calls():
    nb = _nb(code('open("x.json")\nprint("y.csv")\nread_json("one.json")'))
    assert result_reads(nb) == ["one.json"]


def test_result_reads_rejects_a_name_that_is_not_a_literal():
    nb = _nb(code("read_json(name)"))
    with pytest.raises(ValueError, match="literal"):
        result_reads(nb)


def test_result_reads_rejects_a_call_without_a_name():
    nb = _nb(code("read_csv()"))
    with pytest.raises(ValueError, match="literal"):
        result_reads(nb)


def test_prints_lists_the_cells_that_print():
    nb = _nb(code("x = 1"), code('print("hi")'), md("print"), code("y = 2\nprint(y)"))
    assert prints(nb) == [1, 3]
