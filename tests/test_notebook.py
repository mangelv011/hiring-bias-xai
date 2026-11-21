import ast
import json
import re
from pathlib import Path

import pytest
from notebook_rules import hand_typed_numbers, prints, result_reads, source

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "audit.ipynb"
RESULTS = ROOT / "results"

SECTIONS = [
    "What the screener does, and the data",
    "How good are the screeners?",
    "Is it biased?",
    "Why is it biased?",
    "Does changing only the gender change the decision?",
    "Can the explanations be trusted?",
    "Where does gender live inside the model?",
    "What fixes it, and at what cost?",
    "Does an LLM screener behave the same?",
]
FIGURES = {
    "Is it biased?": ["f1_gap_vs_share.png"],
    "Why is it biased?": ["f2_shapley_cases.png", "f3_gap_words.png"],
    "Does changing only the gender change the decision?": ["f4_flip_rates.png"],
    "Can the explanations be trusted?": ["f5_explanation_checks.png"],
    "Where does gender live inside the model?": ["f6_probing_patching.png"],
    "What fixes it, and at what cost?": ["f7_mitigation.png"],
    "Does an LLM screener behave the same?": ["f8_llm_screener.png"],
}
MOVES = ["**Why I look at it.**", "**What I found.**", "**What it means.**"]
IMPORTS = {"json", "pathlib", "pandas", "IPython"}


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sections(nb):
    """The cells of each `## ` section, by title."""
    sections, title = {}, None
    for cell in nb["cells"]:
        text = source(cell)
        if cell["cell_type"] == "markdown" and text.startswith("## "):
            title = text[3:].strip()
            sections[title] = []
        elif title:
            sections[title].append(cell)
    return sections


def _prose(nb):
    """Static Markdown text and every string literal of the code cells."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            yield source(cell)
        else:
            for node in ast.walk(ast.parse(source(cell))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    yield node.value


def test_the_sections_follow_the_questions_of_the_audit(nb):
    assert list(_sections(nb)) == SECTIONS


def test_no_number_in_prose_is_typed_by_hand(nb):
    assert hand_typed_numbers(nb) == []


def test_every_result_the_notebook_reads_exists(nb):
    names = result_reads(nb)
    assert names
    missing = [
        name
        for name in names
        if not (RESULTS / ("figures" if name.endswith(".png") else "") / name).is_file()
    ]
    assert missing == []


def test_each_section_shows_its_figures_in_order(nb):
    sections = _sections(nb)
    for title, cells in sections.items():
        shown = [name for name in result_reads({"cells": cells}) if name.endswith(".png")]
        assert shown == FIGURES.get(title, []), title


def test_each_section_says_how_it_was_run(nb):
    for title, cells in _sections(nb).items():
        runs = [c for c in cells if "**How this was run.**" in source(c)]
        assert len(runs) == 1, title
        assert "scripts/" in source(runs[0]), title


def test_each_section_gives_its_reason_findings_and_meaning_in_that_order(nb):
    for title, cells in _sections(nb).items():
        text = "\n".join(source(c) for c in cells)
        positions = [text.find(move) for move in MOVES]
        assert -1 not in positions, title
        assert positions == sorted(positions), title


def test_prose_goes_through_markdown_not_print(nb):
    assert prints(nb) == []


def test_the_notebook_only_reads_results(nb):
    modules = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(source(cell))):
            if isinstance(node, ast.Import):
                modules |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module.split(".")[0])
    assert modules <= IMPORTS


def test_the_prose_is_in_the_first_person_and_names_no_course(nb):
    text = "\n".join(_prose(nb))
    assert not re.search(r"\b(we|our|us)\b", text, re.IGNORECASE)
    assert "Latin America" not in text
    assert not re.search(r"\b(course|professor|assignment)\b", text, re.IGNORECASE)


def test_it_is_committed_executed_once_top_to_bottom_with_its_outputs(nb):
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))
    assert all(c["outputs"] for c in code[1:])  # the first cell only sets up
    errors = [o for c in code for o in c["outputs"] if o["output_type"] == "error"]
    assert errors == []
