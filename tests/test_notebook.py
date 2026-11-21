import ast
import json
import re
from pathlib import Path

import pytest
from notebook_rules import hand_typed_numbers, prints, result_literals, result_reads, source

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "audit.ipynb"
RESULTS = ROOT / "results"

SECTIONS = [
    "Data",
    "Models and training",
    "The bias (F1)",
    "Local explanation: single decisions (F2)",
    "Global explanation: which words make the gap (F3)",
    "Checking the explanations (F4, F5)",
    "Inside the model (F6)",
    "Mitigation (F7)",
    "LLM screener (F8)",
    "Limits",
]
FIGURES = {
    "The bias (F1)": ["f1_gap_vs_share.png"],
    "Global explanation: which words make the gap (F3)": ["f3_gap_words.png"],
    "Local explanation: single decisions (F2)": ["f2_shapley_cases.png"],
    "Checking the explanations (F4, F5)": ["f4_flip_rates.png", "f5_explanation_checks.png"],
    "Inside the model (F6)": ["f6_probing_patching.png"],
    "Mitigation (F7)": ["f7_mitigation.png"],
    "LLM screener (F8)": ["f8_llm_screener.png"],
}
IMPORTS = {"json", "pathlib", "pandas", "IPython", "matplotlib"}


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sections(nb):
    """The cells of each `## ` section, by the title on its first line."""
    sections, title = {}, None
    for cell in nb["cells"]:
        text = source(cell)
        if cell["cell_type"] == "markdown" and text.startswith("## "):
            title = text[3:].splitlines()[0].strip()
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


def test_every_result_file_the_notebook_names_is_opened_by_a_reader(nb):
    assert set(result_literals(nb)) <= set(result_reads(nb))


def test_no_cell_writes_to_stderr(nb):
    streams = [
        o["name"]
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for o in c["outputs"]
        if o["output_type"] == "stream"
    ]
    assert "stderr" not in streams


def test_it_is_a_short_tour(nb):
    assert 20 <= len(nb["cells"]) <= 30


def test_each_section_names_the_scripts_that_made_its_results_and_they_exist(nb):
    for title, cells in _sections(nb).items():
        if title == "Limits":
            continue
        scripts = re.findall(r"scripts/\w+\.py", "\n".join(source(c) for c in cells))
        assert scripts, title
        assert [s for s in scripts if not (ROOT / s).is_file()] == [], title


def test_the_models_section_plots_the_training_log(nb):
    cells = _sections(nb)["Models and training"]
    assert "training_log.csv" in result_reads({"cells": cells})


def test_the_weight_scrambling_result_is_read_from_stability(nb):
    cells = _sections(nb)["Checking the explanations (F4, F5)"]
    assert any('"randomisation"' in source(c) for c in cells)


def test_the_limits_are_six_bullets(nb):
    limits = source(nb["cells"][-1])
    assert limits.startswith("## Limits")
    bullets = [line for line in limits.splitlines() if line.startswith("- ")]
    assert len(bullets) == 6


def _text(nb, title):
    return "\n".join(source(c) for c in _sections(nb)[title])


def test_the_f4_text_reads_the_direction_of_the_flips_from_the_results(nb):
    text = _text(nb, "Checking the explanations (F4, F5)")
    assert "toward_share" in text
    assert "no extra reaction" not in text


def test_the_pooled_name_contrast_is_called_a_lead_and_its_limit_is_stated(nb):
    text = _text(nb, "LLM screener (F8)") + source(nb["cells"][-1])
    assert "lead" in text
    assert "cancel" in text


def test_the_training_plot_has_a_title_and_labels_the_epochs(nb):
    text = _text(nb, "Models and training")
    assert "suptitle(" in text
    assert 'xlabel="epoch"' in text


def test_the_limits_say_the_task_is_a_proxy_and_attribution_was_not_tried_with_gradients(nb):
    limits = " ".join(source(nb["cells"][-1]).split())
    assert "proxy for screening" in limits
    assert "gradient" in limits


def test_the_first_cell_says_it_is_a_results_viewer_and_points_to_the_script_table(nb):
    intro = source(nb["cells"][0])
    assert "results viewer" in intro
    assert "each stage is one script" in " ".join(intro.split())
    assert "docs/results.md#reproduce-every-result" in intro
