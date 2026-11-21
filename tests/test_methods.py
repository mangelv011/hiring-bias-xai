import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "src/hiring_bias_xai/"

# Each technique the code uses: its heading in docs/methods.md, its figure, the scripts that run it,
# the ADRs that record it, and the code that shows it is there (file, text found in the file).
TECHNIQUES = {
    "Bootstrap intervals and the permutation null": {
        "figure": "F1",
        "scripts": ["gap_statistics.py"],
        "adrs": ["0002", "0007"],
        "code": [
            (SRC + "fairness.py", "def evaluate"),
            (SRC + "gap_statistics.py", "def rms_permutation_test"),
        ],
    },
    "Shapley attribution with permutation sampling": {
        "figure": "F2",
        "scripts": ["shapley_pilot.py", "shapley_run.py", "shapley_stability.py"],
        "adrs": ["0003"],
        "code": [(SRC + "attribution.py", "def shapley")],
    },
    "Gap decomposition": {
        "figure": "F3",
        "scripts": ["gap_decomposition.py"],
        "adrs": ["0004"],
        "code": [(SRC + "gap_decomposition.py", "def decompose")],
    },
    "Counterfactual pronoun swaps with the deletion control": {
        "figure": "F4",
        "scripts": ["counterfactual_gender.py", "counterfactual_hand_check.py"],
        "adrs": ["0005"],
        "code": [
            (SRC + "counterfactual.py", "def swap_gender"),
            (SRC + "counterfactual_effects.py", "def summarise"),
        ],
    },
    "Deletion curves and stability checks": {
        "figure": "F5",
        "scripts": ["deletion_curves.py", "stability_runs.py", "stability.py"],
        "adrs": ["0003"],
        "code": [
            (SRC + "explanation_checks.py", "def deletion_curve"),
            (SRC + "explanation_checks.py", "def rank_agreement"),
        ],
    },
    "Linear probing": {
        "figure": "F6",
        "scripts": ["probing_features.py", "probing.py"],
        "adrs": ["0006"],
        "code": [(SRC + "probing.py", "def probe")],
    },
    "Activation patching": {
        "figure": "F6",
        "scripts": ["patching.py"],
        "adrs": ["0006"],
        "code": [(SRC + "patching.py", "def patch_pairs")],
    },
    "Scrubbed retraining and proxy-word masking": {
        "figure": "F7",
        "scripts": ["train_scrubbed.py", "proxy_words.py", "mitigation.py"],
        "adrs": ["0007"],
        "code": [(SRC + "proxy_words.py", "def mask_words")],
    },
    "Black-box name audit with manipulation check and noise floor": {
        "figure": "F8",
        "scripts": [
            "llm_names.py",
            "llm_name_check.py",
            "llm_pilot.py",
            "llm_run.py",
            "llm_analyse.py",
        ],
        "adrs": ["0008"],
        "code": [
            (SRC + "llm_audit.py", "def merge_decision"),
            (SRC + "llm_metrics.py", "def noise_floor"),
            (SRC + "llm_plan.py", "def build"),
            (SRC + "llm_profiles.py", "def sample_profiles"),
        ],
    },
}

# Modules and scripts that are not a technique of the audit: data, models, plots, checks.
PLUMBING = {
    SRC + name
    for name in (
        "__init__.py config.py data.py evaluation.py first_names.py ine_names.py models.py "
        "plotting.py results.py training.py"
    ).split()
} | {
    "scripts/" + name
    for name in (
        "prepare_data.py train_baseline.py train_deberta.py fetch_first_names.py word_units.py "
        "check_deuteranopia.py check_docs.py"
    ).split()
}

LEADS = ["What it does.", "Why it fits.", "How to read it.", "Pitfalls.", "Script and ADR."]


def _methods():
    text = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    return {part.split("\n", 1)[0].strip(): part for part in text.split("\n## ")[1:]}


def test_every_technique_of_the_code_is_in_the_methods_page_under_its_heading():
    assert set(TECHNIQUES) <= set(_methods())


def test_the_methods_page_has_no_entry_the_mapping_does_not_know():
    assert set(_methods()) == set(TECHNIQUES)


def test_each_technique_is_found_in_the_code_it_is_mapped_to():
    for name, technique in TECHNIQUES.items():
        for file, marker in technique["code"]:
            assert marker in (ROOT / file).read_text(encoding="utf-8"), (name, file, marker)


def test_every_module_and_script_is_a_technique_or_plumbing():
    mapped = {file for t in TECHNIQUES.values() for file, _ in t["code"]}
    mapped |= {"scripts/" + s for t in TECHNIQUES.values() for s in t["scripts"]}
    mapped |= {f"scripts/figure_f{n}.py" for n in range(1, 9)}
    found = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    found |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").glob("*.py")}
    assert found - mapped - PLUMBING == set()


def test_each_entry_has_the_five_parts_in_order():
    for name, section in _methods().items():
        positions = [section.find(f"**{lead}**") for lead in LEADS]
        assert -1 not in positions, name
        assert positions == sorted(positions), name


def test_each_entry_names_its_figure_its_scripts_and_its_adrs():
    sections = _methods()
    for name, technique in TECHNIQUES.items():
        section = sections[name]
        assert re.search(rf"\b{technique['figure']}\b", section), name
        for script in technique["scripts"]:
            assert f"`scripts/{script}`" in section, (name, script)
        for adr in technique["adrs"]:
            assert f"decisions/{adr}-" in section, (name, adr)


def test_each_entry_figure_link_points_into_the_results_page():
    for name, technique in TECHNIQUES.items():
        section = _methods()[name]
        assert f"results.md#{technique['figure'].lower()}-" in section, name
