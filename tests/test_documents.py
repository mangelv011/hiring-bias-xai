import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITTEN = [
    "README.md",
    "GLOSSARY.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")),
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "decisions").glob("*.md")),
]
SHOWCASE = ["surgeon", "software engineer", "composer", "dietitian", "nurse", "model"]


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _sections(text, level="## "):
    return {
        part.split("\n", 1)[0].removeprefix(level).strip(): part
        for part in re.split(rf"(?m)^(?={re.escape(level)})", text)
        if part.startswith(level)
    }


def test_the_model_card_has_the_sections_of_mitchell_et_al_in_order():
    headings = list(_sections(_read("docs/model_card.md")))
    expected = [
        "Model details",
        "Intended use",
        "Out-of-scope uses",
        "Factors",
        "Metrics",
        "Evaluation data",
        "Results per group, with intervals",
        "Ethical considerations",
        "Caveats and limitations",
        "The LLM screener audited in F8",
    ]
    assert headings == expected


def test_the_model_details_name_the_three_models_and_their_training_recipe():
    details = _sections(_read("docs/model_card.md"))["Model details"]
    for name in ("Main model", "Baseline model", "Scrubbed model"):
        assert f"### {name}" in details
    for word in ("seed 42", "learning rate", "epochs", "TF-IDF", "revision"):
        assert word in details


def test_the_metrics_section_says_what_each_metric_means_for_a_recruiter():
    metrics = _sections(_read("docs/model_card.md"))["Metrics"]
    for name in ("Accuracy", "Macro-F1", "True positive rate", "TPR gap and RMS TPR gap"):
        assert f"- **{name}" in metrics
    assert metrics.count("recruiter") >= 4


def test_the_group_results_give_each_showcase_job_with_intervals():
    results = _sections(_read("docs/model_card.md"))["Results per group, with intervals"]
    for job in SHOWCASE:
        assert re.search(rf"(?m)^\| {job} \|.*\[.*,.*\]", results), job


def test_the_documents_speak_in_the_first_person_singular():
    for name in WRITTEN:
        assert not re.search(r"\b([Ww]e|[Oo]ur|[Oo]urs)\b", _read(name)), name


def test_the_documents_write_hispanic_america_and_name_no_course():
    for name in WRITTEN:
        for line in _read(name).splitlines():
            if not line.startswith("_Avoid_"):
                assert "Latin America" not in line, name
        assert not re.search(r"(?i)\b(assignment|coursework|syllabus)\b", _read(name)), name
