import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITTEN = [
    "README.md",
    "GLOSSARY.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md")),
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "decisions").glob("*.md")),
]
SHOWCASE = ["surgeon", "software engineer", "composer", "dietitian", "nurse", "fashion model"]


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


ADRS = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "decisions").glob("*.md"))
STALE = re.compile(
    r"(?i)\b(not yet|is planned|are planned|to be (run|done)|pending|will be (run|done|written))\b"
)


def test_every_decision_record_is_accepted_and_has_an_outcomes_section():
    assert len(ADRS) == 8
    for name in ADRS:
        text = _read(name)
        assert re.search(r"(?m)^Status: accepted\b", text), name
        assert re.search(r"(?m)^## Outcomes\b", text), name


def test_no_decision_record_still_talks_about_work_as_planned():
    for name in ADRS:
        assert not STALE.search(_read(name)), name


def test_decision_records_do_not_point_to_numbers_the_readme_no_longer_holds():
    assert "in the README" not in _read("docs/decisions/0005-counterfactual-gender-swap.md")
    assert "Four results hold" not in _read("docs/decisions/0006-probing-and-patching.md")


def _glossary():
    """The terms of GLOSSARY.md with their Avoid words: {term: [avoided words]}."""
    entries = re.findall(r"(?m)^\*\*(.+?)\*\*:\n(?:.*\n)*?_Avoid_: (.*)$", _read("GLOSSARY.md"))
    return {term: [w.strip() for w in avoid.split(",")] for term, avoid in entries}


def _corpus():
    paths = [ROOT / name for name in WRITTEN if name != "GLOSSARY.md"]
    paths += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    return "\n".join(p.read_text(encoding="utf-8").lower() for p in paths)


def test_every_glossary_term_has_an_avoid_line():
    terms = re.findall(r"(?m)^\*\*(.+?)\*\*:$", _read("GLOSSARY.md"))
    assert len(terms) == len(_glossary())


def test_every_glossary_term_is_used_in_the_docs_or_the_code():
    corpus = _corpus()
    assert [term for term in _glossary() if term.lower() not in corpus] == []


def test_the_glossary_defines_the_terms_the_docs_rely_on():
    required = {"Shortlist gap", "Score gap", "Profile", "Yes-rate", "Merge rule", "Eligible bio"}
    assert required <= set(_glossary())


def test_a_defined_term_is_not_listed_as_one_to_avoid():
    terms = {term.lower() for term in _glossary()}
    avoided = {word.lower() for words in _glossary().values() for word in words}
    assert terms & avoided == set()


def test_the_terms_the_docs_use_for_the_gap_are_not_avoided():
    avoided = {word.lower() for words in _glossary().values() for word in words}
    assert "shortlist gap" not in avoided


def _readme_section(title):
    return _sections(_read("README.md"))[title]


def test_the_readme_is_a_landing_page_of_about_its_current_length():
    assert len(_read("README.md").splitlines()) <= 120


def test_key_findings_has_at_most_one_bullet_per_finding():
    readme = _read("README.md")
    assert "## Key findings\n" in readme and "so far" not in readme
    bullets = re.findall(r"(?m)^- \*\*", _readme_section("Key findings"))
    assert 1 <= len(bullets) <= 8
    labels = re.findall(r"\[F([1-8])\]\(", _readme_section("Key findings"))
    assert len(labels) == len(set(labels))


def test_the_repository_map_lists_the_notebook_the_model_card_and_the_methods_page():
    layout = _readme_section("Repository layout")
    for path in ("notebooks/", "docs/model_card.md", "docs/methods.md"):
        assert path in layout


def test_how_to_run_gives_a_time_or_not_recorded_for_every_step():
    section = _readme_section("How to run")
    rows = [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith(("| Step", "|---"))
    ]
    assert len(rows) >= 8
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 3 and cells[2], row
    commands = set(re.findall(r"`scripts/(\w+\.py)", section))
    checks = {"check_deuteranopia.py", "check_docs.py"}
    assert commands == {p.name for p in (ROOT / "scripts").glob("*.py")} - checks


def test_the_readme_cites_the_data_the_model_the_name_lists_and_the_llm():
    cite = _readme_section("Cite")
    for name in ("De-Arteaga", "DeBERTa", "INE", "Bonsai"):
        assert name in cite


NOTEBOOK = "notebooks/audit.ipynb"


def test_no_document_calls_the_shortlist_gap_the_gender_gap():
    for name in ("README.md", "docs/results.md", "docs/model_card.md", NOTEBOOK):
        assert "gender gap" not in _read(name).lower(), name


def test_no_document_says_gender_words_put_or_set_women_below_men():
    # a decomposition says what the words account for, not what they cause
    for name in ("README.md", "docs/results.md", NOTEBOOK, "report/main.tex"):
        text = " ".join(_read(name).split())
        assert not re.search(r"(?i)\b(?:put|puts|set|sets)\b[^.]{0,40}\b(?:below|above)\b", text), (
            name
        )


def test_the_readme_gives_the_command_that_runs_the_notebook_and_says_what_it_is():
    readme = _read("README.md")
    assert "pytest --nbmake notebooks/audit.ipynb" in readme
    assert "results viewer" in readme
    assert "docs/results.md#reproduce-every-result" in readme


def test_the_run_table_gives_measured_times_only():
    readme = _read("README.md")
    assert "not recorded" not in readme
    assert "only the measured times are given" in _readme_section("How to run")


def test_the_origin_groups_are_four_after_the_merge():
    for name in ("README.md", "docs/model_card.md", "GLOSSARY.md"):
        text = _read(name)
        assert not re.search(r"(?i)\bfive (?:origin )?groups", text), name
    assert "four origin groups" in _read("README.md")


def test_the_readme_limits_say_the_task_is_a_proxy_for_screening():
    limits = _readme_section("Limits and intended use")
    assert "predicting an occupation" in limits
    assert "proxy for screening" in limits


def _report():
    return " ".join(_read("report/main.tex").split())


def test_the_report_says_how_many_bios_each_stability_check_uses():
    assert "different seeds (100 bios)" in _report()


def test_the_report_says_the_intervals_are_too_wide_to_separate_what_it_compares():
    report = _report()
    assert "these per-job intervals overlap, so only the overall cut is established" in report
    assert "but each interval reaches about 7 to 8 points either side of its estimate" in report
    assert "so a smaller difference could not be seen" in report
    assert "The intervals overlap, so I find no reaction to gender beyond the edit" in report
    assert "would not show" in report


def test_the_report_gives_the_surgeon_gap_as_a_breakdown_and_says_so_once():
    report = _report()
    assert "net score gap is small" in report
    assert report.count("a breakdown of the score, not the effect of changing the words") == 1


def test_the_report_reads_freeones_as_a_possible_trace_of_the_work_or_the_site():
    report = _report()
    assert "It says nothing about the person's work" not in report
    assert "\paragraph{A gendered shortcut found with the explanations.}" in report
    assert "may reflect the work or where the bio was published" in report
    assert "I cannot tell which" in report
    assert "it may reflect the work or where some bios were published" in report


def test_the_report_says_the_mitigation_gap_averages_27_jobs():
    report = _report()
    assert "dietitian has no men left there, so the gap averages 27 jobs" in report
    assert "Change in gap vs (a)" in report


def test_the_report_says_the_stated_reasons_were_sorted_by_a_keyword_rule():
    report = _report()
    assert "by a keyword rule (82\% in a hand check of 50) give a reason" in report
    assert "was decided by a keyword rule, checked by hand on 50 cases (90\% agreement)" in report
    assert "labelled by one reader" not in report


def test_the_model_card_does_not_vouch_for_the_data_it_never_saw():
    assert "Real client data" not in _read("docs/model_card.md")


def test_the_mitigation_record_scopes_leace_out_of_the_audit_not_an_issue():
    text = _read("docs/decisions/0007-mitigation.md")
    assert "out of scope for this audit" in text
    assert "out of scope for this issue" not in text


def test_the_models_record_says_what_was_done_with_the_explanations_and_the_llm():
    text = _read("docs/decisions/0002-models.md")
    head = text.split("## Consequences")[0]
    assert "Integrated Gradients" not in head
    assert "with counterfactual swaps only" not in head
    for phrase in (
        "Shapley attribution",
        "hidden states for probing",
        "name headers and pronoun swaps",
    ):
        assert phrase in head


def test_the_results_page_reads_freeones_as_a_possible_trace_and_separates_the_breakdown():
    results = _read("docs/results.md")
    assert "may reflect" in results and "likely spurious correlation" not in results
    assert "not the effect of changing the words" in results


def test_the_results_page_explains_why_condition_a_differs_from_the_first_table():
    text = " ".join(_read("docs/results.md").split())
    assert "(0.148 against 0.147; dietitian 25 against 24 points)" in text
    assert "scored in batches of 4 instead of 64" in text


def test_the_report_describes_the_data_the_models_and_the_gap_plainly():
    report = _report()
    for phrase in (
        "English short biographies, each labelled with one of 28 jobs",
        "257,478 train, 39,642 dev and 99,069 test bios",
        "Common Crawl",
        "select on 10,000 of the dev bios",
        "(rapper) to 93\% (dietitian) in the training bios",
        "(dietitian, nurse, fashion model)",
        "reading at most 256 tokens",
        "TF-IDF (weighted counts of words and word pairs)",
        "overall shortlist gap is lower than the baseline's (0.148 against 0.167),",
        "in every one of the six jobs but software engineer",
        "explanations can help",
        "Change in gap vs (a)",
    ):
        assert phrase in report, phrase
    assert "biographies, text," not in report and "(dietitian, nurse, model)" not in report


def test_the_report_links_the_explanations_to_the_fix_and_states_the_ai_assistance():
    report = _report()
    assert "which is why I scrub them" in report
    assert "next hypothesis, that a few words carry the rest, and it does not hold" in report
    assert "which cut the gap by a fifth in one run" in report
    assert "I used an AI coding assistant for implementation and editing;" in report


def test_the_chance_level_is_named_as_the_95th_percentile_of_what_chance_gives():
    for name in ("report/main.tex", "README.md", "docs/results.md"):
        text = " ".join(_read(name).split())
        assert "twice what chance gives" not in text, name
        assert "twice the 95th percentile of what chance gives (0.050)" in text, name


def test_the_readme_reads_freeones_as_a_possible_trace_not_a_spurious_correlation():
    readme = _read("README.md")
    assert "may reflect" in readme and "spurious" not in readme
    assert "I did not check" not in readme


def test_the_readme_cite_section_states_the_ai_assistance():
    sentence = (
        "I used an AI coding assistant for implementation and editing; "
        "I designed the audit, chose the methods and checked every result."
    )
    assert sentence in _readme_section("Cite")


def test_the_report_prints_the_figures_wide_enough_to_keep_their_text_legible():
    report = _read("report/main.tex")
    by_figure = {figure: width for width, figure in re.findall(r"width=([^\]]+)\]\{f(\d)_", report)}
    for figure in "2356":
        assert by_figure[figure] == r"\columnwidth", figure
    for figure in "1478":
        assert by_figure[figure] == r"0.95\columnwidth", figure


def test_the_figure_captions_explain_what_a_reader_needs():
    report = _report()
    for phrase in (
        "The ten words with the largest contribution to the score gap",
        "The ten words make up 61\% of the fashion model gap",
        "no share is given for surgeon, whose gap interval includes 0",
        "emb.\ is the input layer",
        "pretrained is the same network before fine-tuning",
        "orange: rank correlation; grey: shared top 10 words",
        "intervals narrower than the marker are hidden",
        "the 5 points of the text are the outer bound of its interval, not a margin fixed",
    ):
        assert phrase in report, phrase


def _readme_headings():
    return re.findall(r"(?m)^## (.+)$", _read("README.md"))


def test_the_readme_links_the_report_near_the_top_and_in_how_to_read_this():
    readme = _read("README.md")
    assert "report/main.pdf" in "\n".join(readme.splitlines()[:6])
    assert "report/main.pdf" in readme.split("**How to read this.**")[1].split("## ")[0]


def test_the_readme_answers_why_in_its_first_screen():
    top = "\n".join(_read("README.md").splitlines()[:30])
    assert "Pronouns push scores towards the gender that usually holds a job" in top
    assert (
        "Other words that go with gender, such as a website name found only in women's bios" in top
    )
    assert "Gender is also still readable inside the model after the pronouns are removed" in top
    assert "uses explainable-AI methods to find out why" not in top


def test_why_it_matters_comes_right_after_the_opening_and_before_the_key_findings():
    headings = _readme_headings()
    assert headings.index("Why it matters") < headings.index("Key findings")


def test_what_the_audit_does_has_no_status_column_and_eight_rows():
    rows = [
        line
        for line in _readme_section("What the audit does").splitlines()
        if line.startswith("| ") and not line.startswith(("| Question", "|---"))
    ]
    assert len(rows) == 8
    assert "Status" not in _readme_section("What the audit does")
    assert "TPR" not in _readme_section("What the audit does")


def test_the_readme_defines_rms_and_tpr_once_under_the_results_table():
    readme = _read("README.md")
    assert readme.count("root mean square") == 1 and readme.count("true positive rate") == 1
    assert "root mean square" in _readme_section("Results")


def test_the_readme_layout_lists_the_report_and_the_prompts_and_the_run_note_is_there():
    layout = _readme_section("Repository layout")
    assert "report/" in layout and "prompts/" in layout
    assert "uv run python scripts/<name>.py" in _readme_section("How to run")


def test_every_figure_in_the_report_prints_at_its_true_size():
    # a figure drawn 3.03 in wide scaled to the column would print its 7 pt text smaller
    widths = re.findall(r"\includegraphics\[width=([^\]]*)\]", _read("report/main.tex"))
    assert len(widths) == 8
    assert set(widths) == {r"\columnwidth"}
