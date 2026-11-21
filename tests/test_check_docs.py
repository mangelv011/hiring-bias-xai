import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_docs.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path, files):
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def _errors(tmp_path, files):
    return _load().check_links(_repo(tmp_path, files))


def test_a_repo_with_resolving_links_and_images_has_no_errors(tmp_path):
    files = {
        "README.md": "[res](docs/results.md) ![fig](results/f.png)\n",
        "docs/results.md": "# Results\n\n[back](../README.md)\n",
        "results/f.png": "x",
    }
    assert _errors(tmp_path, files) == []


def test_a_missing_link_target_is_reported_with_file_and_target(tmp_path):
    errors = _errors(tmp_path, {"README.md": "[x](docs/missing.md)\n"})
    assert len(errors) == 1
    assert "README.md" in errors[0] and "docs/missing.md" in errors[0]


def test_a_missing_image_is_reported(tmp_path):
    errors = _errors(tmp_path, {"README.md": "![x](results/none.png)\n"})
    assert len(errors) == 1 and "results/none.png" in errors[0]


def test_an_html_image_source_is_checked(tmp_path):
    files = {"README.md": '<img src="results/none.png" width="520" alt="a">\n'}
    errors = _errors(tmp_path, files)
    assert len(errors) == 1 and "results/none.png" in errors[0]


def test_an_html_image_source_that_exists_passes(tmp_path):
    files = {"README.md": '<img src="results/f.png" alt="a">\n', "results/f.png": "x"}
    assert _errors(tmp_path, files) == []


def test_links_resolve_from_the_folder_of_the_document(tmp_path):
    files = {
        "docs/results.md": "[a](decisions/0001-a.md) [b](../results/f.csv)\n",
        "docs/decisions/0001-a.md": "# A\n",
        "results/f.csv": "x",
    }
    assert _errors(tmp_path, files) == []
    files = {"docs/results.md": "[a](docs/decisions/0001-a.md)\n"}
    assert len(_errors(tmp_path / "other", files)) == 1


def test_a_link_to_a_folder_resolves(tmp_path):
    files = {"README.md": "[d](docs/decisions/)\n", "docs/decisions/0001-a.md": "# A\n"}
    assert _errors(tmp_path, files) == []


def test_external_links_are_not_followed(tmp_path):
    files = {"README.md": "[a](https://example.org/x) [b](mailto:a@b.c) [c](http://e.org)\n"}
    assert _errors(tmp_path, files) == []


def test_a_link_title_is_not_part_of_the_path(tmp_path):
    files = {"README.md": '[a](docs/a.md "the title")\n', "docs/a.md": "# A\n"}
    assert _errors(tmp_path, files) == []


def test_a_percent_encoded_path_is_decoded(tmp_path):
    files = {"README.md": "[a](docs/a%20b.md)\n", "docs/a b.md": "# A\n"}
    assert _errors(tmp_path, files) == []


def test_a_path_with_the_wrong_case_is_reported_because_github_is_case_sensitive(tmp_path):
    files = {"README.md": "[a](Docs/a.md)\n", "docs/a.md": "# A\n"}
    errors = _errors(tmp_path, files)
    assert len(errors) == 1 and "Docs/a.md" in errors[0]


def test_links_inside_code_are_not_checked(tmp_path):
    text = "```\n[a](docs/none.md)\n```\n\nUse `[b](docs/none.md)` here.\n"
    assert _errors(tmp_path, {"README.md": text}) == []


def test_every_document_in_the_three_places_is_read(tmp_path):
    files = {
        "README.md": "[x](gone1.md)\n",
        "docs/results.md": "[x](gone2.md)\n",
        "docs/decisions/0001-a.md": "[x](gone3.md)\n",
    }
    errors = _errors(tmp_path, files)
    assert len(errors) == 3


def test_a_document_outside_the_three_places_is_not_read(tmp_path):
    files = {"notes.md": "[x](gone.md)\n", "docs/deep/more/x.md": "[x](gone.md)\n"}
    assert _errors(tmp_path, files) == []


def test_each_broken_link_is_one_error_and_line_numbers_are_given(tmp_path):
    errors = _errors(tmp_path, {"README.md": "ok\n\n[a](x.md) and [b](y.md)\n"})
    assert len(errors) == 2
    assert all("README.md:3" in error for error in errors)


def test_an_anchor_that_matches_a_heading_resolves(tmp_path):
    files = {
        "README.md": "[a](docs/r.md#f1-the-gap-follows-the-share-of-women) [b](#top)\n\n# Top\n",
        "docs/r.md": "## F1: the gap follows the share of women\n",
    }
    assert _errors(tmp_path, files) == []


def test_a_missing_anchor_is_reported_with_the_target_file(tmp_path):
    files = {"README.md": "[a](docs/r.md#nope)\n", "docs/r.md": "# Yes\n"}
    errors = _errors(tmp_path, files)
    assert len(errors) == 1 and "#nope" in errors[0]


def test_a_missing_anchor_in_the_same_document_is_reported(tmp_path):
    errors = _errors(tmp_path, {"README.md": "[a](#nope)\n\n# Yes\n"})
    assert len(errors) == 1 and "#nope" in errors[0]


def test_slugs_drop_punctuation_lowercase_and_join_with_hyphens():
    slug = _load().slug
    assert (
        slug("F8: does an LLM screener react to names?") == "f8-does-an-llm-screener-react-to-names"
    )
    assert slug("Is the gap real?") == "is-the-gap-real"
    assert slug("What I did not check") == "what-i-did-not-check"
    assert slug("Data and models") == "data-and-models"


def test_slugs_keep_hyphens_underscores_digits_and_letters_with_accents():
    slug = _load().slug
    assert slug("DeBERTa-v3-small, fine_tuned") == "deberta-v3-small-fine_tuned"
    assert slug("Ética 2") == "ética-2"


def test_slugs_keep_every_space_as_a_hyphen():
    assert _load().slug("A  b - c") == "a--b---c"


def test_heading_markup_is_removed_before_the_slug(tmp_path):
    files = {
        "README.md": "[a](#score-in-code) [b](#a-link-here) [c](#bold-and-em)\n\n"
        "## Score in `code`\n\n## A [link](x.md) here\n\n## **Bold** and *em*\n",
        "x.md": "x",
    }
    assert _errors(tmp_path, files) == []


def test_a_repeated_heading_gets_a_numbered_slug(tmp_path):
    text = "[a](#same) [b](#same-1) [c](#same-2) [d](#same-3)\n\n# Same\n\n# Same\n\n# Same\n"
    errors = _errors(tmp_path, {"README.md": text})
    assert len(errors) == 1 and "#same-3" in errors[0]


def test_hashes_inside_fenced_code_are_not_headings(tmp_path):
    text = "[a](#comment)\n\n```bash\n# comment\n```\n"
    assert len(_errors(tmp_path, {"README.md": text})) == 1


def test_a_heading_needs_a_space_after_the_hashes_and_at_most_six_hashes(tmp_path):
    text = "[a](#tag) [b](#seven)\n\n#tag\n\n####### Seven\n"
    assert len(_errors(tmp_path, {"README.md": text})) == 2


def test_an_anchor_on_a_non_markdown_file_is_not_checked(tmp_path):
    files = {"README.md": "[a](results/f.csv#row1)\n", "results/f.csv": "x"}
    assert _errors(tmp_path, files) == []


def test_an_anchor_is_compared_with_the_exact_slug(tmp_path):
    files = {"README.md": "[a](#Top)\n\n# Top\n"}
    assert len(_errors(tmp_path, files)) == 1


def _numbers(tmp_path, readme, cells, allowed=(), results=None, documents=("README.md",)):
    """Run check_numbers on a README and results files; `results` maps file name to content."""
    module = _load()
    files = {"README.md": readme}
    files.update({f"results/{name}": content for name, content in (results or {}).items()})
    root = _repo(tmp_path, files)
    table = [module.Cell(*cell) for cell in cells]
    allow = [module.Allowed(*entry) for entry in allowed]
    return module.check_numbers(root, table, allow, documents)


ACCURACY = '{"test": {"accuracy": {"value": 0.8370, "ci_low": 0.8346}}, "n": 99069}'
RIGHT = ("README.md", r"right on ([\d.]+)%", "m.json", "test.accuracy.value", "pct1")


def test_a_number_that_matches_its_results_value_leaves_no_error(tmp_path):
    assert _numbers(tmp_path, "right on 83.7%\n", [RIGHT], results={"m.json": ACCURACY}) == []


def test_a_number_that_differs_from_the_results_is_reported_with_both_values(tmp_path):
    errors = _numbers(tmp_path, "right on 84.7%\n", [RIGHT], results={"m.json": ACCURACY})
    assert len(errors) == 1
    assert "84.7" in errors[0] and "83.7" in errors[0] and "README.md:1" in errors[0]


def test_a_row_whose_pattern_is_not_found_is_reported(tmp_path):
    errors = _numbers(tmp_path, "nothing here\n", [RIGHT], results={"m.json": ACCURACY})
    assert len(errors) == 1 and "right on" in errors[0]


def test_every_match_of_a_pattern_is_checked(tmp_path):
    text = "right on 83.7%\n\nright on 80.0%\n"
    errors = _numbers(tmp_path, text, [RIGHT], results={"m.json": ACCURACY})
    assert len(errors) == 1 and "README.md:3" in errors[0]


def test_a_number_outside_the_table_and_the_allow_list_is_reported(tmp_path):
    errors = _numbers(tmp_path, "I read 12 bios and 3.5 more\n", [])
    assert len(errors) == 2
    assert "12" in errors[0] and "README.md:1" in errors[0] and "3.5" in errors[1]


def test_an_allowed_phrase_covers_the_numbers_inside_it_and_no_others(tmp_path):
    allowed = [("README.md", r"28 jobs", "the number of occupations")]
    assert _numbers(tmp_path, "It reads 28 jobs\n", [], allowed) == []
    assert len(_numbers(tmp_path / "b", "It reads 27 jobs\n", [], allowed)) == 1
    assert len(_numbers(tmp_path / "c", "It reads 28 jobs and 28 more\n", [], allowed)) == 1


def test_an_allowed_phrase_of_another_document_does_not_apply(tmp_path):
    allowed = [("docs/model_card.md", r"28 jobs", "the number of occupations")]
    assert len(_numbers(tmp_path, "It reads 28 jobs\n", [], allowed)) == 1


def test_an_allowed_phrase_for_all_documents_applies(tmp_path):
    allowed = [(None, r"\b2019\b", "a year")]
    assert _numbers(tmp_path, "De-Arteaga et al., 2019\n", [], allowed) == []


def test_numbers_with_thousands_separators_are_one_number(tmp_path):
    cells = [("README.md", r"on ([\d,]+) bios", "m.json", "n", "count")]
    assert _numbers(tmp_path, "on 99,069 bios\n", cells, results={"m.json": ACCURACY}) == []


def test_numbers_in_a_list_are_separate_numbers(tmp_path):
    assert len(_numbers(tmp_path, "Between [3, 5] jobs\n", [])) == 2


def test_numbers_glued_to_letters_are_names_not_numbers(tmp_path):
    text = "DeBERTa-v3-small and F1 and 9B and f1_gap and x_2\n"
    assert _numbers(tmp_path, text, []) == []


def test_numbers_in_code_and_link_targets_are_not_read(tmp_path):
    text = "Run `scripts/a.py --n 5` and see [it](docs/0007-a.md) and ![x](r/f1.png).\n"
    text += '<img src="results/f9.png" alt="no number">\n```\nuv run 30 minutes\n```\n'
    assert _numbers(tmp_path, text, []) == []


def test_numbers_in_alt_text_and_tables_are_read(tmp_path):
    text = '<img src="results/f.png" alt="the gap is 0.5">\n\n| a | 7 |\n|---|---|\n'
    assert len(_numbers(tmp_path, text, [])) == 2


def test_a_signed_number_is_compared_with_its_sign(tmp_path):
    results = {"m.json": '{"d": -0.0047}'}
    cells = [("README.md", r"change ([-+]\d+\.\d)", "m.json", "d", "signed_points1")]
    assert _numbers(tmp_path, "change -0.5\n", cells, results=results) == []
    assert len(_numbers(tmp_path / "b", "change +0.5\n", cells, results=results)) == 1


def test_formats_round_half_up_on_the_decimal_value():
    formats = _load().FORMATS
    assert formats["per100"](0.755) == "76"
    assert formats["per100"](0.754) == "75"
    assert formats["pct1"](0.8370) == "83.7"
    assert formats["dec3"](0.1476) == "0.148"
    assert formats["dec3"](0.1475) == "0.148"
    assert formats["count"](99069) == "99,069"
    assert formats["signed_points1"](0.0047) == "+0.5"
    assert formats["signed_points1"](-0.0047) == "-0.5"
    assert formats["hours2"](10773.6) == "2.99"
    assert formats["minutes0"](296.6) == "5"


def test_a_value_is_read_from_a_list_by_position(tmp_path):
    results = {"m.json": '{"runs": [{"n": 5}, {"n": 7}]}'}
    cells = [("README.md", r"has (\d+) runs", "m.json", "runs.1.n", "count")]
    assert _numbers(tmp_path, "has 7 runs\n", cells, results=results) == []


def test_a_value_is_read_from_a_csv_row_by_its_keys(tmp_path):
    results = {"t.csv": "screener,occupation,flip_rate\ndeberta,all,0.069\nbaseline,all,0.135\n"}
    path = "screener=baseline,occupation=all:flip_rate"
    cells = [("README.md", r"flips (\d+) of", "t.csv", path, "per100")]
    assert _numbers(tmp_path, "flips 14 of\n", cells, results=results) == []


def test_a_csv_path_that_matches_no_row_or_many_rows_is_an_error(tmp_path):
    results = {"t.csv": "a,v\nx,1\nx,2\n"}
    for name, path in (("none", "a=y:v"), ("many", "a=x:v")):
        cells = [("README.md", r"is (\d+)", "t.csv", path, "count")]
        errors = _numbers(tmp_path / name, "is 1\n", cells, results=results)
        assert len(errors) == 1 and path in errors[0]


def test_a_missing_results_file_or_json_path_is_an_error(tmp_path):
    cells = [("README.md", r"is (\d+)", "gone.json", "a", "count")]
    assert len(_numbers(tmp_path, "is 1\n", cells)) == 1
    cells = [("README.md", r"is (\d+)", "m.json", "no.such.path", "count")]
    assert len(_numbers(tmp_path / "b", "is 1\n", cells, results={"m.json": ACCURACY})) == 1


def test_a_cell_for_a_document_that_is_not_checked_is_an_error(tmp_path):
    cells = [("docs/x.md", r"is (\d+)", "m.json", "n", "count")]
    errors = _numbers(tmp_path, "nothing\n", cells, results={"m.json": ACCURACY})
    assert len(errors) == 1 and "docs/x.md" in errors[0]


def test_each_checked_document_is_read_in_order(tmp_path):
    root = _repo(tmp_path, {"README.md": "a 5\n", "docs/model_card.md": "b 6\n"})
    errors = _load().check_numbers(root, [], [], ("README.md", "docs/model_card.md"))
    assert [error.split(":")[0] for error in errors] == ["README.md", "docs/model_card.md"]


def test_count_rounds_half_up_and_keeps_separators():
    count = _load().FORMATS["count"]
    assert count(12410.9) == "12,411"
    assert count(6167.4) == "6,167"
    assert count(0.5) == "1"
    assert count(1721) == "1,721"


def test_the_other_formats_the_documents_use():
    formats = _load().FORMATS
    assert formats["abs_per100"](-0.0601) == "6"
    assert formats["abs_points1"](-0.1012) == "10.1"
    assert formats["learning_rate"](3e-05) == "3e-5"
    assert formats["learning_rate"](5e-05) == "5e-5"
    assert formats["seconds0"](32.4) == "32"
    assert formats["minutes1"](144.7) == "2.4"
    assert formats["of150"](0.36) == "54"


REPO = Path(__file__).resolve().parents[1]


def test_the_numbers_of_the_readme_and_the_model_card_match_results():
    assert _load().check_numbers(REPO) == []


def test_main_returns_zero_on_this_repository():
    assert _load().main(REPO) == 0


def test_main_prints_each_problem_and_returns_one(tmp_path, capsys):
    root = _repo(tmp_path, {"README.md": "[a](gone.md) 12\n", "docs/model_card.md": "x\n"})
    assert _load().main(root) == 1
    lines = capsys.readouterr().out.splitlines()
    assert any("gone.md" in line for line in lines)
    assert any("number 12" in line for line in lines)


def test_ci_runs_the_docs_check():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "- run: python scripts/check_docs.py\n" in ci


def test_an_indented_hash_line_of_four_spaces_is_not_a_heading(tmp_path):
    text = "[a](#code)\n\n    # code\n"
    assert len(_errors(tmp_path, {"README.md": text})) == 1


def test_a_tilde_fence_hides_headings_and_links(tmp_path):
    text = "[a](#hidden)\n\n~~~\n# hidden\n[b](gone.md)\n~~~\n"
    errors = _errors(tmp_path, {"README.md": text})
    assert len(errors) == 1 and "#hidden" in errors[0]


def test_a_link_that_leaves_the_repository_is_reported(tmp_path):
    root = tmp_path / "repo"
    (tmp_path / "outside.md").write_text("x", encoding="utf-8")
    errors = _errors(root, {"README.md": "[a](../outside.md)\n"})
    assert len(errors) == 1 and "../outside.md" in errors[0]


def test_a_percent_encoded_anchor_is_decoded(tmp_path):
    files = {"README.md": "[a](#%C3%A9tica-2)\n\n# Ética 2\n"}
    assert _errors(tmp_path, files) == []


def test_a_broken_link_with_a_title_is_still_reported(tmp_path):
    errors = _errors(tmp_path, {"README.md": '[a](gone.md "the title")\n'})
    assert len(errors) == 1 and "gone.md" in errors[0]


def test_an_uncovered_number_with_a_thousands_separator_is_one_error(tmp_path):
    errors = _numbers(tmp_path, "I read 1,721 bios\n", [])
    assert len(errors) == 1 and "1,721" in errors[0]


def test_only_the_group_of_a_row_is_covered_not_the_rest_of_the_match(tmp_path):
    cells = [("README.md", r"right on ([\d.]+)% of 100", "m.json", "test.accuracy.value", "pct1")]
    errors = _numbers(tmp_path, "right on 83.7% of 100\n", cells, results={"m.json": ACCURACY})
    assert len(errors) == 1 and "number 100" in errors[0]


def test_a_missing_document_is_reported_by_name(tmp_path):
    root = _repo(tmp_path, {"README.md": "x\n"})
    errors = _load().check_numbers(root, [], [], ("README.md", "docs/model_card.md"))
    assert errors == ["docs/model_card.md: document is missing"]


def test_both_documents_are_checked_and_every_row_belongs_to_one_of_them():
    module = _load()
    assert module.NUMBER_DOCUMENTS == ("README.md", "docs/model_card.md")
    assert {cell.document for cell in module.CELLS} == set(module.NUMBER_DOCUMENTS)
