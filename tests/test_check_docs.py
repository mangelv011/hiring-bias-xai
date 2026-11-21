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


def test_an_indented_fence_hides_headings_and_links(tmp_path):
    text = "[a](#hidden)\n\n  ```\n  # hidden\n  [b](gone.md)\n  ```\n"
    errors = _errors(tmp_path, {"README.md": text})
    assert len(errors) == 1 and "#hidden" in errors[0]


def test_triple_backticks_in_the_middle_of_a_line_do_not_open_a_fence(tmp_path):
    text = "Use ```x``` here\n\n[a](gone.md)\n"
    assert len(_errors(tmp_path, {"README.md": text})) == 1


def test_an_empty_code_span_is_removed_from_prose(tmp_path):
    assert list(_load().prose_lines("a `` [b](c.md)\n")) == [(1, "a  [b](c.md)")]


def test_an_html_href_is_checked(tmp_path):
    errors = _errors(tmp_path, {"README.md": '<a href="gone.md">x</a>\n'})
    assert len(errors) == 1 and "gone.md" in errors[0]


def test_an_html_reference_with_spaces_single_quotes_or_a_prefix(tmp_path):
    def refs(text):
        return [t for _, t in _load().references(text)]

    assert refs("<img src = 'a.png'>") == ["a.png"]
    assert refs('<img xsrc="a.png">') == []


def test_a_markdown_image_target_is_checked_like_a_link():
    assert [t for _, t in _load().references("![alt](a.png) [b](c.md)")] == ["a.png", "c.md"]


def test_closing_hashes_and_trailing_blanks_are_not_part_of_the_slug(tmp_path):
    text = "[a](#top) [b](#two)\n\n# Top #\n\n## Two  \n"
    assert _errors(tmp_path, {"README.md": text}) == []


def test_html_tags_are_removed_from_the_slug():
    assert _load().slug("A <b>bold</b> word") == "a-bold-word"


def test_an_empty_fragment_is_not_a_missing_heading(tmp_path):
    files = {"README.md": "[a](docs/a.md#)\n", "docs/a.md": "# A\n"}
    assert _errors(tmp_path, files) == []


def test_the_scheme_of_an_external_link_may_have_digits_dots_pluses_and_capitals(tmp_path):
    files = {"README.md": "[a](HTTPS://e.org) [b](git+ssh://h/x) [c](web.v2:x)\n"}
    assert _errors(tmp_path, files) == []


def test_an_unresolved_link_is_reported_once_without_an_anchor_error(tmp_path):
    errors = _errors(tmp_path, {"README.md": "[a](gone.md#x)\n"})
    assert len(errors) == 1 and "does not resolve" in errors[0]


def test_a_missing_anchor_error_names_the_line_and_the_file(tmp_path):
    errors = _errors(tmp_path, {"README.md": "ok\n[a](#nope)\n"})
    assert errors == ["README.md:2: #nope has no such heading"]


def test_error_names_use_forward_slashes_for_nested_documents(tmp_path):
    errors = _errors(tmp_path, {"docs/decisions/0001-a.md": "[a](gone.md)\n"})
    assert errors == ["docs/decisions/0001-a.md:1: gone.md does not resolve"]


def test_documents_other_than_markdown_in_docs_are_not_read(tmp_path):
    files = {"docs/notes.txt": "[a](gone.md)\n", "docs/results.md": "# R\n"}
    assert _errors(tmp_path, files) == []


def test_documents_are_listed_in_a_fixed_order(tmp_path):
    root = _repo(
        tmp_path,
        {
            "README.md": "x",
            "docs/b.md": "x",
            "docs/a.md": "x",
            "docs/decisions/0002-b.md": "x",
            "docs/decisions/0001-a.md": "x",
        },
    )
    names = [p.relative_to(root).as_posix() for p in _load().documents(root)]
    assert names == [
        "README.md",
        "docs/a.md",
        "docs/b.md",
        "docs/decisions/0001-a.md",
        "docs/decisions/0002-b.md",
    ]


def test_the_directory_of_a_link_must_also_match_in_case(tmp_path):
    files = {"README.md": "[a](docs/a.md) [b](DOCS/a.md)\n", "docs/a.md": "# A\n"}
    errors = _errors(tmp_path, files)
    assert len(errors) == 1 and "DOCS/a.md" in errors[0]


def test_default_root_is_the_repository_of_the_script():
    assert _load().REPO_ROOT == REPO


def test_halves_round_up_in_the_fixed_formats():
    formats = _load().FORMATS
    assert formats["per100"](0.745) == "75"
    assert formats["per100"](0.745) != "74"
    assert formats["dec3"](0.1485) == "0.149"
    assert formats["dec2"](0.125) == "0.13"
    assert formats["pct1"](0.8365) == "83.7"
    assert formats["abs_points1"](-0.8365) == "83.7"


def test_a_float_that_is_just_below_a_half_in_binary_rounds_by_its_written_value():
    assert _load().FORMATS["dec3"](0.0015) == "0.002"
    assert _load().FORMATS["per100"](0.285) == "29"


def test_formats_pad_decimals_and_keep_one_decimal_of_minutes():
    formats = _load().FORMATS
    assert formats["pct1"](0.5) == "50.0"
    assert formats["dec2"](0.5) == "0.50"
    assert formats["dec3"](0.5) == "0.500"
    assert formats["signed_points1"](0.0) == "+0.0"
    assert formats["minutes1"](60) == "1.0"
    assert formats["hours2"](3600) == "1.00"
    assert formats["of150"](1) == "150"
    assert formats["learning_rate"](1e-4) == "1e-4"
    assert formats["learning_rate"](3e-05) == "3e-5"


def test_a_csv_path_with_no_matching_row_names_the_cause(tmp_path):
    results = {"t.csv": "a,v\nx,1\n"}
    cells = [("README.md", r"is (\d+)", "t.csv", "a=y:v", "count")]
    errors = _numbers(tmp_path, "is 1\n", cells, results=results)
    assert len(errors) == 1 and "0 rows match" in errors[0]


def test_a_list_position_past_the_end_is_an_error_not_a_crash(tmp_path):
    results = {"m.json": '{"runs": [{"n": 5}]}'}
    cells = [("README.md", r"has (\d+) runs", "m.json", "runs.3.n", "count")]
    errors = _numbers(tmp_path, "has 7 runs\n", cells, results=results)
    assert len(errors) == 1 and "runs.3.n" in errors[0]


def test_a_missing_key_is_one_error_naming_file_path_and_cause(tmp_path):
    cells = [("README.md", r"is (\d+)", "m.json", "no.such", "count")]
    errors = _numbers(tmp_path, "is 1\n", cells, results={"m.json": ACCURACY})
    assert len(errors) == 1
    assert "m.json" in errors[0] and "no.such" in errors[0] and "KeyError" in errors[0]


def test_a_row_whose_pattern_is_not_found_does_not_also_read_its_results(tmp_path):
    cells = [("README.md", r"is (\d+)", "gone.json", "a", "count")]
    errors = _numbers(tmp_path, "nothing\n", cells)
    assert len(errors) == 1 and "no text matches" in errors[0]


def test_a_missing_document_alone_is_one_error_even_with_rows_for_it(tmp_path):
    root = _repo(tmp_path, {"README.md": "x\n"})
    cells = [_load().Cell("docs/model_card.md", r"is (\d+)", "m.json", "n", "count")]
    errors = _load().check_numbers(root, cells, [], ("README.md", "docs/model_card.md"))
    assert errors == ["docs/model_card.md: document is missing"]


def test_the_written_number_must_equal_the_formatted_value_text(tmp_path):
    cells = [("README.md", r"right on ([\d.]+)%", "m.json", "test.accuracy.value", "pct1")]
    assert len(_numbers(tmp_path, "right on 83.70%\n", cells, results={"m.json": ACCURACY})) == 1
    assert (
        len(_numbers(tmp_path / "b", "right on 083.7%\n", cells, results={"m.json": ACCURACY})) == 1
    )


def test_a_count_must_keep_its_thousands_separator(tmp_path):
    cells = [("README.md", r"on ([\d,]+) bios", "m.json", "n", "count")]
    errors = _numbers(tmp_path, "on 99069 bios\n", cells, results={"m.json": ACCURACY})
    assert len(errors) == 1


def test_a_prefix_of_the_expected_value_is_a_mismatch(tmp_path):
    cells = [("README.md", r"right on ([\d.]+)%", "m.json", "test.accuracy.value", "pct1")]
    assert len(_numbers(tmp_path, "right on 83%\n", cells, results={"m.json": ACCURACY})) == 1


def test_an_error_for_a_number_names_the_line(tmp_path):
    errors = _numbers(tmp_path, "ok\n\nread 12\n", [])
    assert errors == ["README.md:3: number 12 is covered by no row or allowed phrase"]


def test_a_mismatch_names_line_both_values_file_and_path(tmp_path):
    errors = _numbers(tmp_path, "x\nright on 84.7%\n", [RIGHT_], results={"m.json": ACCURACY})
    assert errors == ["README.md:2: 84.7 in the text, 83.7 in m.json test.accuracy.value"]


RIGHT_ = ("README.md", r"right on ([\d.]+)%", "m.json", "test.accuracy.value", "pct1")


def test_a_number_at_the_start_of_a_line_is_a_number(tmp_path):
    errors = _numbers(tmp_path, "12 bios\n", [])
    assert len(errors) == 1 and "number 12" in errors[0]


def test_a_number_followed_by_an_underscore_is_a_name(tmp_path):
    assert _numbers(tmp_path, "the 3_foo column\n", []) == []


def test_a_number_ending_a_line_is_a_number(tmp_path):
    assert len(_numbers(tmp_path, "read bios 12\n", [])) == 1


def test_an_allowed_phrase_equal_to_the_number_covers_it(tmp_path):
    allowed = [("README.md", r"\b28\b", "occupations")]
    assert _numbers(tmp_path, "It reads 28\n", [], allowed) == []


def test_an_allowed_phrase_overlapping_a_number_without_containing_it_does_not_cover_it(tmp_path):
    allowed = [("README.md", r"jobs 2", "x")]
    assert len(_numbers(tmp_path, "jobs 28\n", [], allowed)) == 1


def test_every_occurrence_of_an_allowed_phrase_in_a_line_covers(tmp_path):
    allowed = [("README.md", r"\b28\b", "x")]
    assert _numbers(tmp_path, "28 and 28\n", [], allowed) == []


def test_coverage_is_per_line(tmp_path):
    allowed = [("README.md", r"\b28\b", "x")]
    errors = _numbers(tmp_path, "28 jobs\nand 12 jobs\n", [], allowed)
    assert len(errors) == 1 and "README.md:2" in errors[0]


def test_a_row_covers_the_number_of_its_line_only(tmp_path):
    text = "right on 83.7%\n\nread 83.7\n"
    errors = _numbers(tmp_path, text, [RIGHT_], results={"m.json": ACCURACY})
    assert len(errors) == 1 and "README.md:3" in errors[0]


def test_two_rows_cover_two_numbers_of_the_same_line(tmp_path):
    cells = [RIGHT_, ("README.md", r"and ([\d,]+) bios", "m.json", "n", "count")]
    assert (
        _numbers(tmp_path, "right on 83.7% and 99,069 bios\n", cells, results={"m.json": ACCURACY})
        == []
    )


def test_a_markdown_link_text_number_is_read_but_its_target_is_not(tmp_path):
    errors = _numbers(tmp_path, "[the 12 jobs](docs/results.md)\n", [])
    assert len(errors) == 1 and "number 12" in errors[0]


def test_an_html_target_number_is_not_read_but_the_alt_is(tmp_path):
    text = '<img src="results/2.png" alt="the 7 jobs">\n'
    errors = _numbers(tmp_path, text, [])
    assert len(errors) == 1 and "number 7" in errors[0]


def test_two_links_on_one_line_are_both_blanked(tmp_path):
    assert _numbers(tmp_path, "[a](d/1.md) [b](d/2.md)\n", []) == []


def test_a_number_straight_after_a_link_target_is_not_joined_to_it(tmp_path):
    errors = _numbers(tmp_path, "[a](d/x.md)7\n", [])
    assert len(errors) == 1 and "number 7" in errors[0]


def test_fenced_code_is_not_read_for_numbers_even_with_tilde(tmp_path):
    assert _numbers(tmp_path, "~~~\nrun 30 minutes\n~~~\n", []) == []


def test_numbers_read_with_decimals_and_separators_are_whole(tmp_path):
    errors = _numbers(tmp_path, "see 1,234.5 and 12.75\n", [])
    assert [e.split(" number ")[1].split(" ")[0] for e in errors] == ["1,234.5", "12.75"]


def test_the_loose_allowed_phrases_of_the_repository_cover_only_what_they_name(tmp_path):
    module = _load()
    for text in ("27 jobs\n", "about 4 minutes\n", "in 96% of\n", "101 bios\n", "ADR 12\n"):
        errors = module.check_numbers(
            _repo(tmp_path / str(abs(hash(text))), {"README.md": text}),
            [],
            module.ALLOWED,
            ("README.md",),
        )
        assert len(errors) == 1, text


def test_the_allowed_phrase_of_another_document_does_not_apply_in_the_repository(tmp_path):
    module = _load()
    root = _repo(
        tmp_path, {"README.md": "x\n", "docs/model_card.md": 'about 30 seconds\nwidth="520"\n'}
    )
    errors = module.check_numbers(root, [], module.ALLOWED, ("README.md", "docs/model_card.md"))
    assert len(errors) == 2 and all(e.startswith("docs/model_card.md") for e in errors)


def test_a_row_of_the_table_for_the_card_is_not_applied_to_the_readme():
    module = _load()
    assert all(c.document in module.NUMBER_DOCUMENTS for c in module.CELLS)
    assert any(c.document == "docs/model_card.md" for c in module.CELLS)
    assert any(c.document == "README.md" for c in module.CELLS)


def test_a_screener_row_checks_the_value_and_the_two_ends_of_the_interval_by_column(tmp_path):
    module = _load()
    row = "| Main | 83.7 [83.5, 83.9] | 0.700 [0.690, 0.710] | 0.100 [0.090, 0.110] |\n"
    results = {
        "m.json": '{"a": {"value": 0.837, "ci_low": 0.835, "ci_high": 0.839},'
        ' "f": {"value": 0.7, "ci_low": 0.69, "ci_high": 0.71},'
        ' "g": {"value": 0.1, "ci_low": 0.09, "ci_high": 0.11}}'
    }
    cells = module.screener_rows("README.md", "Main", "m.json", "a", "f", "g")
    assert _numbers(tmp_path, row, [tuple(c) for c in cells], results=results) == []
    for bad in (
        row.replace("83.5", "83.4"),
        row.replace("83.9", "83.8"),
        row.replace("83.7", "83.6"),
        row.replace("0.690", "0.680"),
        row.replace("0.110", "0.120"),
        row.replace("| Main", "| Other"),
    ):
        assert _numbers(
            tmp_path / str(abs(hash(bad))), bad, [tuple(c) for c in cells], results=results
        )


def test_a_table_row_is_anchored_at_the_line_start_and_its_label_is_literal(tmp_path):
    module = _load()
    cells = [tuple(c) for c in module.interval("README.md", "A.B", 0, "m.json", "a", "pct1")]
    results = {"m.json": '{"a": {"value": 0.837, "ci_low": 0.835, "ci_high": 0.839}}'}
    good = "| A.B | 83.7 [83.5, 83.9] |\n"
    assert _numbers(tmp_path, good, cells, results=results) == []
    for bad in ("text | A.B | 83.7 [83.5, 83.9] |\n", "| AxB | 83.7 [83.5, 83.9] |\n"):
        errors = _numbers(tmp_path / str(abs(hash(bad))), bad, cells, results=results)
        assert any("no text matches" in e for e in errors)


def test_a_signed_interval_with_bold_marks_is_read(tmp_path):
    module = _load()
    cells = [
        tuple(c) for c in module.interval("README.md", "A", 0, "m.json", "a", "signed_points1")
    ]
    results = {"m.json": '{"a": {"value": -0.0047, "ci_low": -0.0100, "ci_high": 0.0012}}'}
    row = "| A | **-0.5** [-1.0, +0.1] |\n"
    assert _numbers(tmp_path, row, cells, results=results) == []


def test_plain_cells_need_the_closing_bar(tmp_path):
    module = _load()
    cell = tuple(module.plain("README.md", "A", 1, "m.json", "n", "count"))
    results = {"m.json": '{"n": 1721}'}
    assert _numbers(tmp_path, "| A | x | 1,721 |\n", [cell], results=results) == []
    assert any(
        "no text" in e
        for e in _numbers(tmp_path / "b", "| A | x | 1,721\n", [cell], results=results)
    )
    assert any(
        "no text" in e
        for e in _numbers(tmp_path / "c", "| A | 1,721 | x |\n", [cell], results=results)
    )


def test_a_document_two_levels_below_decisions_is_not_read(tmp_path):
    files = {"docs/decisions/deep/0001-a.md": "[x](gone.md)\n", "docs/decisions/0002-b.md": "# B\n"}
    assert _errors(tmp_path, files) == []


def test_a_csv_key_value_may_contain_a_colon(tmp_path):
    results = {"t.csv": "a,v\nx:y,7\nz,8\n"}
    cells = [("README.md", r"is (\d+)", "t.csv", "a=x:y:v", "count")]
    assert _numbers(tmp_path, "is 7\n", cells, results=results) == []


def test_a_csv_key_value_may_contain_an_equals_sign(tmp_path):
    results = {"t.csv": "a,v\nx=y,7\nz,8\n"}
    cells = [("README.md", r"is (\d+)", "t.csv", "a=x=y:v", "count")]
    assert _numbers(tmp_path, "is 7\n", cells, results=results) == []


def test_a_number_cut_off_by_a_longer_thousands_group_is_read_whole(tmp_path):
    errors = _numbers(tmp_path, "read 1,2345\n", [])
    assert len(errors) == 1


def test_a_wrong_first_match_is_reported_when_the_last_is_right(tmp_path):
    text = "right on 80.0%\n\nright on 83.7%\n"
    errors = _numbers(tmp_path, text, [RIGHT_], results={"m.json": ACCURACY})
    assert len(errors) == 1 and "README.md:1" in errors[0]


def test_non_ascii_text_is_read_as_utf8_in_the_document_and_in_results(tmp_path):
    results = {"m.json": '{"Á": 5}', "t.csv": "Á,v\nx,7\n"}
    cells = [
        ("README.md", r"json (\d+)", "m.json", "Á", "count"),
        ("README.md", r"csv (\d+)", "t.csv", "Á=x:v", "count"),
    ]
    assert _numbers(tmp_path, "Álvaro json 5 csv 7\n", cells, results=results) == []
    assert len(_numbers(tmp_path / "b", "Álvaro 12\n", [])) == 1


def test_main_lists_link_errors_before_number_errors(tmp_path, capsys):
    root = _repo(tmp_path, {"README.md": "12 [a](gone.md)\n", "docs/model_card.md": "x\n"})
    _load().main(root)
    lines = capsys.readouterr().out.splitlines()
    assert "gone.md" in lines[0]
    assert any("number 12" in line for line in lines[1:])


def test_main_defaults_to_the_repository_whatever_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _load().main() == 0


def test_a_table_cell_allows_at_most_two_bold_marks_and_only_digits_in_the_value(tmp_path):
    module = _load()
    cells = [tuple(c) for c in module.interval("README.md", "A", 0, "m.json", "a", "pct1")]
    results = {"m.json": '{"a": {"value": 0.837, "ci_low": 0.835, "ci_high": 0.839}}'}
    for bad in ("| A | ***83.7 [83.5, 83.9] |\n", "| A | x83.7 [83.5, 83.9] |\n"):
        errors = _numbers(tmp_path / str(abs(hash(bad))), bad, cells, results=results)
        assert sum("no text matches" in e for e in errors) == 3, bad


def test_a_cell_without_an_interval_matches_none_of_the_three_numbers(tmp_path):
    module = _load()
    cells = [tuple(c) for c in module.interval("README.md", "A", 0, "m.json", "a", "pct1")]
    results = {"m.json": '{"a": {"value": 0.837, "ci_low": 0.835, "ci_high": 0.839}}'}
    errors = _numbers(tmp_path, "| A | 83.7 |\n", cells, results=results)
    assert sum("no text matches" in e for e in errors) == 3


def test_a_link_through_a_file_is_reported_not_a_crash(tmp_path):
    errors = _errors(tmp_path, {"README.md": "[a](README.md/x)\n"})
    assert len(errors) == 1 and "README.md/x" in errors[0]


def test_a_json_path_through_a_number_is_an_error_line_not_a_crash(tmp_path):
    cells = [("README.md", r"is (\d+)", "m.json", "n.deeper", "count")]
    errors = _numbers(tmp_path, "is 1\n", cells, results={"m.json": ACCURACY})
    assert len(errors) == 1 and "n.deeper" in errors[0]


REPORT = "report/main.tex"


def _report(tmp_path, body, cells=(), allowed=(), results=None, preamble=""):
    """Run check_numbers on a LaTeX report holding `body` between begin and end document."""
    module = _load()
    tex = f"{preamble}\\begin{{document}}\n{body}\\end{{document}}\n"
    files = {REPORT: tex}
    files.update({f"results/{name}": content for name, content in (results or {}).items()})
    root = _repo(tmp_path, files)
    table = [module.Cell(*cell) for cell in cells]
    allow = [module.Allowed(*entry) for entry in allowed]
    return module.check_numbers(root, table, allow, (REPORT,))


def _tex_cell(pattern, path, fmt):
    return (REPORT, pattern, "m.json", path, fmt)


def _numbers_of(errors):
    return [e.split(" number ")[1].split(" ")[0] for e in errors]


GAPS = '{"gap": {"value": -0.7712, "low": -1.5432, "high": 0.0801}, "r": 0.8512, "lr": 3e-05}'


def test_dollar_signs_around_a_signed_number_are_removed_before_matching(tmp_path):
    cells = [_tex_cell(r"gap (-[\d.]+) here", "gap.value", "dec2")]
    assert _report(tmp_path, "gap $-0.77$ here\n", cells, results={"m.json": GAPS}) == []
    errors = _report(tmp_path / "x", "gap $-0.78$ here\n", cells, results={"m.json": GAPS})
    assert errors == [f"{REPORT}:2: -0.78 in the text, -0.77 in m.json gap.value"]


def test_an_escaped_percent_is_a_percent_sign(tmp_path):
    cells = [_tex_cell(r"right on ([\d.]+)%", "r", "pct1")]
    body = "right on 85.1\\% of bios\n"
    assert _report(tmp_path, body, cells, results={"m.json": GAPS}) == []


def test_an_interval_in_math_mode_matches_both_ends(tmp_path):
    cells = [
        _tex_cell(r"\[(-[\d.]+),", "gap.low", "dec2"),
        _tex_cell(r", (\+[\d.]+)\]", "gap.high", "signed2"),
    ]
    body = "($-0.77$ [$-1.54$, $+0.08$])\n"
    errors = _report(tmp_path, body, cells, [(None, r"-0\.77", "the value")], {"m.json": GAPS})
    assert errors == []


def test_a_correlation_written_in_math_mode_is_read(tmp_path):
    cells = [_tex_cell(r"r = ([\d.]+)", "r", "dec2")]
    assert _report(tmp_path, "($r = 0.85$)\n", cells, results={"m.json": GAPS}) == []


def test_a_power_of_ten_is_read_as_the_written_exponent_form(tmp_path):
    cells = [_tex_cell(r"rate (3e-5)", "lr", "learning_rate")]
    body = "rate $3\\times10^{-5}$ chosen\n"
    assert _report(tmp_path, body, cells, results={"m.json": GAPS}) == []


def test_a_comment_is_not_read_but_an_escaped_percent_does_not_start_one(tmp_path):
    errors = _report(tmp_path, "% 99 bios\nof 12\\% and 13 % 14 trailing\n")
    assert _numbers_of(errors) == ["12", "13"]


def test_the_preamble_is_not_read(tmp_path):
    assert _report(tmp_path, "text\n", preamble="\\documentclass[11pt]{article}\n") == []


def test_citation_keys_references_labels_files_and_links_are_not_read(tmp_path):
    body = (
        "\\citep[Annex III]{hardt2016equality} \\citet{dearteaga2019bias} \\fig{f1} "
        "Table~\\ref{tab:methods}\n"
        "\\label{fig:f2}\\includegraphics[width=0.5\\columnwidth]{f2_x.pdf} \\url{https://x/2025}\n"
    )
    assert _report(tmp_path, body) == []


def test_a_table_column_specification_is_not_read(tmp_path):
    body = "\\begin{tabular}{@{}p{0.27\\columnwidth}p{0.66\\columnwidth}@{}}\n"
    assert _report(tmp_path, body) == []


def test_commands_and_braces_are_removed_but_their_text_is_kept(tmp_path):
    body = "\\paragraph{Data.} the \\emph{showcase} of \\texttt{[UNK]} and 7\\\\ 8~9 vs.\\ 10\n"
    assert _numbers_of(_report(tmp_path, body)) == ["7", "8", "9", "10"]


def test_a_percent_in_a_markdown_document_does_not_start_a_comment(tmp_path):
    assert _numbers_of(_numbers(tmp_path, "50 % 7 left\n", [])) == ["50", "7"]


def test_the_formats_the_report_uses_for_log_odds_and_counts_out_of_ten():
    formats = _load().FORMATS
    assert formats["dec1"](6.9826) == "7.0"
    assert formats["dec1"](1.5324) == "1.5"
    assert formats["abs1"](-0.675) == "0.7"
    assert formats["abs1"](0.675) == "0.7"
    assert formats["abs2"](-1.2415) == "1.24"
    assert formats["signed1"](3.2147) == "+3.2"
    assert formats["signed1"](-0.65) == "-0.7"
    assert formats["signed2"](0.0801) == "+0.08"
    assert formats["signed2"](-0.7744) == "-0.77"
    assert formats["of10"](0.8) == "8"
    assert formats["hours1"](10773.6) == "3.0"
    assert formats["hours1"](5400) == "1.5"
