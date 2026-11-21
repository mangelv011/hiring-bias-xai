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
