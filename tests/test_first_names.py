import hashlib
import io
import zipfile

import pytest

from hiring_bias_xai import first_names


def _archive(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
    return buffer.getvalue()


def test_names_are_summed_over_years_and_sexes_and_kept_above_the_threshold():
    archive = _archive(
        {
            "yob1990.txt": "Maria,F,60\nJordan,F,30\nJordan,M,40\nRare,F,5\n",
            "yob1991.txt": "Maria,F,50\nRare,M,4\n",
            "NationalReadMe.pdf": "Ignored,F,999999\n",
            "notes/yob1992.txt": "Ignored,F,999999\n",
        }
    )
    assert first_names.parse(archive, min_count=70) == ["Jordan", "Maria"]
    assert first_names.parse(archive, min_count=111) == []


def test_an_archive_that_unpacks_to_too_much_is_refused():
    archive = _archive({"yob1990.txt": "Maria,F,60\n" * 100})
    with pytest.raises(ValueError, match="too large"):
        first_names.parse(archive, min_count=1, max_bytes=100)


@pytest.mark.parametrize(
    "line", ["Maria,F,sixty", "Maria,F,", "Maria,X,5", "Mar1a,F,5", "Maria,F,5,6"]
)
def test_a_line_that_is_not_name_sex_count_is_refused(line):
    with pytest.raises(ValueError, match="yob1990.txt"):
        first_names.parse(_archive({"yob1990.txt": line + "\n"}), min_count=1)


CONTENT = b"archive bytes"
URL = "https://example.org/names.zip"


def test_download_writes_the_bytes_when_the_hash_matches(tmp_path):
    target = tmp_path / "names.zip"
    digest = hashlib.sha256(CONTENT).hexdigest()
    assert first_names.download(URL, digest, target, fetch=lambda url: CONTENT) == CONTENT
    assert target.read_bytes() == CONTENT


def test_a_hash_mismatch_stops_the_download_and_leaves_no_file(tmp_path):
    target = tmp_path / "names.zip"
    with pytest.raises(ValueError, match="sha256"):
        first_names.download(URL, "0" * 64, target, fetch=lambda url: CONTENT)
    assert not target.exists()


@pytest.mark.parametrize("url", ["http://example.org/names.zip", "file:///names.zip"])
def test_only_https_urls_are_fetched(tmp_path, url):
    calls = []
    with pytest.raises(ValueError, match="https"):
        first_names.download(url, "0" * 64, tmp_path / "names.zip", fetch=calls.append)
    assert calls == []
