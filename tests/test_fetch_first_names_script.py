import importlib.util
import json
from pathlib import Path

import pytest
from test_first_names import _archive

from hiring_bias_xai import first_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_first_names.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_first_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_downloads_parses_and_records_the_list(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    archive = _archive({"yob1990.txt": "Maria,F,20000\nRare,M,3\n"})
    seen = []

    def fake_download(url, sha256, target, **kwargs):
        seen.append((url, sha256))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive)
        return archive

    monkeypatch.setattr(first_names, "download", fake_download)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _load_script().main(results_dir)
    assert seen == [(first_names.URL, first_names.SHA256)]
    assert first_names.load() == frozenset({"Maria"})
    record = json.loads((results_dir / "first_names.json").read_text())
    assert record["n_names"] == 1
    assert record["url"] == first_names.URL
    assert record["sha256"] == first_names.SHA256
    assert record["min_count"] == first_names.MIN_COUNT


def test_loading_before_the_download_says_which_script_to_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    with pytest.raises(FileNotFoundError, match="fetch_first_names.py"):
        first_names.load()
