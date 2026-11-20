import importlib.util
import json
from pathlib import Path

from test_ine_names import _files

from hiring_bias_xai import first_names, ine_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "llm_names.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("llm_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    sheets = _files()
    seen = []

    def fake_download(url, sha256, target, **kwargs):
        seen.append((url, sha256, target))
        content = next(key for key, (u, _) in ine_names.FILES.items() if u == url).encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return content

    monkeypatch.setattr(first_names, "download", fake_download)
    monkeypatch.setattr(ine_names, "read_sheets", lambda content: sheets[content.decode()])
    results_dir = tmp_path / "results"
    _load_script().main(results_dir)
    return seen, json.loads((results_dir / "llm" / "names.json").read_text(encoding="utf-8"))


def test_the_script_downloads_the_four_pinned_https_files_into_the_artifacts(monkeypatch, tmp_path):
    seen, _ = _run(monkeypatch, tmp_path)
    assert [(url, sha) for url, sha, _ in seen] == list(ine_names.FILES.values())
    assert all(url.startswith("https://www.ine.es/") for url, _ in ine_names.FILES.values())
    assert [target for _, _, target in seen] == [
        tmp_path / "artifacts" / "ine" / Path(url).name for url, _ in ine_names.FILES.values()
    ]


def test_names_json_holds_the_groups_the_source_the_licence_and_the_dropped_surnames(
    monkeypatch, tmp_path
):
    _, record = _run(monkeypatch, tmp_path)
    assert sorted(record["groups"]) == sorted(ine_names.GROUPS)
    assert record["groups"]["Romania"]["surnames"] == ["Popescu", "Ionescu", "Popa"]
    assert record["dropped_surnames"] == ["Rodriguez", "Perez", "Gomez", "Gonzalez"]
    assert record["n_dropped_surnames"] == 4
    source = record["source"]
    assert source["licence"] == "CC BY 4.0"
    assert source["attribution"] == (
        "Elaboración propia con datos extraídos del sitio web del INE: www.ine.es"
    )
    assert source["files"]["first_names"] == {
        "url": ine_names.FILES["first_names"][0],
        "sha256": ine_names.FILES["first_names"][1],
        "date": "01/01/2022",
    }
    assert source["files"]["surnames"]["date"] == "01/01/2025"
    assert record["shared_surnames"] == {}
