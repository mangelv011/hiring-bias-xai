import pandas as pd
import pytest

from hiring_bias_xai import results


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    return tmp_path / "artifacts"


def _test_split(ids=("test-0", "test-1", "test-2")):
    return pd.DataFrame({"id": list(ids), "occupation": ["nurse", "surgeon", "nurse"][: len(ids)]})


def test_write_json_indents_by_two_sorts_the_keys_and_ends_with_one_unix_newline(tmp_path):
    path = tmp_path / "out.json"
    results.write_json(path, {"b": [1, 2], "a": {"y": 1.5, "x": None}})
    assert path.read_bytes() == (
        b'{\n  "a": {\n    "x": null,\n    "y": 1.5\n  },\n  "b": [\n    1,\n    2\n  ]\n}\n'
    )


def test_write_json_escapes_non_ascii_text(tmp_path):
    path = tmp_path / "out.json"
    results.write_json(path, {"name": "Núñez"})
    assert path.read_bytes() == b'{\n  "name": "N\\u00fa\\u00f1ez"\n}\n'


def test_predictions_are_saved_per_screener_under_the_artifacts_directory(artifacts):
    results.save_predictions("baseline", _test_split(), ["nurse", "surgeon", "nurse"])
    results.save_predictions("deberta", _test_split(), ["surgeon", "surgeon", "nurse"])
    saved = pd.read_csv(artifacts / "predictions" / "baseline_test.csv")
    assert list(saved.columns) == ["id", "predicted_occupation"]
    assert saved["id"].tolist() == ["test-0", "test-1", "test-2"]
    assert saved["predicted_occupation"].tolist() == ["nurse", "surgeon", "nurse"]
    other = pd.read_csv(artifacts / "predictions" / "deberta_test.csv")
    assert other["predicted_occupation"].tolist() == ["surgeon", "surgeon", "nurse"]


def test_saving_again_replaces_the_predictions_of_that_screener(artifacts):
    results.save_predictions("baseline", _test_split(), ["nurse", "nurse", "nurse"])
    results.save_predictions("baseline", _test_split(), ["surgeon", "surgeon", "surgeon"])
    saved = pd.read_csv(artifacts / "predictions" / "baseline_test.csv")
    assert saved["predicted_occupation"].tolist() == ["surgeon"] * 3


def test_loaded_predictions_follow_the_order_and_index_of_the_test_frame(artifacts):
    results.save_predictions("deberta", _test_split(), ["nurse", "surgeon", "dentist"])
    test = _test_split(["test-2", "test-0"])
    test.index = [7, 3]
    loaded = results.load_predictions("deberta", test)
    assert loaded.tolist() == ["dentist", "nurse"]
    assert loaded.index.tolist() == [7, 3]
    assert loaded.name == "predicted_occupation"


def test_loading_ignores_predictions_for_bios_outside_the_test_frame(artifacts):
    results.save_predictions("deberta", _test_split(), ["nurse", "surgeon", "dentist"])
    loaded = results.load_predictions("deberta", _test_split(["test-1"]))
    assert loaded.tolist() == ["surgeon"]


def test_loading_stops_when_a_test_bio_has_no_prediction(artifacts):
    results.save_predictions("deberta", _test_split(["test-0", "test-1"]), ["nurse", "surgeon"])
    with pytest.raises(ValueError, match="deberta: some test bios have no prediction"):
        results.load_predictions("deberta", _test_split())


def test_loading_stops_when_a_bio_has_two_predictions(artifacts):
    results.save_predictions(
        "deberta", _test_split(["test-0", "test-1", "test-1"]), ["nurse", "surgeon", "nurse"]
    )
    with pytest.raises(pd.errors.MergeError):
        results.load_predictions("deberta", _test_split(["test-0", "test-1"]))


def test_loading_stops_when_a_prediction_is_blank(artifacts):
    path = artifacts / "predictions"
    path.mkdir(parents=True)
    (path / "deberta_test.csv").write_text("id,predicted_occupation\ntest-0,nurse\ntest-1,\n")
    with pytest.raises(ValueError, match="no prediction"):
        results.load_predictions("deberta", _test_split(["test-0", "test-1"]))
