import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "shapley_run.py"

BIOS = [
    ("test-0", "alpha beta gamma delta", "nurse"),
    ("test-1", "alpha beta", "surgeon"),
    ("test-2", "gamma delta epsilon zeta eta", "nurse"),
    ("test-3", "theta", "dentist"),
]
PILOT = {
    "chosen": {"method": "shapley", "removal": "unk", "n_permutations": 3},
    "inference_batch_size": 16,
}


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return run(): the script's main() on four synthetic bios and fake classifiers.

    `run.fakes` holds the classifier of each screener (their `calls` show what was scored) and
    `run.results` is the results directory. A new `run.pilot` can be set before calling.
    """
    results = tmp_path / "results"
    results.mkdir()
    ids = "".join(f"{bio_id}\n" for bio_id, _, _ in BIOS)
    (results / "showcase_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    test = pd.DataFrame(BIOS, columns=["id", "text", "occupation"]).assign(gender="female")
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": test)
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("shapley_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fakes = {"deberta": Hashed(), "baseline": Hashed()}
    loads = []
    monkeypatch.setattr(
        module, "_load", lambda name, batch_size: (loads.append((name, batch_size)), fakes[name])[1]
    )

    def go():
        (results / "shapley_pilot.json").write_text(json.dumps(go.pilot))
        module.main(results)
        return results

    go.pilot = PILOT
    go.fakes = fakes
    go.loads = loads
    go.results = results
    go.module = module
    return go


def _saved(name):
    return {bio_id: attribution.load(attribution.path(name, bio_id)) for bio_id, _, _ in BIOS}


def test_each_screener_gets_one_attribution_file_per_bio_with_the_chosen_settings(run):
    run()
    for name in ("deberta", "baseline"):
        for bio_id, text, occupation in BIOS:
            saved = attribution.load(attribution.path(name, bio_id))
            assert saved.words == text.split()
            assert saved.target == data.OCCUPATIONS.index(occupation)
            assert saved.removal == "unk"
            assert saved.orders.shape == (6, len(text.split()))


def test_the_models_are_loaded_with_the_inference_batch_size_of_the_pilot(run):
    run()
    assert run.loads == [("deberta", 16), ("baseline", 16)]


def test_the_summary_has_per_screener_counts_cost_and_the_efficiency_error(run):
    summary = json.loads((run() / "shapley_summary.json").read_text())
    assert summary["settings"] == {
        "removal": "unk",
        "n_permutations": 3,
        "target": "true occupation",
        "seed": 42,
    }
    for name in ("deberta", "baseline"):
        entry = summary["screeners"][name]
        saved = _saved(name)
        assert entry["n_bios"] == 4
        assert entry["mean_words"] == pytest.approx(np.mean([4, 2, 5, 1]))
        assert entry["mean_evaluations"] == pytest.approx(
            np.mean([len(item.logits) for item in saved.values()])
        )
        seconds = sum(item.seconds for item in saved.values())
        assert entry["total_seconds"] == pytest.approx(seconds, abs=0.05)  # rounded to 0.1 s
        assert 0 <= entry["efficiency_max_error"] < 1e-9


def test_a_second_run_scores_nothing_and_leaves_the_files_alone(run):
    run()
    before = {name: _saved(name) for name in ("deberta", "baseline")}
    for fake in run.fakes.values():
        fake.calls.clear()
    run()
    assert [fake.calls for fake in run.fakes.values()] == [[], []]
    after = {name: _saved(name) for name in ("deberta", "baseline")}
    for name in before:
        for bio_id in before[name]:
            np.testing.assert_array_equal(after[name][bio_id].logits, before[name][bio_id].logits)


def test_a_run_that_stopped_halfway_scores_only_the_bios_without_a_file(run):
    run()
    attribution.path("deberta", "test-1").unlink()
    attribution.path("deberta", "test-3").unlink()
    for fake in run.fakes.values():
        fake.calls.clear()
    run()
    assert len(run.fakes["deberta"].calls) == 2
    assert run.fakes["baseline"].calls == []
    assert attribution.load(attribution.path("deberta", "test-1")).words == ["alpha", "beta"]


def test_the_values_do_not_depend_on_where_a_run_resumed(run):
    run()
    complete = _saved("deberta")
    attribution.path("deberta", "test-2").unlink()
    run()
    resumed = _saved("deberta")["test-2"]
    np.testing.assert_array_equal(resumed.orders, complete["test-2"].orders)
    np.testing.assert_array_equal(resumed.logits, complete["test-2"].logits)


def test_bios_get_different_orders_but_the_same_ones_on_every_run(run):
    run()
    first = _saved("deberta")
    assert not np.array_equal(first["test-0"].orders[:, :2], first["test-2"].orders[:, :2])
    for name in ("deberta", "baseline"):
        attribution.path(name, "test-0").unlink()
    run()
    np.testing.assert_array_equal(_saved("deberta")["test-0"].orders, first["test-0"].orders)


def test_a_damaged_file_is_scored_again(run):
    run()
    attribution.path("deberta", "test-0").write_bytes(b"not an npz")
    run.fakes["deberta"].calls.clear()
    run()
    assert len(run.fakes["deberta"].calls) == 1
    assert attribution.load(attribution.path("deberta", "test-0")).words[0] == "alpha"


@pytest.mark.parametrize(
    "changed",
    [{"removal": "delete"}, {"n_permutations": 5}],
)
def test_a_file_made_with_other_settings_is_scored_again(run, changed):
    run()
    run.pilot = {**PILOT, "chosen": {**PILOT["chosen"], **changed}}
    for fake in run.fakes.values():
        fake.calls.clear()
    run()
    assert [len(fake.calls) for fake in run.fakes.values()] == [4, 4]
    saved = attribution.load(attribution.path("deberta", "test-0"))
    assert saved.removal == run.pilot["chosen"]["removal"]
    assert saved.orders.shape[0] == 2 * run.pilot["chosen"]["n_permutations"]


def test_a_file_of_another_bio_target_is_scored_again(run):
    run()
    other = attribution.load(attribution.path("deberta", "test-1"))
    attribution.save(attribution.path("deberta", "test-0"), other)  # wrong words and target
    run.fakes["deberta"].calls.clear()
    run()
    assert len(run.fakes["deberta"].calls) == 1
    assert attribution.load(attribution.path("deberta", "test-0")).words[0] == "alpha"


def test_the_run_stops_when_the_pilot_chose_the_fallback(run):
    run.pilot = {**PILOT, "chosen": {"method": "leave_one_out", "removal": "unk"}}
    with pytest.raises(SystemExit, match="leave-one-out"):
        run()
    assert run.fakes["deberta"].calls == []


def test_the_run_stops_without_a_pilot_file(run, monkeypatch):
    (run.results / "shapley_pilot.json").unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="shapley_pilot.py"):
        run.module.main(run.results)
