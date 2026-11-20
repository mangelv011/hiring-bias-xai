import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stability_runs.py"
PILOT = {
    "chosen": {"method": "shapley", "removal": "delete", "n_permutations": 3},
    "inference_batch_size": 16,
}
# Two occupations by two genders, three bios each: the sample of 4 takes 1 from each group.
BIOS = [
    (f"test-{i}", f"alpha beta gamma delta{i}", occupation)
    for i, occupation in enumerate(["nurse", "nurse", "nurse", "surgeon", "surgeon", "surgeon"] * 2)
]
GENDERS = ["female", "male"] * 6


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(*variants): the script's main() on twelve synthetic bios and a fake classifier.

    `go.fake` scores everything; `go.module` is the script; `go.results` the results directory.
    """
    results = tmp_path / "results"
    results.mkdir()
    ids = "".join(f"{bio_id}\n" for bio_id, _, _ in BIOS)
    (results / "showcase_subset_ids.csv").write_bytes(f"id\n{ids}".encode())
    (results / "shapley_pilot.json").write_text(json.dumps(PILOT))
    test = pd.DataFrame(BIOS, columns=["id", "text", "occupation"]).assign(gender=GENDERS)
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": test)
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("stability_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SAMPLE_SIZE = 4
    fake = Hashed()
    loads = []
    monkeypatch.setattr(module, "_load", lambda batch_size: (loads.append(batch_size), fake)[1])

    def go(*variants):
        module.main(results, *([variants] if variants else []))
        return results

    go.fake = fake
    go.loads = loads
    go.module = module
    go.results = results
    return go


def _sample_ids(results):
    return pd.read_csv(results / "stability_sample_ids.csv")["id"].tolist()


def test_the_sample_ids_are_saved_and_come_from_the_subset_one_per_group(run):
    ids = _sample_ids(run("deberta-seed7"))
    assert len(ids) == 4
    table = pd.DataFrame(BIOS, columns=["id", "text", "occupation"]).assign(gender=GENDERS)
    chosen = table[table["id"].isin(ids)]
    assert chosen.groupby(["occupation", "gender"]).size().tolist() == [1, 1, 1, 1]


def test_each_variant_saves_one_attribution_per_sampled_bio_under_its_own_folder(run):
    ids = _sample_ids(run("deberta-seed7", "deberta-unk"))
    for variant in ("deberta-seed7", "deberta-unk"):
        files = sorted(attribution.path(variant, "test-0").parent.glob("*.npz"))
        assert sorted(file.stem for file in files) == sorted(ids)
    assert not attribution.path("deberta", "test-0").parent.exists()


def test_the_seed_variant_changes_the_orders_and_keeps_delete_the_unk_variant_keeps_the_orders(run):
    ids = _sample_ids(run("deberta-seed7", "deberta-unk"))
    bio_id = ids[0]
    position = [bio[0] for bio in BIOS].index(bio_id)
    seed7 = attribution.load(attribution.path("deberta-seed7", bio_id))
    unk = attribution.load(attribution.path("deberta-unk", bio_id))
    assert (seed7.removal, unk.removal) == ("delete", "unk")
    assert seed7.orders.shape[0] == unk.orders.shape[0] == 6
    # the orders of the main run are those of seed (42, position of the bio in the subset)
    (main,) = attribution.shapley(
        Hashed(), [BIOS[position][1]], [BIOS[position][2]], 3, "delete", seed=(42, position)
    )
    np.testing.assert_array_equal(unk.orders, main.orders)
    assert not np.array_equal(seed7.orders, main.orders)


def test_a_second_run_scores_nothing_and_an_interrupted_run_scores_only_what_is_missing(run):
    ids = _sample_ids(run("deberta-seed7"))
    run.fake.calls.clear()
    run("deberta-seed7")
    assert run.fake.calls == []
    attribution.path("deberta-seed7", ids[0]).unlink()
    run("deberta-seed7")
    assert len(run.fake.calls) == 1


def test_the_model_is_loaded_with_the_inference_batch_size_of_the_pilot(run):
    run("deberta-unk")
    assert run.loads == [16]


def test_the_variants_use_the_permutation_count_of_the_pilot_and_all_variants_run_by_default(run):
    run()
    for variant in ("deberta-seed7", "deberta-unk"):
        saved = attribution.load(attribution.path(variant, _sample_ids(run.results)[0]))
        assert saved.orders.shape[0] == 6


def test_an_unknown_variant_is_an_error(run):
    with pytest.raises(SystemExit, match="deberta-other"):
        run("deberta-other")
