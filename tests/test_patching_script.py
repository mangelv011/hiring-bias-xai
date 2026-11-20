import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hiring_bias_xai import counterfactual, data, patching
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patching.py"
SWAPPED = {"alpha": "zeta", "zeta": "alpha", "beta": "eta", "eta": "beta", "gamma": "gamma delta"}


def _swap(text):
    return " ".join(SWAPPED.get(word, word) for word in text.split())


def _remove(text):
    return " ".join(word for word in text.split() if word != "theta")


def _bios():
    """40 bios. Every fourth has nothing to swap; every fifth has a "gamma", whose swap adds a
    token, so its pair is dropped; two have a "theta", which the removal control deletes."""
    rng = np.random.default_rng(0)
    swappable = ["alpha", "beta", "zeta", "eta"]
    rows = []
    for i in range(40):
        words = [str(rng.choice(["delta", "epsilon", "theta"] if i % 4 == 0 else swappable))]
        words += [str(rng.choice(["delta", "epsilon"])) for _ in range(2)]
        if i % 5 == 0:
            words.append("gamma")
        if i in (1, 2):
            words.append("theta")
        rng.shuffle(words)
        rows.append((f"test-{i}", " ".join(words), "nurse" if i % 2 else "surgeon", "female"))
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


@pytest.fixture
def run(monkeypatch, tmp_path, tiny_network, tiny_tokenizer):
    """Return go(): the script's main() on 40 synthetic bios with a tiny network; gives the
    results directory. `go.model` is the network and `go.bios` the bios."""
    results = tmp_path / "results"
    results.mkdir()
    bios = _bios()
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: bios)
    monkeypatch.setattr(counterfactual, "swap_gender", _swap)
    monkeypatch.setattr(data, "scrub", _remove)
    spec = importlib.util.spec_from_file_location("patching_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_BOOT = 50
    model = DebertaModel(tiny_network(spread=3.0), tiny_tokenizer(64), batch_size=7)
    monkeypatch.setattr(module, "_load", lambda: model)

    def go():
        module.main(results)
        return results

    go.model, go.bios, go.module = model, bios, module
    return go


def _expected_selection(model, bios):
    target = np.array([model.labels.index(o) for o in bios["occupation"]])
    texts = list(bios["text"])

    def lo(items):
        return patching.true_class_log_odds(model.logits(items), target)

    original = lo(texts)
    swap_change = lo([_swap(t) for t in texts]) - original
    removal_change = lo([_remove(t) for t in texts]) - original
    eligible, bar = patching.eligible(swap_change, removal_change)
    same = patching.equal_length(model.tokenizer, texts, [_swap(t) for t in texts])
    return eligible, same, bar


def test_the_csv_has_a_row_per_layer_and_position_set_with_the_summary_columns(run):
    table = pd.read_csv(run() / "patching.csv")
    assert list(table.columns) == [
        "layer", "position_set", "n", "mean", "mean_ci_low", "mean_ci_high",
        "median", "median_ci_low", "median_ci_high",
    ]  # fmt: skip
    assert len(table) == 3 * 2
    assert sorted(table["layer"].unique()) == [0, 1, 2]
    assert set(table["position_set"]) == {"differing", "all"}


def test_only_eligible_pairs_of_equal_token_length_are_patched(run):
    results = run()
    eligible, same, bar = _expected_selection(run.model, run.bios)
    table = pd.read_csv(results / "patching.csv")
    assert (table["n"] == (eligible & same).sum()).all()
    record = json.loads((results / "patching.json").read_text())
    assert record["n_bios"] == 40
    assert record["n_eligible"] == eligible.sum()
    assert record["bar"] == pytest.approx(bar)
    assert record["n_unequal_length"] == (~same).sum()
    assert record["n_unequal_length_eligible"] == (eligible & ~same).sum()
    assert record["n_patched"] == (eligible & same).sum()
    assert 0 < record["n_patched"] < record["n_eligible"] < 40  # the fixture exercises each rule


def test_all_positions_restore_everything_at_every_layer_and_the_checks_are_recorded(run):
    results = run()
    table = pd.read_csv(results / "patching.csv")
    everywhere = table[table["position_set"] == "all"]
    np.testing.assert_allclose(everywhere["mean"], 1.0, atol=1e-4)
    np.testing.assert_allclose(everywhere["median"], 1.0, atol=1e-4)
    record = json.loads((results / "patching.json").read_text())
    assert record["max_all_positions_error"] < 1e-4


def test_the_differing_positions_restore_everything_at_the_embeddings_and_nothing_at_the_end(run):
    table = pd.read_csv(run() / "patching.csv")
    differing = table[table["position_set"] == "differing"].set_index("layer")
    assert differing.loc[0, "mean"] == pytest.approx(1.0, abs=1e-4)
    assert differing.loc[2, "mean"] == pytest.approx(0.0, abs=1e-4)


def test_the_summary_is_the_bootstrap_summary_of_the_per_bio_shares(run):
    results = run()
    eligible, same, _ = _expected_selection(run.model, run.bios)
    keep = eligible & same
    bios = run.bios[keep]
    texts, swapped = list(bios["text"]), [_swap(t) for t in bios["text"]]
    target = np.array([run.model.labels.index(o) for o in bios["occupation"]])
    out = patching.patch_pairs(run.model, texts, swapped, target)
    shares = patching.restored_share(out["clean"], out["corrupted"], out["patched"][:, 1, 0])
    row = pd.read_csv(results / "patching.csv").query("layer == 1 and position_set == 'differing'")
    expected = patching.summarise(shares, 50)
    for name in ("n", "mean", "mean_ci_low", "mean_ci_high", "median", "median_ci_low"):
        assert row[name].iloc[0] == pytest.approx(expected[name], abs=1e-4)


def test_the_json_records_the_settings(run):
    record = json.loads((run() / "patching.json").read_text())
    assert record["settings"] == {
        "n_boot": 50,
        "seed": 42,
        "quantile": 0.95,
        "batch_size": run.module.BATCH_SIZE,
    }
    assert record["seconds"] >= 0


def test_the_script_stops_if_patching_all_positions_does_not_restore_the_clean_run(
    run, monkeypatch
):
    real = patching.restored_share

    def broken(clean, corrupted, patched):
        return real(clean, corrupted, patched) * 0.5

    monkeypatch.setattr(patching, "restored_share", broken)
    with pytest.raises(RuntimeError, match="all positions"):
        run()
