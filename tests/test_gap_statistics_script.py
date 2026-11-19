import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hiring_bias_xai import data, fairness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gap_statistics.py"
OCCUPATIONS = ["a", "b", "c", "d"]


def _split(name, n_per_cell, seed):
    rng = np.random.default_rng(seed)
    rows = [
        (f"{name}-{i}", "text", o, g)
        for i, (o, g) in enumerate(
            (o, g) for o in OCCUPATIONS for g in ("female", "male") for _ in range(n_per_cell)
        )
    ]
    frame = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    # Women are 20% of "a" and 90% of "d" in the training subsample.
    share = {"a": 0.2, "b": 0.4, "c": 0.7, "d": 0.9}
    if name == "train":
        keep = [
            rng.random() < (share[o] if g == "female" else 1 - share[o])
            for o, g in zip(frame["occupation"], frame["gender"], strict=True)
        ]
        frame = frame[keep].reset_index(drop=True)
    return frame


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return run(): the script's main() on synthetic splits and predictions, giving the output."""
    splits = {"train": _split("train", 60, 0), "test": _split("test", 40, 1)}
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": splits[split])
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    spec = importlib.util.spec_from_file_location("gap_statistics_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_BOOT = module.N_PERM = 30
    test = splits["test"]
    share = (splits["train"]["gender"] == "female").groupby(splits["train"]["occupation"]).mean()
    rng = np.random.default_rng(2)
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "artifacts" / "predictions").mkdir(parents=True)
    for file, hit_women, hit_men in (
        ("baseline", 0.6, 0.8),
        ("deberta", 0.7, 0.9),
    ):
        hit = np.where(
            test["gender"] == "female",
            rng.random(len(test)) < hit_women,
            rng.random(len(test)) < hit_men,
        )
        predicted = np.where(hit, test["occupation"], "other")
        pd.DataFrame({"id": test["id"], "predicted_occupation": predicted}).to_csv(
            tmp_path / "artifacts" / "predictions" / f"{file}_test.csv", index=False
        )
        report = fairness.evaluate(
            test["occupation"], predicted, test["gender"], share.to_dict(), n_boot=2
        )
        (results / f"metrics_{file}.json").write_text(json.dumps({"test": report}))

    def main():
        module.main(results)
        return json.loads((results / "gap_statistics.json").read_text())

    main.results = results
    return main


def test_the_script_writes_gap_statistics_for_both_models(run):
    out = run()
    assert sorted(out) == ["baseline_model", "main_model"]
    for statistics in out.values():
        assert statistics["n_occupations"] == 4
        assert sorted(statistics["occupations"]) == OCCUPATIONS
        assert statistics["rms_tpr_gap"]["n_permutations"] == 30
    # The shares are the training subsample's, not the test split's (which is 50% everywhere).
    shares = {o: r["share_women"] for o, r in out["main_model"]["occupations"].items()}
    assert shares["a"] < 0.4 < 0.6 < shares["d"]


def test_the_script_stops_when_its_point_values_disagree_with_the_metrics_file(run):
    path = run.results / "metrics_deberta.json"
    run()
    metrics = json.loads(path.read_text())
    metrics["test"]["rms_tpr_gap"]["value"] += 0.01
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="rms_tpr_gap"):
        run()
