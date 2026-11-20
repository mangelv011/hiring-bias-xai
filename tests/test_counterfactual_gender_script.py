import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import data, first_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "counterfactual_gender.py"
OCCUPATIONS = ["surgeon", "software_engineer", "composer", "dietitian", "nurse", "model"]
BIOS = [
    ("She trains her team and he thanks her.", "female"),
    ("He leads his team and she thanks him. His mother is proud.", "male"),
    ("Maria says she writes. Mr. Lee knows her.", "female"),
    ("Nothing here names a gender at all.", "male"),
]


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(): the script's main() on 24 synthetic bios and fake screeners; gives (script,
    results dir)."""
    rows = []
    for occupation in OCCUPATIONS:
        for text, gender in BIOS:
            rows.append((f"test-{len(rows)}", text, occupation, gender))
    subset = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    train = pd.DataFrame(
        {
            "occupation": [o for o in data.OCCUPATIONS for _ in range(2)],
            "gender": ["female", "male"] * len(data.OCCUPATIONS),
        }
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_text("id\n" + "\n".join(subset["id"]) + "\n")
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: subset)
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": train)
    monkeypatch.setattr(first_names, "load", lambda: frozenset({"Maria"}))
    spec = importlib.util.spec_from_file_location("counterfactual_gender", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_load", lambda name: Hashed())
    monkeypatch.setattr(module, "N_BOOT", 20)

    def go():
        module.main(results_dir)
        return module, results_dir

    return go


def test_the_csv_has_a_row_per_screener_occupation_subset_and_condition(run):
    _, results_dir = run()
    table = pd.read_csv(results_dir / "counterfactual_gender.csv")
    assert len(table) == 2 * (len(OCCUPATIONS) + 1) * 2 * 2
    assert set(table["screener"]) == {"deberta", "baseline"}
    pooled = table[(table["occupation"] == "pooled") & (table["subset"] == "all")]
    assert set(pooled["n"]) == {len(OCCUPATIONS) * len(BIOS)}
    unflagged = table[(table["occupation"] == "pooled") & (table["subset"] == "unflagged")]
    assert set(unflagged["n"]) == {len(OCCUPATIONS) * 2}  # the 1st and 4th bios have no cue


def test_the_json_counts_the_flags_and_the_edits(run):
    _, results_dir = run()
    record = json.loads((results_dir / "counterfactual_gender.json").read_text())
    flags = record["flags"]
    assert flags["n_bios"] == 24
    assert flags["n_noun"] == 6  # "mother" in the 2nd bio
    assert flags["n_name"] == 6  # "Maria" in the 3rd
    assert flags["n_flagged"] == 12
    assert flags["n_distinct_cues"] == 2
    assert flags["top_cues"] == {"Maria": 6, "mother": 6}
    assert record["edits"] == {"swap_changed": 18, "removal_changed": 18}
    assert "hand_check" not in record
    assert record["direction_basis"].startswith("training share of women")


def test_a_committed_hand_check_is_summarised_in_the_json(run):
    module, results_dir = run()
    checks = pd.DataFrame(
        {"id": ["test-0", "test-1"], "coherent": ["yes", "no"], "reason": ["", "ambiguous her"]}
    )
    checks.to_csv(results_dir / "counterfactual_hand_check.csv", index=False)
    module.main(results_dir)
    record = json.loads((results_dir / "counterfactual_gender.json").read_text())
    assert record["hand_check"]["n"] == 2
    assert record["hand_check"]["n_incoherent"] == 1
