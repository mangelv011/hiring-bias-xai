import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import FakeClassifier, Hashed

from hiring_bias_xai import data, first_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "counterfactual_gender.py"
OCCUPATIONS = ["surgeon", "software_engineer", "composer", "dietitian", "nurse", "model"]
BIOS = [
    ("She trains her team and he thanks her.", "female"),
    ("He leads his team and she thanks him. His mother and wife are proud.", "male"),
    ("Maria says she writes. Mr. Lee knows her.", "female"),
    ("Nothing here names a gender at all.", "male"),
    ("He thanks his coach.", "male"),
]


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(): the script's main() on 30 synthetic bios and fake screeners; gives (script,
    results dir)."""
    rows = []
    for occupation in OCCUPATIONS:
        for text, gender in BIOS:
            rows.append((f"test-{len(rows)}", text, occupation, gender))
    subset = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    state = {"subset": subset}
    train = pd.DataFrame(  # half women everywhere, except 75% in nurse and 25% in surgeon
        {
            "occupation": [o for o in data.OCCUPATIONS for _ in range(2)]
            + ["nurse", "nurse", "surgeon", "surgeon"],
            "gender": ["female", "male"] * len(data.OCCUPATIONS)
            + ["female", "female", "male", "male"],
        }
    )
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_text("id\n" + "\n".join(subset["id"]) + "\n")
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: state["subset"])
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": train)
    monkeypatch.setattr(first_names, "load", lambda: frozenset({"Maria"}))
    spec = importlib.util.spec_from_file_location("counterfactual_gender", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "N_BOOT", 20)

    def go(screener=Hashed, bios=None):
        if bios is not None:
            state["subset"] = pd.DataFrame(bios, columns=["id", "text", "occupation", "gender"])
        monkeypatch.setattr(module, "_load", lambda name: screener())
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
    assert set(unflagged["n"]) == {len(OCCUPATIONS) * 3}  # the 1st, 4th and 5th bios have no cue


def test_the_json_counts_the_flags_and_the_edits(run):
    _, results_dir = run()
    record = json.loads((results_dir / "counterfactual_gender.json").read_text())
    flags = record["flags"]
    assert flags["n_bios"] == 30
    assert flags["n_noun"] == 6  # "mother" and "wife" in the 2nd bio count the bio once
    assert flags["n_name"] == 6  # "Maria" in the 3rd
    assert flags["n_flagged"] == 12
    assert flags["by_occupation"] == dict.fromkeys(sorted(OCCUPATIONS), 2)
    assert flags["n_distinct_cues"] == 3
    assert flags["top_cues"] == {"Maria": 6, "mother": 6, "wife": 6}
    assert record["edits"] == {"swap_changed": 24, "removal_changed": 24}
    assert "hand_check" not in record
    assert record["direction_basis"].startswith("training share of women")


class SheMeansNurse(FakeClassifier):
    """Says nurse (logit 5) for a text with the word "she" and surgeon (logit 5) otherwise."""

    def score(self, text):
        logits = np.zeros(len(self.labels))
        nurse = "nurse" if re.search(r"\bshe\b", text.lower()) else "surgeon"
        logits[self.labels.index(nurse)] = 5
        return logits


def test_the_swap_and_the_removal_are_scored_and_compared_to_the_original_on_the_true_occupation(
    run,
):
    # Log-odds of nurse: 5 - ln 27 = 1.70 when the screener says nurse, -ln(e^5 + 26) = -5.16 when
    # it says surgeon, so a flip moves the score by 6.86 in either direction.
    bios = [("test-0", "She runs.", "nurse", "female"), ("test-1", "He runs.", "nurse", "male")]
    _, results_dir = run(SheMeansNurse, bios)
    table = pd.read_csv(results_dir / "counterfactual_gender.csv")
    for screener in ("deberta", "baseline"):
        rows = table[
            (table["screener"] == screener)
            & (table["occupation"] == "nurse")
            & (table["subset"] == "all")
        ].set_index("condition")
        assert rows.loc["swap", "flip_rate"] == 1.0  # she gives he and he gives she
        assert rows.loc["swap", "mean_log_odds_change"] == pytest.approx(0.0, abs=1e-6)
        assert rows.loc["swap", "toward_share"] == 1.0  # both flips follow the new gender
        assert rows.loc["removal", "flip_rate"] == 0.5  # only the bio that said she changes
        assert rows.loc["removal", "mean_log_odds_change"] == pytest.approx(-6.86 / 2, abs=0.01)


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
