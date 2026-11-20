import importlib.util
from pathlib import Path

import pandas as pd

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "counterfactual_hand_check.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("counterfactual_hand_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_sentences_the_swap_changed_are_shown():
    script = _load_script()
    original = "Lee is a nurse. She loves her job! Nothing else changes. He agrees."
    swapped = "Lee is a nurse. He loves his job! Nothing else changes. She agrees."
    assert script.changed_sentences(original, swapped) == [
        ("She loves her job!", "He loves his job!"),
        ("He agrees.", "She agrees."),
    ]


def test_the_draw_skips_bios_the_swap_leaves_alone_and_is_in_subset_order(monkeypatch, tmp_path):
    texts = ["She runs."] * 30 + ["Nothing here."] * 30
    subset = pd.DataFrame(
        {
            "id": [f"test-{i}" for i in range(60)],
            "text": texts,
            "occupation": "nurse",
            "gender": "female",
        }
    )
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: subset)
    script = _load_script()
    drawn = script.draw(tmp_path / "ids.csv", n=10)
    assert len(drawn) == 10
    assert [bio_id for bio_id, _ in drawn] == sorted(
        (bio_id for bio_id, _ in drawn), key=lambda bio_id: int(bio_id.split("-")[1])
    )
    assert all(int(bio_id.split("-")[1]) < 30 for bio_id, _ in drawn)
    assert all(pairs == [("She runs.", "He runs.")] for _, pairs in drawn)
