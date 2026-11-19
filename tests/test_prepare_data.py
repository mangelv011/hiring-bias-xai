import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def _raw(groups):
    """Raw split from {(profession, gender): size}, groups in blocks."""
    rows = [(f"bio {p} {g}", p, g) for (p, g), size in groups.items() for _ in range(size)]
    return pd.DataFrame(rows, columns=["hard_text", "profession", "gender"])


@pytest.fixture
def prepare(monkeypatch, tmp_path):
    """Run the script's main() on a small synthetic dataset; returns the results directory."""
    raw = {
        "train": _raw({(0, 0): 20, (0, 1): 20, (1, 0): 30, (1, 1): 10}),
        "dev": _raw({(0, 0): 6, (0, 1): 6}),
        "test": _raw({(0, 0): 8, (0, 1): 4}),
    }
    monkeypatch.setattr(data, "_load_raw", lambda split: raw[split])
    monkeypatch.setattr(data, "_SUBSAMPLE_SIZES", {"train": 40, "dev": 6})
    monkeypatch.setattr(data, "_XAI_SUBSET_SIZE", 6)
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run():
        module.main(tmp_path)
        return tmp_path

    return run


def test_prepare_data_writes_the_xai_subset_ids_as_a_single_column_csv_with_lf(prepare):
    results = prepare()
    ids = list(data.xai_subset()["id"])
    assert len(ids) == 6
    expected = "id\n" + "".join(f"{i}\n" for i in ids)
    assert (results / "xai_subset_ids.csv").read_bytes() == expected.encode()


def test_prepare_data_writes_the_summary_counts_and_training_shares_of_women(prepare):
    # Train: 80 bios, n = 40 gives exactly half of every group. Dev: 12 bios, n = 6, half.
    # XAI subset: 12 test bios, n = 6, half. Share of women in the train subsample: accountant
    # 10 / 20, architect 5 / 20.
    results = prepare()
    raw = (results / "data_summary.json").read_bytes()
    assert json.loads(raw) == {
        "bios": {"train": 40, "dev": 6, "test": 12, "xai_subset": 6},
        "counts": {
            "train": {
                "accountant": {"female": 10, "male": 10},
                "architect": {"female": 5, "male": 15},
            },
            "dev": {"accountant": {"female": 3, "male": 3}},
            "test": {"accountant": {"female": 4, "male": 8}},
            "xai_subset": {"accountant": {"female": 2, "male": 4}},
        },
        "share_women_train": {"accountant": 0.5, "architect": 0.25},
    }
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    assert raw.decode() == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
