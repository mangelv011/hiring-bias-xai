import importlib.util
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
    assert (results / "xai_subset_ids.csv").read_bytes() == ("id\n" + "".join(f"{i}\n" for i in ids)).encode()
