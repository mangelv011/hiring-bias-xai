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


def _script_runner(monkeypatch, results_dir, raw, sizes, xai_size):
    """Return run(): the script's main() on synthetic data, giving the results directory."""
    monkeypatch.setattr(data, "_load_raw", lambda split: raw[split])
    monkeypatch.setattr(data, "_SUBSAMPLE_SIZES", sizes)
    monkeypatch.setattr(data, "_XAI_SUBSET_SIZE", xai_size)
    spec = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run():
        module.main(results_dir)
        return results_dir

    return run


@pytest.fixture
def prepare(monkeypatch, tmp_path):
    raw = {
        "train": _raw({(0, 0): 40, (0, 1): 20, (1, 0): 30, (1, 1): 10}),
        "dev": _raw({(0, 0): 6, (0, 1): 6}),
        "test": _raw({(0, 0): 8, (0, 1): 4}),
    }
    return _script_runner(monkeypatch, tmp_path, raw, {"train": 50, "dev": 6}, 6)


def test_prepare_data_writes_the_xai_subset_ids_as_a_single_column_csv_with_lf(prepare):
    results = prepare()
    ids = list(data.xai_subset()["id"])
    assert len(ids) == 6
    expected = "id\n" + "".join(f"{i}\n" for i in ids)
    assert (results / "xai_subset_ids.csv").read_bytes() == expected.encode()


def test_prepare_data_writes_the_showcase_occupations_file(monkeypatch, tmp_path):
    # Occupations 0 to 5 (accountant ... composer) have a training share of women of
    # (p + 1) / 10 and 500 test bios each, so all six qualify and split 3 and 3. The dev
    # and train frames differ from the test frame, so mixing them up changes the result.
    raw = {
        "train": _raw({(p, g): (p + 1 if g else 9 - p) for p in range(6) for g in (0, 1)}),
        "dev": _raw({(0, 0): 6, (0, 1): 6}),
        "test": _raw({(p, g): 250 for p in range(6) for g in (0, 1)}),
    }
    run = _script_runner(monkeypatch, tmp_path, raw, {"train": 60, "dev": 6}, 30)
    raw_file = (run() / "showcase_occupations.json").read_bytes()
    assert json.loads(raw_file) == {
        "rule": (
            "Among occupations with at least 500 test bios, the three with the lowest and "
            "the three with the highest share of women in the training subsample."
        ),
        "male_dominated": [
            {"occupation": "accountant", "share_women": 0.1},
            {"occupation": "architect", "share_women": 0.2},
            {"occupation": "attorney", "share_women": 0.3},
        ],
        "female_dominated": [
            {"occupation": "composer", "share_women": 0.6},
            {"occupation": "comedian", "share_women": 0.5},
            {"occupation": "chiropractor", "share_women": 0.4},
        ],
    }
    assert raw_file.endswith(b"}\n")
    assert b"\r" not in raw_file


def test_prepare_data_twice_gives_byte_identical_files(prepare):
    results = prepare()
    names = ["xai_subset_ids.csv", "data_summary.json", "showcase_occupations.json"]
    first = {name: (results / name).read_bytes() for name in names}
    prepare()
    assert {name: (results / name).read_bytes() for name in names} == first


def test_prepare_data_writes_the_summary_counts_and_training_shares_of_women(prepare):
    # Train: 100 bios, n = 50 gives exactly half of every group. Dev: 12 bios, n = 6, half.
    # XAI subset: 12 test bios, n = 6, half. Share of women in the train subsample: accountant
    # 10 / 30 = 0.3333, architect 5 / 20.
    results = prepare()
    raw = (results / "data_summary.json").read_bytes()
    assert json.loads(raw) == {
        "bios": {"train": 50, "dev": 6, "test": 12, "xai_subset": 6},
        "counts": {
            "train": {
                "accountant": {"female": 10, "male": 20},
                "architect": {"female": 5, "male": 15},
            },
            "dev": {"accountant": {"female": 3, "male": 3}},
            "test": {"accountant": {"female": 4, "male": 8}},
            "xai_subset": {"accountant": {"female": 2, "male": 4}},
        },
        "share_women_train": {"accountant": 0.3333, "architect": 0.25},
    }
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    assert raw.decode() == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
