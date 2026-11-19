import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"


def _raw(groups):
    """Raw split from {(profession, gender): size}, groups in blocks."""
    rows = [(f"bio {p} {g}", p, g) for (p, g), size in groups.items() for _ in range(size)]
    return pd.DataFrame(rows, columns=["hard_text", "profession", "gender"])


def _script_runner(monkeypatch, results_dir, raw, sizes, per_group, min_test_bios=500):
    """Return run(): the script's main() on synthetic data, giving the results directory."""
    monkeypatch.setattr(data, "_load_raw", lambda split: raw[split])
    monkeypatch.setattr(data, "_SUBSAMPLE_SIZES", sizes)
    monkeypatch.setattr(data, "_SHOWCASE_PER_GROUP", per_group)
    monkeypatch.setattr(data, "_SHOWCASE_MIN_TEST_BIOS", min_test_bios)
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
        "test": _raw({(0, 0): 8, (0, 1): 4, (1, 0): 6, (1, 1): 6}),
    }
    return _script_runner(monkeypatch, tmp_path, raw, {"train": 50, "dev": 6}, 2, min_test_bios=1)


def test_prepare_data_writes_the_showcase_subset_ids_as_a_single_column_csv_with_lf(prepare):
    # Both occupations are showcase occupations (the cut-off is 1 test bio here), 2 bios per
    # occupation and gender, drawn from the test split.
    results = prepare()
    ids = (results / "showcase_subset_ids.csv").read_bytes().decode()
    assert ids.startswith("id\n")
    assert ids.endswith("\n")
    assert "\r" not in ids
    chosen = ids.split("\n")[1:-1]
    assert len(chosen) == 8
    assert all(re.fullmatch(r"test-\d+", bio_id) for bio_id in chosen)
    test = data.load_split("test").set_index("id")
    counts = test.loc[chosen].groupby(["occupation", "gender"]).size()
    assert counts.tolist() == [2, 2, 2, 2]
    assert not (results / "xai_subset_ids.csv").exists()


def test_prepare_data_writes_the_showcase_occupations_file(monkeypatch, tmp_path):
    # Occupations 0 to 5 (accountant ... composer) have a training share of women of
    # (p + 1) / 10 and 500 test bios each, so all six qualify and split 3 and 3. The dev
    # and train frames differ from the test frame, so mixing them up changes the result.
    raw = {
        "train": _raw({(p, g): (p + 1 if g else 9 - p) for p in range(6) for g in (0, 1)}),
        "dev": _raw({(0, 0): 6, (0, 1): 6}),
        "test": _raw({(p, g): 250 for p in range(6) for g in (0, 1)}),
    }
    run = _script_runner(monkeypatch, tmp_path, raw, {"train": 60, "dev": 6}, 10)
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
    names = ["showcase_subset_ids.csv", "data_summary.json", "showcase_occupations.json"]
    first = {name: (results / name).read_bytes() for name in names}
    prepare()
    assert {name: (results / name).read_bytes() for name in names} == first


def test_prepare_data_writes_the_summary_counts_and_training_shares_of_women(prepare):
    # Train: 100 bios, n = 50 gives exactly half of every group. Dev: 12 bios, n = 6, half.
    # Showcase subset: 2 bios per occupation and gender from the 24 test bios. Share of women in
    # the train subsample: accountant 10 / 30 = 0.3333, architect 5 / 20.
    results = prepare()
    raw = (results / "data_summary.json").read_bytes()
    assert json.loads(raw) == {
        "bios": {"train": 50, "dev": 6, "test": 24, "showcase_subset": 8},
        "counts": {
            "train": {
                "accountant": {"female": 10, "male": 20},
                "architect": {"female": 5, "male": 15},
            },
            "dev": {"accountant": {"female": 3, "male": 3}},
            "test": {
                "accountant": {"female": 4, "male": 8},
                "architect": {"female": 6, "male": 6},
            },
            "showcase_subset": {
                "accountant": {"female": 2, "male": 2},
                "architect": {"female": 2, "male": 2},
            },
        },
        "share_women_train": {"accountant": 0.3333, "architect": 0.25},
    }
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    assert raw.decode() == json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
