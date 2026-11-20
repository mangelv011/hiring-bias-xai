import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_xai import proxy_words, results

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "proxy_words.py"
MALE_DOMINATED = ["surgeon", "composer", "software_engineer"]
FEMALE_DOMINATED = ["dietitian", "nurse", "model"]
COLUMNS = [
    "screener",
    "occupation",
    "word",
    "n_bios",
    "contribution",
    "contribution_ci_low",
    "contribution_ci_high",
    "gendered",
]


def _row(screener, occupation, word, contribution, gendered=False):
    return (
        screener,
        occupation,
        word,
        30,
        contribution,
        contribution - 0.1,
        contribution + 0.1,
        gendered,
    )


@pytest.fixture
def run(tmp_path):
    """run(rows, sums): the script's main() on a gap table of `rows` and the main model's gendered
    sums `sums` (occupation to value; the others get 1.0). Gives the parsed proxy_words.json."""

    def go(rows, sums=None):
        spec = importlib.util.spec_from_file_location("proxy_words_script", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            results_dir / "gap_decomposition.csv", index=False
        )
        names = lambda names: [{"occupation": n, "share_women": 0.5} for n in names]  # noqa: E731
        results.write_json(
            results_dir / "showcase_occupations.json",
            {"male_dominated": names(MALE_DOMINATED), "female_dominated": names(FEMALE_DOMINATED)},
        )
        sums = {o: 1.0 for o in MALE_DOMINATED + FEMALE_DOMINATED} | (sums or {})
        value = lambda v: {"value": v, "ci_low": v - 0.1, "ci_high": v + 0.1}  # noqa: E731
        results.write_json(
            results_dir / "gap_explained.json",
            {
                "screeners": {
                    "deberta": {o: {"gendered_sum": value(v)} for o, v in sums.items()},
                    "baseline": {o: {"gendered_sum": value(-v)} for o, v in sums.items()},
                }
            },
        )
        module.main(results_dir)
        return json.loads((results_dir / "proxy_words.json").read_text())

    return go


def test_the_mask_list_is_the_sorted_union_of_the_words_of_the_six_occupations(run):
    rows = [
        _row("deberta", "surgeon", "pager", 3.0),
        _row("deberta", "nurse", "ward", 2.0),
        _row("deberta", "model", "ward", 1.0),
        _row("deberta", "composer", "album", 1.5),
    ]
    out = run(rows)
    assert out["words"] == ["album", "pager", "ward"]
    assert [w["word"] for w in out["occupations"]["nurse"]] == ["ward"]
    assert out["occupations"]["dietitian"] == []


def test_each_occupation_keeps_at_most_five_words(run):
    rows = [_row("deberta", "surgeon", f"w{n}", float(n)) for n in range(1, 9)]
    out = run(rows)
    assert [w["word"] for w in out["occupations"]["surgeon"]] == ["w8", "w7", "w6", "w5", "w4"]
    assert out["words"] == ["w4", "w5", "w6", "w7", "w8"]


def test_only_the_rows_of_the_main_model_count(run):
    rows = [
        _row("baseline", "surgeon", "baseline_only", 9.0),
        _row("deberta", "surgeon", "kept", 1.0),
    ]
    assert run(rows)["words"] == ["kept"]


def test_the_sign_is_that_of_the_main_models_gendered_sum_of_the_occupation(run):
    # The baseline sums in the fixture have the opposite sign, so reading them would flip this.
    rows = [_row("deberta", "surgeon", "up", 1.0), _row("deberta", "surgeon", "down", -1.0)]
    assert run(rows, {"surgeon": -0.5})["words"] == ["down"]
    assert run(rows, {"surgeon": 0.5})["words"] == ["up"]


def test_words_that_pandas_reads_as_missing_values_stay_words(run):
    rows = [
        _row("deberta", "surgeon", w, 1.0 + i) for i, w in enumerate(["nan", "null", "None", "NA"])
    ]
    assert run(rows)["words"] == ["NA", "None", "nan", "null"]


def test_the_file_holds_the_rule_the_sums_and_the_contribution_with_its_interval_of_each_word(run):
    out = run([_row("deberta", "surgeon", "pager", 3.0)], {"nurse": -2.0})
    assert out["rule"] == proxy_words.RULE
    assert out["screener"] == "deberta"
    assert out["top_per_occupation"] == 5
    assert out["gendered_sums"]["nurse"] == -2.0
    assert sorted(out["gendered_sums"]) == sorted(MALE_DOMINATED + FEMALE_DOMINATED)
    (pager,) = out["occupations"]["surgeon"]
    assert pager == {"word": "pager", "contribution": 3.0, "ci_low": 2.9, "ci_high": 3.1}
