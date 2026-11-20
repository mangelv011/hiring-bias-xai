import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fakes import Hashed

from hiring_bias_xai import attribution, data, gap_decomposition, results
from hiring_bias_xai.config import SEED

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gap_decomposition.py"
MALE_DOMINATED = ["surgeon", "composer", "software_engineer"]
FEMALE_DOMINATED = ["dietitian", "nurse", "model"]
SCREENERS = ("deberta", "baseline")
TEXTS = {
    "female": [
        "She trains new people every year and she writes her notes late at night in the ward",
        "As a woman in the field she often explains her methods to students who ask about it",
    ],
    "male": [
        "He teaches students every spring and he reviews his notes before his shifts start now",
        "His ward is busy and he supports the patients while his colleagues cover the night",
    ],
}


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(): the script's main() on 24 synthetic bios (two women and two men in each of
    the six showcase occupations) with attributions of a fake classifier; gives the results dir."""
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    model = Hashed()
    rows = []
    for occupation in MALE_DOMINATED + FEMALE_DOMINATED:
        for gender in ("female", "male"):
            for i in range(2):
                rows.append((f"test-{len(rows)}", TEXTS[gender][i], occupation, gender))
    subset = pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "showcase_subset_ids.csv").write_bytes(
        ("id\n" + "".join(f"{i}\n" for i in subset["id"])).encode()
    )
    names = lambda names: [{"occupation": n, "share_women": 0.5} for n in names]  # noqa: E731
    results.write_json(
        results_dir / "showcase_occupations.json",
        {"male_dominated": names(MALE_DOMINATED), "female_dominated": names(FEMALE_DOMINATED)},
    )
    monkeypatch.setattr(data, "load_split", lambda split, variant="original": subset)
    for screener in SCREENERS:
        for row in subset.itertuples():
            seed = 1 if screener == "deberta" else 2  # the two screeners store different values
            (saved,) = attribution.shapley(
                model, [row.text], [row.occupation], 3, "delete", seed=seed
            )
            attribution.save(attribution.path(screener, row.id), saved)
    spec = importlib.util.spec_from_file_location("gap_decomposition_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_BOOT = 20

    def go():
        module.main(results_dir)
        return results_dir

    go.module, go.subset = module, subset
    return go


def test_the_csv_has_the_top_words_of_every_screener_and_occupation_with_their_columns(run):
    table = pd.read_csv(run() / "gap_decomposition.csv")
    assert list(table.columns) == [
        "screener",
        "occupation",
        "word",
        "n_bios",
        "importance",
        "contribution",
        "contribution_ci_low",
        "contribution_ci_high",
        "freq_female",
        "freq_male",
        "value_female",
        "value_male",
        "usage",
        "weight",
        "gendered",
        "top",
    ]
    pairs = table.groupby(["screener", "occupation"])
    assert set(pairs.groups) == {
        (s, o) for s in SCREENERS for o in MALE_DOMINATED + FEMALE_DOMINATED
    }
    # Four bios per occupation: no word reaches 20 bios, so only the top 10 words are kept.
    assert (pairs["top"].sum() == 10).all()
    assert (pairs.size() == 10).all()
    assert table["importance"].isna().all()


def test_a_damaged_or_pickled_attribution_file_stops_the_run_with_a_value_error(run):
    # Boundary: attribution files are local artifacts, read only through attribution.load (no
    # pickle). A file that is not a plain npz attribution must raise, not load or be skipped.
    victim = attribution.path("deberta", run.subset["id"].iloc[0])
    victim.write_bytes(b"not an npz file")
    with pytest.raises(ValueError, match="not a valid attribution file"):
        run()


def test_the_json_has_the_summary_of_every_screener_and_occupation(run, capsys):
    results_dir = run()
    explained = json.loads((results_dir / "gap_explained.json").read_text())
    assert explained["settings"]["n_boot"] == 20
    assert explained["settings"]["min_bios"] == 20
    assert explained["settings"]["top"] == 10
    assert set(explained["screeners"]) == set(SCREENERS)
    table = pd.read_csv(results_dir / "gap_decomposition.csv")
    for screener in SCREENERS:
        assert set(explained["screeners"][screener]) == set(MALE_DOMINATED + FEMALE_DOMINATED)
        for occupation, summary in explained["screeners"][screener].items():
            assert (summary["n_female"], summary["n_male"]) == (2, 2)
            assert summary["decomposition_error"] < 1e-9
            rows = table[(table["screener"] == screener) & (table["occupation"] == occupation)]
            assert summary["top_words"] == list(rows["word"][rows["top"]])
    worst = max(
        summary["decomposition_error"]
        for by_occupation in explained["screeners"].values()
        for summary in by_occupation.values()
    )
    assert explained["max_decomposition_error"] == worst
    shown = capsys.readouterr().out
    for screener, by_occupation in explained["screeners"].items():
        for occupation, summary in by_occupation.items():
            flagged = f"{screener} {occupation}: the total gap" in shown
            assert flagged == summary["gap_unstable"]
            gap = summary["decomposed_gap"]
            if summary["gap_unstable"]:
                assert (
                    f"{screener} {occupation}: the total gap {gap['value']:+.3f} has an interval "
                    f"[{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}] that includes 0, so its "
                    "shares are unstable\n"
                ) in shown


def test_each_summary_is_what_decompose_gives_for_that_screeners_bios_with_their_genders(run):
    results_dir = run()
    explained = json.loads((results_dir / "gap_explained.json").read_text())
    assert explained["settings"]["seed"] == SEED
    for screener in SCREENERS:
        for occupation in MALE_DOMINATED + FEMALE_DOMINATED:
            rows = run.subset[run.subset["occupation"] == occupation]
            bios = [
                gap_decomposition.bio_from_attribution(
                    attribution.load(attribution.path(screener, row.id)), row.gender
                )
                for row in rows.itertuples()
            ]
            expected = gap_decomposition.decompose(bios, n_boot=20).summary
            assert explained["screeners"][screener][occupation] == json.loads(json.dumps(expected))
