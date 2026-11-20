import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hiring_bias_xai import counterfactual, data, first_names

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probing.py"
N_TRAIN, N_TEST, N_LAYERS = 80, 40, 3
N_PERMUTATIONS = 5


def _bios(prefix, n, seed):
    """Two occupations, 80% and 20% female; the text names a topic of the occupation and, for
    some bios, a first name. The tiny vocabulary keeps the TF-IDF reference well defined."""
    rng = np.random.default_rng(seed)
    occupation = rng.integers(0, 2, n)
    female = rng.random(n) < np.where(occupation == 0, 0.8, 0.2)
    names = np.where(rng.random(n) < 0.25, "Maria ", "")
    topics = np.where(occupation == 0, "patients care ward", "surgery theatre scalpel")
    frame = pd.DataFrame(
        {
            "id": [f"{prefix}-{i}" for i in range(n)],
            "text": [
                f"{a}works with {b}. She is calm." for a, b in zip(names, topics, strict=True)
            ],
            "occupation": np.where(occupation == 0, "nurse", "surgeon"),
            "gender": np.where(female, "female", "male"),
        }
    )
    return frame, female


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Return go(*names): the script's main() with synthetic bios and features. The `deberta`
    features carry gender, the `pretrained` ones are noise."""
    train, train_female = _bios("train", N_TRAIN, 0)
    showcase, test_female = _bios("test", N_TEST, 1)
    results = tmp_path / "results"
    results.mkdir()
    train_ids = train["id"].sample(frac=1, random_state=0)  # the ids file has its own order
    train_ids.to_frame().to_csv(results / "probing_train_ids.csv", index=False)
    train = train.set_index("id").loc[train_ids].reset_index()
    train_female = pd.Series(train_female, index=[f"train-{i}" for i in range(N_TRAIN)])[train_ids]
    folder = tmp_path / "artifacts" / "probing"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(2)
    for name in ("pretrained", "deberta"):
        for part, n, female in (
            ("train", N_TRAIN, train_female),
            ("showcase", N_TEST, test_female),
        ):
            x = rng.normal(size=(n, N_LAYERS, 4)).astype(np.float16)
            if name == "deberta":
                x[:, :, 0] += 4 * np.asarray(female)[:, None]
            np.save(folder / f"{name}_{part}.npy", x)
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        data, "load_split", lambda split, variant="original": train.sample(frac=1, random_state=1)
    )
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: showcase)
    monkeypatch.setattr(first_names, "load", lambda: frozenset({"Maria"}))
    monkeypatch.setattr(
        counterfactual, "first_names_in", lambda text, names: ["Maria"] if "Maria" in text else []
    )
    spec = importlib.util.spec_from_file_location("probing_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_PERMUTATIONS = N_PERMUTATIONS
    module.N_BOOT = 30

    def go(*names):
        for name in names:
            module.main(name, results)
        return results

    go.module, go.showcase, go.train = module, showcase, train
    return go


def _table(results):
    return pd.read_csv(results / "probing.csv")


def test_nothing_is_written_to_results_until_every_part_has_run(run):
    results = run("pretrained", "references")
    assert not (results / "probing.csv").exists()
    run("deberta")
    assert (results / "probing.csv").exists()


def test_the_csv_has_a_row_per_model_layer_and_subset_and_one_per_reference_and_subset(run):
    table = _table(run("pretrained", "deberta", "references"))
    assert list(table.columns) == [
        "model", "layer", "subset", "accuracy", "ci_low", "ci_high", "p_value", "n", "c",
        "n_permutations", "null_mean",
    ]  # fmt: skip
    probes = table[table["model"].isin(["pretrained", "deberta"])]
    assert len(probes) == 2 * N_LAYERS * 2
    assert sorted(probes["layer"].unique()) == [0, 1, 2]
    references = table[~table["model"].isin(["pretrained", "deberta"])]
    assert sorted(references["model"].unique()) == ["majority", "occupation_only", "tfidf"]
    assert len(references) == 3 * 2
    assert references["p_value"].isna().all() and references["layer"].isna().all()
    assert set(table["subset"]) == {"all", "unflagged"}


def test_subsets_hold_all_bios_and_those_without_a_first_name(run):
    table = _table(run("pretrained", "deberta", "references"))
    flagged = run.showcase["text"].str.contains("Maria").sum()
    sizes = table.groupby("subset")["n"].unique()
    assert list(sizes["all"]) == [N_TEST]
    assert list(sizes["unflagged"]) == [N_TEST - flagged]


def test_a_probe_on_features_that_carry_gender_is_beyond_the_occupation_reference(run):
    table = _table(run("pretrained", "deberta", "references"))
    everyone = table[table["subset"] == "all"]
    deberta = everyone[everyone["model"] == "deberta"]
    reference = everyone[everyone["model"] == "occupation_only"]["accuracy"].iloc[0]
    assert (deberta["accuracy"] > reference + 0.05).all()
    assert (deberta["p_value"] < 0.5).all()  # the p-value itself is tested in test_probing
    assert (deberta["n_permutations"] == N_PERMUTATIONS).all()


def test_references_are_computed_on_the_same_test_bios_from_the_training_ids(run):
    table = _table(run("pretrained", "deberta", "references"))
    everyone = table[table["subset"] == "all"].set_index("model")
    test_gender = run.showcase["gender"].to_numpy()
    train_gender = run.train["gender"]
    majority = train_gender.value_counts().idxmax()
    assert everyone.loc["majority", "accuracy"] == (test_gender == majority).mean()
    per_occupation = run.train.groupby("occupation")["gender"].agg(
        lambda g: g.value_counts().idxmax()
    )
    expected = (run.showcase["occupation"].map(per_occupation).to_numpy() == test_gender).mean()
    assert everyone.loc["occupation_only", "accuracy"] == expected
    assert everyone.loc["tfidf", "accuracy"] > 0.5
    assert (everyone.loc[["majority", "occupation_only", "tfidf"], "n"] == N_TEST).all()


def test_the_json_records_the_settings_the_flag_counts_and_the_seconds_of_every_part(run):
    results = run("pretrained", "deberta", "references")
    record = json.loads((results / "probing.json").read_text())
    flagged = int(run.showcase["text"].str.contains("Maria").sum())
    assert record["settings"] == {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_boot": 30,
        "n_permutations": N_PERMUTATIONS,
        "seed": 42,
        "c_grid": list(run.module.probing.C_GRID),
        "tfidf_c_grid": list(run.module.probing.TFIDF_C_GRID),
        "folds": 5,
    }
    assert record["n_flagged"] == flagged
    assert set(record["seconds"]) == {"pretrained", "deberta", "references"}
    assert all(value >= 0 for value in record["seconds"].values())


def test_an_unknown_part_is_refused(run):
    with pytest.raises(ValueError, match="part"):
        run("llama")


def test_the_defaults_are_100_shuffles_and_1000_resamples():
    spec = importlib.util.spec_from_file_location("probing_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.N_PERMUTATIONS == 100
    assert module.N_BOOT == 1000


def test_layers_are_written_as_whole_numbers_and_references_leave_them_empty(run):
    results = run("pretrained", "deberta", "references")
    lines = (results / "probing.csv").read_text().splitlines()
    layers = {line.split(",")[1] for line in lines[1:]}
    assert layers == {"0", "1", "2", ""}


def test_the_probe_gets_the_shuffle_count_the_resamples_the_seed_and_the_workers(run, monkeypatch):
    calls = []
    real = run.module.probing.probe

    def spy(*args):
        calls.append(args[6:])
        return real(*args)

    monkeypatch.setattr(run.module.probing, "probe", spy)
    run("pretrained")
    assert set(calls) == {(N_PERMUTATIONS, 30, 42, run.module.N_JOBS)}


def test_the_text_that_goes_to_the_tfidf_reference_is_scrubbed(run, monkeypatch):
    seen = []
    real = run.module.probing.tfidf_predictions

    def spy(train_texts, train_gender, test_texts):
        seen.append(list(test_texts))
        return real(train_texts, train_gender, test_texts)

    monkeypatch.setattr(run.module.probing, "tfidf_predictions", spy)
    run("references")
    assert seen and not any("She" in text for text in seen[0])
