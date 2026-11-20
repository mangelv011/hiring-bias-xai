import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hiring_bias_xai import data
from hiring_bias_xai.models import DebertaModel

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probing_features.py"
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _train():
    rows = []
    for i in range(24):  # two occupations, two genders, six bios each
        text = " ".join(WORDS[(i + k) % 8] for k in range(2 + i % 3))
        rows.append(
            (f"train-{i}", text, ["nurse", "surgeon"][i % 2], ["female", "male"][i // 2 % 2])
        )
    return pd.DataFrame(rows, columns=["id", "text", "occupation", "gender"])


SHOWCASE = pd.DataFrame(
    [
        ("test-1", "alpha she beta", "nurse", "female"),
        ("test-2", "gamma delta his epsilon", "surgeon", "male"),
        ("test-3", "zeta", "nurse", "male"),
    ],
    columns=["id", "text", "occupation", "gender"],
)


@pytest.fixture
def run(monkeypatch, tmp_path, tiny_network, tiny_tokenizer):
    """Return go(name): the script's main() on 24 train bios and 3 showcase bios with a tiny
    network. `go.model` is the network every name loads; `go.results` the results directory."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "showcase_subset_ids.csv").write_text("id\ntest-1\ntest-2\ntest-3\n")
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setattr(data, "load_showcase_subset", lambda path: SHOWCASE)
    asked = []

    def load_split(split, variant="original"):
        asked.append((split, variant))
        return _train()

    monkeypatch.setattr(data, "load_split", load_split)
    spec = importlib.util.spec_from_file_location("probing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.N_TRAIN = 8
    model = DebertaModel(tiny_network(spread=3.0), tiny_tokenizer(64), batch_size=3)
    loaded = []
    monkeypatch.setattr(module, "_load", lambda name: (loaded.append(name), model)[1])

    def go(name):
        module.main(name, results)

    go.model, go.results, go.loaded, go.asked, go.module = model, results, loaded, asked, module
    return go


def _features(tmp_path, name, part):
    return np.load(tmp_path / "artifacts" / "probing" / f"{name}_{part}.npy")


def test_the_train_ids_are_a_saved_stratified_sample_of_the_scrubbed_train_bios(run, tmp_path):
    run("deberta")
    assert ("train", "scrubbed") in run.asked
    ids = pd.read_csv(run.results / "probing_train_ids.csv")["id"]
    expected = data.stratified_sample(_train(), 8)["id"]
    assert list(ids) == list(expected)
    groups = _train().set_index("id").loc[ids].groupby(["occupation", "gender"]).size()
    assert list(groups) == [2, 2, 2, 2]


def test_features_are_float16_mean_pooled_states_of_every_layer_in_the_order_of_the_ids(
    run, tmp_path
):
    run("deberta")
    ids = pd.read_csv(run.results / "probing_train_ids.csv")["id"]
    train = _train().set_index("id").loc[ids]
    saved = _features(tmp_path, "deberta", "train")
    assert saved.dtype == np.float16
    assert saved.shape == (8, 3, 32)
    expected = run.model.representations(list(train["text"]), pooling="mean")[0]
    np.testing.assert_allclose(saved, expected, atol=2e-3)


def test_showcase_bios_are_scrubbed_before_encoding(run, tmp_path):
    run("deberta")
    saved = _features(tmp_path, "deberta", "showcase")
    assert saved.shape == (3, 3, 32)
    clean = ["alpha beta", "gamma delta epsilon", "zeta"]
    expected = run.model.representations(clean, pooling="mean")[0]
    np.testing.assert_allclose(saved, expected, atol=2e-3)


def test_each_model_name_loads_that_model_and_writes_its_own_files(run, tmp_path):
    run("pretrained")
    run("deberta")
    assert run.loaded == ["pretrained", "deberta"]
    for name in ("pretrained", "deberta"):
        for part in ("train", "showcase"):
            assert _features(tmp_path, name, part).ndim == 3


def test_an_unknown_model_name_is_refused(run):
    with pytest.raises(ValueError, match="model"):
        run("llama")
    assert run.loaded == []


def test_the_default_sample_is_10000_train_bios():
    spec = importlib.util.spec_from_file_location("probing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.N_TRAIN == 10_000
    assert module.MODELS == ("pretrained", "deberta")


def test_the_pretrained_encoder_gets_the_maximum_length_and_batch_size_of_the_main_model(
    monkeypatch, tiny_network, tiny_tokenizer
):
    spec = importlib.util.spec_from_file_location("probing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = DebertaModel(tiny_network(), tiny_tokenizer(77))
    calls = []
    monkeypatch.setattr(DebertaModel, "load", classmethod(lambda cls, batch_size=32: main))
    monkeypatch.setattr(
        DebertaModel,
        "pretrained",
        classmethod(
            lambda cls, max_length, batch_size: calls.append((max_length, batch_size)) or main
        ),
    )
    module._load("pretrained")
    assert calls == [(77, module.BATCH_SIZE)]


def test_the_ids_file_holds_one_id_column_and_nothing_else(run):
    run("deberta")
    assert list(pd.read_csv(run.results / "probing_train_ids.csv").columns) == ["id"]


def test_the_main_model_is_loaded_with_the_script_batch_size(
    monkeypatch, tiny_network, tiny_tokenizer
):
    spec = importlib.util.spec_from_file_location("probing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = []
    model = DebertaModel(tiny_network(), tiny_tokenizer(64))
    monkeypatch.setattr(
        DebertaModel,
        "load",
        classmethod(lambda cls, batch_size: loaded.append(batch_size) or model),
    )
    module.DEVICE = "cpu"
    assert module._load("deberta") is model
    assert loaded == [module.BATCH_SIZE] == [32]
