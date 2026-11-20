"""Mean-pooled hidden states of scrubbed bios, the input of the probes (`scripts/probing.py`).

`python scripts/probing_features.py <model>` with model `pretrained` (DeBERTa-v3-small before any
fine-tuning, same tokenizer and maximum length) or `deberta` (the main model). It encodes 10,000
scrubbed train bios, a stratified sample of the training subsample whose ids go to
`results/probing_train_ids.csv`, and the scrubbed showcase subset. For each bio it keeps the
mean over the non-padding tokens of the state after the embeddings and after each of the six
layers, as float16, in `HBX_ARTIFACTS/probing/<model>_<train|showcase>.npy`. Forward passes only;
needs the GPU and nothing else running.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from hiring_bias_xai import data
from hiring_bias_xai.config import RESULTS_DIR, artifacts_dir
from hiring_bias_xai.models import DebertaModel

MODELS = ("pretrained", "deberta")
N_TRAIN = 10_000
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(name: str) -> DebertaModel:
    if name == "deberta":
        model = DebertaModel.load(BATCH_SIZE)
    else:
        max_length = DebertaModel.load().tokenizer.model_max_length
        model = DebertaModel.pretrained(max_length, BATCH_SIZE)
    model.network.to(DEVICE)
    return model


def _encode(model: DebertaModel, texts: list[str]) -> np.ndarray:
    layers, _ = model.representations(texts, pooling="mean")
    return layers.astype(np.float16)


def main(name: str, results_dir: Path = RESULTS_DIR) -> None:
    if name not in MODELS:
        raise ValueError(f"model must be one of {', '.join(MODELS)}; got {name!r}")
    train = data.stratified_sample(data.load_split("train", "scrubbed"), N_TRAIN)
    train[["id"]].to_csv(results_dir / "probing_train_ids.csv", index=False)
    showcase = data.load_showcase_subset(results_dir / "showcase_subset_ids.csv")
    texts = {"train": list(train["text"]), "showcase": [data.scrub(t) for t in showcase["text"]]}
    model = _load(name)
    folder = artifacts_dir() / "probing"
    folder.mkdir(parents=True, exist_ok=True)
    for part, bios in texts.items():
        started = time.perf_counter()
        features = _encode(model, bios)
        np.save(folder / f"{name}_{part}.npy", features)
        print(
            f"{name} {part}: {features.shape} in {time.perf_counter() - started:.0f} s", flush=True
        )


if __name__ == "__main__":
    main(sys.argv[1])
