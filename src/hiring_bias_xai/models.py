"""Models and the Classifier interface that every screener satisfies."""

import re
from collections.abc import Sequence
from typing import NamedTuple, Protocol

import joblib
import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from hiring_bias_xai import data
from hiring_bias_xai.config import artifacts_dir


class Classifier(Protocol):
    """A screener that maps texts to occupation scores.

    `logits` returns an array of shape (n, n_labels), column i the score of `labels[i]`.
    `predict_proba` is its softmax, so each row sums to 1. `predict` gives the name of the
    highest-scoring occupation of each text. `class_log_odds` gives, for one occupation, the
    log of its odds against all the others, computed from the logits (see `log_odds`).
    """

    labels: Sequence[str]

    def logits(self, texts: Sequence[str]) -> np.ndarray: ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...

    def predict(self, texts: Sequence[str]) -> list[str]: ...

    def class_log_odds(self, texts: Sequence[str], label: str) -> np.ndarray: ...


def log_odds(logits: np.ndarray, index: int) -> np.ndarray:
    """Log-odds of column `index` of `logits` against the others, as float64.

    z_c - logsumexp over k != c of z_k. Computed from logits because probabilities saturate to
    exactly 0 and 1 in float32.
    """
    logits = np.asarray(logits, dtype=np.float64)
    return logits[:, index] - logsumexp(np.delete(logits, index, axis=1), axis=1)


class _Screener:
    """`predict` and `class_log_odds` for any class with `labels` and `logits`."""

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]

    def class_log_odds(self, texts: Sequence[str], label: str) -> np.ndarray:
        return log_odds(self.logits(texts), self.labels.index(label))


def _baseline_file():
    return artifacts_dir() / "models" / "baseline.joblib"


class BaselineModel(_Screener):
    """TF-IDF (word 1-2 grams) with multinomial logistic regression; satisfies `Classifier`."""

    labels = data.OCCUPATIONS

    def __init__(self, C: float = 1.0, min_df: int = 2):
        self.C = C
        self.min_df = min_df

    def fit(self, texts: Sequence[str], occupations: Sequence[str]) -> "BaselineModel":
        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, dtype=np.float32),
            LogisticRegression(C=self.C, max_iter=1000),
        ).fit(texts, occupations)
        return self

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Float64 scores, shape (n, n_labels); occupations never seen in `fit` get -inf."""
        seen = self.pipeline.decision_function(texts)
        if seen.ndim == 1:  # two occupations: sklearn gives one score, the logit of the second
            seen = np.stack([-seen / 2, seen / 2], axis=1)
        columns = [self.labels.index(name) for name in self.pipeline.classes_]
        logits = np.full((len(seen), len(self.labels)), -np.inf)
        logits[:, columns] = seen
        return logits

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        seen = self.pipeline.predict_proba(texts)
        columns = [self.labels.index(name) for name in self.pipeline.classes_]
        proba = np.zeros((len(seen), len(self.labels)))
        proba[:, columns] = seen
        return proba

    def save(self) -> None:
        """Write the model to `artifacts_dir()/models/baseline.joblib`."""
        _baseline_file().parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, _baseline_file())

    @classmethod
    def load(cls) -> "BaselineModel":
        """Read the model that `save` wrote.

        joblib uses pickle, so loading a file can run code. Only load the file that this
        project wrote under `artifacts_dir()`; never one from another source.
        """
        return joblib.load(_baseline_file())


DEBERTA_NAME = "microsoft/deberta-v3-small"
# A commit of the Microsoft repository, so a later push there cannot change the weights.
DEBERTA_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"


def autocast(network):
    """Mixed precision: bfloat16 matrix products on a CUDA network, nothing on the CPU."""
    on_gpu = network.device.type == "cuda"
    return torch.autocast(device_type=network.device.type, dtype=torch.bfloat16, enabled=on_gpu)


def _deberta_folder(name: str):
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError(
            f"checkpoint name must be lowercase letters, digits, hyphens; got {name!r}"
        )
    return artifacts_dir() / "models" / name


class Representations(NamedTuple):
    layers: np.ndarray
    final: np.ndarray


class DebertaModel(_Screener):
    """DeBERTa-v3-small with a classification head over the occupations; satisfies `Classifier`."""

    labels = data.OCCUPATIONS

    def __init__(self, network, tokenizer, batch_size: int = 32):
        self.network = network
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @classmethod
    def pretrained(cls, max_length: int, batch_size: int = 32) -> "DebertaModel":
        """DeBERTa-v3-small at the pinned revision with a new head over the occupations.

        Downloads into `artifacts_dir()/hf`. The repository ships pickle weights only, which
        transformers loads with `weights_only=True`; `save` writes safetensors. `max_length`
        is kept in the tokenizer.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cache_dir = str(artifacts_dir() / "hf")
        network = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_NAME,
            revision=DEBERTA_REVISION,
            cache_dir=cache_dir,
            dtype=torch.float32,
            num_labels=len(cls.labels),
            id2label=dict(enumerate(cls.labels)),
            label2id={name: i for i, name in enumerate(cls.labels)},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            DEBERTA_NAME,
            revision=DEBERTA_REVISION,
            cache_dir=cache_dir,
            model_max_length=max_length,
        )
        return cls(network, tokenizer, batch_size)

    def save(self, name: str = "deberta") -> None:
        """Write the network (safetensors) and the tokenizer to `artifacts_dir()/models/<name>`.

        `name` is a plain word (lowercase letters, digits, hyphens), so it cannot leave that folder.
        """
        folder = _deberta_folder(name)
        self.network.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)

    @classmethod
    def load(cls, batch_size: int = 32, name: str = "deberta") -> "DebertaModel":
        """Read the model that `save` wrote under `name`."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        folder = _deberta_folder(name)
        network = AutoModelForSequenceClassification.from_pretrained(folder, use_safetensors=True)
        return cls(network, AutoTokenizer.from_pretrained(folder), batch_size)

    def batches(self, texts: Sequence[str]):
        """Tokenized batches on the network's device; puts the network in eval mode.

        Texts are batched by token length, so short ones are not padded to long ones. Yields
        (positions, inputs), where `positions` are the indices of the batch's texts in `texts`.
        """
        self.network.eval()
        encoded = self.tokenizer(list(texts), truncation=True)
        by_length = np.argsort([len(ids) for ids in encoded["input_ids"]], kind="stable")
        for start in range(0, len(texts), self.batch_size):
            positions = by_length[start : start + self.batch_size]
            features = [{key: encoded[key][i] for key in encoded} for i in positions]
            inputs = self.tokenizer.pad(features, return_tensors="pt")
            yield positions, inputs.to(self.network.device)

    @staticmethod
    def _in_input_order(batches: list, positions: list) -> np.ndarray:
        """Concatenate per-batch outputs and put the rows back in the order of the texts."""
        stacked = torch.cat(batches).float().cpu().numpy()
        return stacked[np.argsort(np.concatenate(positions))]

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Float32 logits of shape (n, n_labels), in eval mode without gradients."""
        batches, positions = [], []
        with torch.inference_mode(), autocast(self.network):
            for chosen, inputs in self.batches(texts):
                batches.append(self.network(**inputs).logits)
                positions.append(chosen)
        return self._in_input_order(batches, positions)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self.logits(texts)), dim=-1).numpy()

    def representations(self, texts: Sequence[str], pooling: str = "cls") -> Representations:
        """The state after the embeddings and after each transformer layer, and the final vector
        the classification head reads.

        `layers` has shape (n, n_layers + 1, hidden_size): with `pooling="cls"` the [CLS] state,
        with `pooling="mean"` the mean over the non-padding tokens, special tokens included.
        `final` (n, hidden_size) is the [CLS] state of the last layer passed through the head's
        pooler (dense layer and activation); the classifier is a linear map of it.
        """
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be cls or mean; got {pooling!r}")
        layers, final, positions = [], [], []
        with torch.inference_mode(), autocast(self.network):
            for chosen, inputs in self.batches(texts):
                states = self.network.deberta(**inputs, output_hidden_states=True)
                stacked = torch.stack(states.hidden_states, dim=1)  # (batch, layer, token, dim)
                if pooling == "cls":
                    layers.append(stacked[:, :, 0])
                else:
                    mask = inputs["attention_mask"][:, None, :, None].to(stacked.dtype)
                    layers.append((stacked * mask).sum(dim=2) / mask.sum(dim=2))
                final.append(self.network.pooler(states.last_hidden_state))
                positions.append(chosen)
        return Representations(
            self._in_input_order(layers, positions), self._in_input_order(final, positions)
        )

    def states(self, inputs) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Logits and the token-level states of one batch from `batches`.

        The states are the output of the embeddings and of each transformer layer, each of shape
        (batch, token, hidden_size).
        """
        with torch.inference_mode(), autocast(self.network):
            output = self.network(**inputs, output_hidden_states=True)
        return output.logits.float(), output.hidden_states

    def patched_logits(
        self, inputs, layer: int, replacement: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Logits of one batch with the output of `layer` replaced where `mask` is True.

        `layer` 0 is the embeddings, `layer` k the output of transformer layer k. `replacement`
        has the shape of that output, (batch, token, hidden_size); `mask` is (batch, token).
        """
        module = (
            self.network.deberta.embeddings
            if layer == 0
            else self.network.deberta.encoder.layer[layer - 1]
        )

        def patch(_, __, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = torch.where(mask[..., None], replacement.to(hidden.dtype), hidden)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = module.register_forward_hook(patch)
        try:
            with torch.inference_mode(), autocast(self.network):
                return self.network(**inputs).logits.float()
        finally:
            handle.remove()
