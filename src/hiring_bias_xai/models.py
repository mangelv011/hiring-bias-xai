"""Models and the Classifier interface that every screener satisfies."""

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from hiring_bias_xai import data
from hiring_bias_xai.config import artifacts_dir


class Classifier(Protocol):
    """A screener that maps texts to occupation probabilities.

    `predict_proba` returns an array of shape (n, n_labels); each row sums to 1 and
    column i is the probability of `labels[i]`.
    """

    labels: Sequence[str]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


def _baseline_file():
    return artifacts_dir() / "models" / "baseline.joblib"


class BaselineModel:
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


def _deberta_folder():
    return artifacts_dir() / "models" / "deberta"


class Representations(NamedTuple):
    layers: np.ndarray
    final: np.ndarray


class DebertaModel:
    """DeBERTa-v3-small with a classification head over the occupations; satisfies `Classifier`."""

    labels = data.OCCUPATIONS

    def __init__(self, network, tokenizer, batch_size: int = 32):
        self.network = network
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @classmethod
    def pretrained(cls, max_length: int, batch_size: int = 32) -> "DebertaModel":
        """DeBERTa-v3-small at the pinned revision with a new head over the occupations.

        Downloads into `artifacts_dir()/hf`. The repository ships pickle weights only
        (`pytorch_model.bin`); transformers loads them with `torch.load(weights_only=True)`,
        and `save` writes safetensors. `max_length` is the token limit, kept in the tokenizer.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cache_dir = str(artifacts_dir() / "hf")
        network = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_NAME,
            revision=DEBERTA_REVISION,
            cache_dir=cache_dir,
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

    def save(self) -> None:
        """Write the network (safetensors) and the tokenizer to `artifacts_dir()/models/deberta`."""
        self.network.save_pretrained(_deberta_folder())
        self.tokenizer.save_pretrained(_deberta_folder())

    @classmethod
    def load(cls, batch_size: int = 32) -> "DebertaModel":
        """Read the model that `save` wrote."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        network = AutoModelForSequenceClassification.from_pretrained(
            _deberta_folder(), use_safetensors=True
        )
        return cls(network, AutoTokenizer.from_pretrained(_deberta_folder()), batch_size)

    def _batches(self, texts: Sequence[str]):
        """Tokenized batches on the network's device; puts the network in eval mode."""
        self.network.eval()
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            yield inputs.to(self.network.device)

    def logits(self, texts: Sequence[str]) -> np.ndarray:
        """Float32 logits of shape (n, n_labels), in eval mode without gradients."""
        with torch.inference_mode():
            batches = [self.network(**inputs).logits.float() for inputs in self._batches(texts)]
        return torch.cat(batches).cpu().numpy()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return torch.softmax(torch.from_numpy(self.logits(texts)), dim=-1).numpy()

    def representations(self, texts: Sequence[str]) -> Representations:
        """The [CLS] state after the embeddings and after each transformer layer, and the
        final vector the classification head reads.

        `layers` has shape (n, n_layers + 1, hidden_size), `final` (n, hidden_size). `final`
        is the [CLS] state of the last layer passed through the head's pooler (dense layer and
        activation); the classifier is a linear map of it.
        """
        layers, final = [], []
        with torch.inference_mode():
            for inputs in self._batches(texts):
                states = self.network.deberta(**inputs, output_hidden_states=True)
                layers.append(torch.stack([state[:, 0] for state in states.hidden_states], dim=1))
                final.append(self.network.pooler(states.last_hidden_state))
        return Representations(
            torch.cat(layers).float().cpu().numpy(), torch.cat(final).float().cpu().numpy()
        )
