"""Models and the Classifier interface that every screener satisfies."""

from collections.abc import Sequence
from typing import Protocol

import joblib
import numpy as np
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
