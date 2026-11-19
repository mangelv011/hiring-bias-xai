"""Models and the Classifier interface that every screener satisfies."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from hiring_bias_xai import data


class Classifier(Protocol):
    """A screener that maps texts to occupation probabilities.

    `predict_proba` returns an array of shape (n, n_labels); each row sums to 1 and
    column i is the probability of `labels[i]`.
    """

    labels: Sequence[str]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


class BaselineModel:
    """TF-IDF (word 1-2 grams) with multinomial logistic regression; satisfies `Classifier`."""

    labels = data.OCCUPATIONS

    def __init__(self, C: float = 1.0, min_df: int = 2):
        self.C = C
        self.min_df = min_df

    def fit(self, texts: Sequence[str], occupations: Sequence[str]) -> "BaselineModel":
        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df),
            LogisticRegression(C=self.C),
        ).fit(texts, occupations)
        return self
