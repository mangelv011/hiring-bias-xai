"""Models and the Classifier interface that every screener satisfies."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Classifier(Protocol):
    """A screener that maps texts to occupation probabilities.

    `predict_proba` returns an array of shape (n, n_labels); each row sums to 1 and
    column i is the probability of `labels[i]`.
    """

    labels: Sequence[str]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...
