"""Fake classifiers for tests: they satisfy `models.Classifier` without a network."""

import zlib

import numpy as np

from hiring_bias_xai import data


class FakeClassifier:
    """Records every `logits` call; subclasses define `score(text)` as the 28 logits."""

    labels = data.OCCUPATIONS

    def __init__(self):
        self.calls = []

    def logits(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.score(text) for text in texts]).astype(np.float32)

    def predict(self, texts):
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]


class BagOfWords(FakeClassifier):
    """Logit of an occupation = sum of the weights its table gives to the words of the text,
    with no normalisation; every other logit is 0. The log-odds of that occupation are then the
    sum of the word weights minus the constant log(27)."""

    def __init__(self, weights_by_occupation):
        super().__init__()
        self.weights_by_occupation = weights_by_occupation

    def score(self, text):
        logits = np.zeros(28)
        for occupation, weights in self.weights_by_occupation.items():
            logits[data.OCCUPATIONS.index(occupation)] = sum(
                weights.get(word, 0.0) for word in text.split()
            )
        return logits


class Hashed(FakeClassifier):
    """Arbitrary nonlinear logits that depend only on the text."""

    def score(self, text):
        return np.random.default_rng(zlib.crc32(text.encode())).normal(size=28) * 3
