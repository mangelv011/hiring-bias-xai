"""Probing for gender: a linear probe on a layer's representations, its references and its null."""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from hiring_bias_xai.config import SEED

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FOLDS = 5
N_BOOT = 1000


def shuffle_within(labels: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`labels` permuted among the rows of each group, so the label counts of a group stay."""
    shuffled = np.array(labels, copy=True)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        shuffled[rows] = labels[rng.permutation(rows)]
    return shuffled


def majority_class(train_gender: Sequence[str], n: int) -> np.ndarray:
    """The most common training gender, predicted for each of `n` bios; a tie goes to the first
    gender in sorted order."""
    values, counts = np.unique(np.asarray(train_gender), return_counts=True)
    return np.full(n, values[np.argmax(counts)])


def occupation_only(
    train_occupation: Sequence[str], train_gender: Sequence[str], occupation: Sequence[str]
) -> np.ndarray:
    """For each bio, the majority gender of its occupation in the training bios; a tie goes to
    the first gender in sorted order. An occupation absent from training raises KeyError."""
    train_occupation, train_gender = np.asarray(train_occupation), np.asarray(train_gender)
    majority = {
        name: majority_class(train_gender[train_occupation == name], 1)[0]
        for name in np.unique(train_occupation)
    }
    return np.array([majority[name] for name in occupation])


def accuracy_ci(correct: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Accuracy of per-bio hits with the 95% percentile interval of `n_boot` bio resamples."""
    correct = np.asarray(correct, dtype=bool)
    n = len(correct)
    drawn = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    low, high = np.percentile(correct[drawn].mean(axis=1), [2.5, 97.5])
    return {"accuracy": float(correct.mean()), "ci_low": float(low), "ci_high": float(high), "n": n}


def new_probe(c: float) -> Pipeline:
    """Logistic regression on standardised features."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def select_c(features: np.ndarray, labels: np.ndarray, grid: Sequence[float] = C_GRID) -> float:
    """The value of `grid` with the best 5-fold cross-validated accuracy; a tie goes to the
    smallest value. The scaler is fitted inside each fold."""
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    scores = [cross_val_score(new_probe(c), features, labels, cv=folds).mean() for c in grid]
    best = max(scores)
    return min(c for c, score in zip(grid, scores, strict=True) if score == best)
