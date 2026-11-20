"""Word-level Shapley values of a classifier's score, by permutation sampling.

The score of a text is the log-odds of one occupation against all the others, computed from the
logits (`models.log_odds`). A word's value is its marginal effect on that score, averaged over
random orders in which the words of the bio are added to an empty bio. There is no reference bio:
what "absent" means is the `removal`, and the pilot in `scripts/shapley_pilot.py` measures which
one disturbs the model less.

- Words are the whitespace-separated tokens of the bio. A word is removed from the text, so all
  its sub-words go together.
- `removal="delete"` drops the word and rejoins the rest with single spaces.
- `removal="unk"` replaces each removed word with one `[UNK]`. The main model's tokenizer turns
  it into its unknown token. The baseline model's TF-IDF reads it as the word "unk", which is not
  in its vocabulary: the word is ignored, but it still breaks the bigrams around it.
- Orders are sampled with antithetic pairs: each random order is followed by its reverse.
- Each order walks from the empty bio to the full bio, one word at a time, so the values of one
  order add up to score(full bio) - score(empty bio) (efficiency), and so does their average.
  The full bio is the words rejoined with single spaces.
"""

import os
import re
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiring_bias_xai.config import artifacts_dir
from hiring_bias_xai.models import Classifier, log_odds

UNK = "[UNK]"
REMOVALS = ("delete", "unk")


def _check_removal(removal: str) -> None:
    if removal not in REMOVALS:
        raise ValueError(f"removal must be one of {', '.join(REMOVALS)}; got {removal!r}")


def remove_words(words: Sequence[str], removed: Sequence[bool], removal: str) -> str:
    """The text of `words` without the ones flagged in `removed`, joined with single spaces."""
    _check_removal(removal)
    if removal == "delete":
        return " ".join(word for word, gone in zip(words, removed, strict=True) if not gone)
    return " ".join(UNK if gone else word for word, gone in zip(words, removed, strict=True))


@dataclass(frozen=True)
class Attribution:
    """The Shapley estimate for one bio, with what is needed to recompute it for any occupation.

    `orders` has one row per sampled order and one column per word (`orders[0::2]` are the random
    orders, `orders[1::2]` their reverses). Row `k` of `index` lists, for the coalitions that
    order walks through (the first `j` words of the order are present, j = 0 to the number of
    words), the row of `logits` that holds the 28 logits of that coalition's text. Identical
    texts share a row.
    """

    words: list[str]
    target: int
    removal: str
    orders: np.ndarray
    index: np.ndarray
    logits: np.ndarray
    seconds: float

    def values_for(self, label: int) -> np.ndarray:
        """Shapley value of each word for the occupation at column `label`, in float64."""
        scores = log_odds(self.logits, label)[self.index]
        values = np.zeros(len(self.words))
        np.add.at(values, self.orders.ravel(), np.diff(scores, axis=1).ravel())
        return values / len(self.orders)

    @property
    def values(self) -> np.ndarray:
        """Shapley value of each word for the target occupation."""
        return self.values_for(self.target)

    def efficiency_error(self) -> float:
        """How far the values are from adding up to score(full bio) - score(empty bio)."""
        scores = log_odds(self.logits, self.target)[self.index]
        return float(abs(self.values.sum() - (scores[0, -1] - scores[0, 0])))


def _shapley_one(
    classifier: Classifier,
    text: str,
    target: int,
    n_permutations: int,
    removal: str,
    rng: np.random.Generator,
) -> Attribution:
    started = time.perf_counter()
    words = text.split()
    n_words = len(words)
    orders = np.empty((2 * n_permutations, n_words), dtype=np.int32)
    for k in range(n_permutations):
        orders[2 * k] = rng.permutation(n_words)
        orders[2 * k + 1] = orders[2 * k, ::-1]
    row_of_text: dict[str, int] = {}
    index = np.empty((len(orders), n_words + 1), dtype=np.int32)
    for k, order in enumerate(orders):
        removed = [True] * n_words
        for step in range(n_words + 1):
            index[k, step] = row_of_text.setdefault(
                remove_words(words, removed, removal), len(row_of_text)
            )
            if step < n_words:
                removed[order[step]] = False
    logits = np.asarray(classifier.logits(list(row_of_text)), dtype=np.float32)
    return Attribution(words, target, removal, orders, index, logits, time.perf_counter() - started)


def shapley(
    classifier: Classifier,
    texts: Sequence[str],
    target: Sequence[str],
    n_permutations: int,
    removal: str,
    seed,
) -> list[Attribution]:
    """Shapley values of the words of each bio for its target occupation.

    `target[i]` is the occupation whose log-odds are explained for `texts[i]`. Each bio gets
    `n_permutations` random orders and their reverses, from one generator seeded with `seed`
    (an int or a sequence of ints). All the coalitions of a bio go to the classifier in one
    `logits` call, each distinct text once.
    """
    _check_removal(removal)
    if len(target) != len(texts):
        raise ValueError(f"{len(texts)} bios but {len(target)} targets")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1; got {n_permutations}")
    columns = [classifier.labels.index(label) for label in target]
    rng = np.random.default_rng(seed)
    return [
        _shapley_one(classifier, text, column, n_permutations, removal, rng)
        for text, column in zip(texts, columns, strict=True)
    ]


# Files are written by `save` under the artifacts directory and read back to resume a run. The
# bio ID and the screener name become part of a path, so both must match a fixed pattern.
_BIO_ID = re.compile(r"(?:train|dev|test)-[0-9]+")
_SCREENER = re.compile(r"[a-z]+(?:-[a-z0-9]+)?")


def path(screener: str, bio_id: str) -> Path:
    """Where the attribution of one bio by one screener is stored."""
    if not _SCREENER.fullmatch(screener):
        raise ValueError(
            f"screener must be lowercase letters, with an optional -variant; got {screener!r}"
        )
    if not _BIO_ID.fullmatch(bio_id):
        raise ValueError(f"bio ID must be <split>-<number>; got {bio_id!r}")
    return artifacts_dir() / "attributions" / screener / f"{bio_id}.npz"


def save(file_path: Path, result: Attribution) -> None:
    """Write `result` as an npz file (no pickle) through a temporary file, so a crash cannot
    leave a half-written file under the final name."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_name(file_path.name + ".tmp")
    try:
        with open(temporary, "wb") as file:
            np.savez(
                file,
                words=np.array(result.words, dtype=str),
                target=result.target,
                removal=result.removal,
                orders=result.orders,
                index=result.index,
                logits=result.logits,
                seconds=result.seconds,
            )
        os.replace(temporary, file_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _check(result: Attribution) -> None:
    n_words = len(result.words)
    rows = result.orders.shape[0] if result.orders.ndim == 2 else -1
    if result.removal not in REMOVALS:
        raise ValueError(f"unknown removal {result.removal!r}")
    if result.orders.shape != (rows, n_words) or result.index.shape != (rows, n_words + 1):
        raise ValueError("orders and index do not match the words")
    if result.logits.ndim != 2 or not 0 <= result.target < result.logits.shape[1]:
        raise ValueError("logits do not match the target")
    if result.index.size and not 0 <= result.index.min() <= result.index.max() < len(result.logits):
        raise ValueError("index points outside the logits")


def load(file_path: Path) -> Attribution:
    """Read a file that `save` wrote. Raises FileNotFoundError if there is none and ValueError
    if it is damaged or not an attribution file; pickled objects are never loaded."""
    try:
        with np.load(file_path, allow_pickle=False) as file:
            result = Attribution(
                words=[str(word) for word in file["words"]],
                target=int(file["target"]),
                removal=str(file["removal"]),
                orders=file["orders"],
                index=file["index"],
                logits=file["logits"],
                seconds=float(file["seconds"]),
            )
        _check(result)
    except FileNotFoundError:
        raise
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"{file_path.name} is not a valid attribution file: {error}") from error
    return result
