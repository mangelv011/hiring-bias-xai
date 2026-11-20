"""Checks of the Shapley explanations: deletion curves, seed and removal stability, leave-one-out.

A deletion curve removes words from a bio in an order and follows the main model's (or the
baseline model's) call as more words go. The words are removed with `unk`, the removal that the
attributions did not use, so the check is not circular for the main model.
"""


def words_to_remove(n_words: int, percent: int) -> int:
    """How many of `n_words` words go at `percent` percent: the share, rounded half up."""
    return (percent * n_words + 50) // 100
