import pandas as pd

from hiring_bias_xai import proxy_words


def _table(*rows):
    """Rows of (word, contribution, ci_low, ci_high, gendered) of one occupation."""
    return pd.DataFrame(
        rows,
        columns=["word", "contribution", "contribution_ci_low", "contribution_ci_high", "gendered"],
    )


def _words(table, gendered_sum=1.0, **kwargs):
    return proxy_words.proxy_rows(table, gendered_sum, **kwargs)["word"].tolist()


def test_a_gendered_word_is_never_a_proxy_word():
    table = _table(("she", 9.0, 8.0, 10.0, True), ("alpha", 1.0, 0.5, 1.5, False))
    assert _words(table) == ["alpha"]


def test_a_word_whose_interval_includes_zero_is_not_a_proxy_word():
    table = _table(
        ("alpha", 1.0, 0.5, 1.5, False),
        ("beta", 2.0, -0.1, 4.0, False),
        ("gamma", 2.0, 0.0, 4.0, False),
        ("delta", 2.0, -4.0, 0.0, False),
    )
    assert _words(table) == ["alpha"]
