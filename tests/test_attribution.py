import zlib

import numpy as np
import pytest

from hiring_bias_xai import attribution, data
from hiring_bias_xai.models import log_odds

NURSE, SURGEON = data.OCCUPATIONS.index("nurse"), data.OCCUPATIONS.index("surgeon")


class FakeClassifier:
    """Records every `logits` call; subclasses define `score(text)` as the 28 logits."""

    labels = data.OCCUPATIONS

    def __init__(self):
        self.calls = []

    def logits(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.score(text) for text in texts]).astype(np.float32)


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


def _log_odds_of(classifier, text, occupation):
    return float(log_odds(classifier.logits([text]), data.OCCUPATIONS.index(occupation))[0])


@pytest.mark.parametrize(
    ("removed", "removal", "expected"),
    [
        ([False, True, False, True], "delete", "a c"),
        ([False, True, False, True], "unk", "a [UNK] c [UNK]"),
        ([True, True, True, True], "delete", ""),
        ([True, True, True, True], "unk", "[UNK] [UNK] [UNK] [UNK]"),
        ([False] * 4, "delete", "a b c d"),
        ([False] * 4, "unk", "a b c d"),
    ],
)
def test_remove_words_drops_them_or_puts_one_unk_in_each_place(removed, removal, expected):
    assert attribution.remove_words(["a", "b", "c", "d"], removed, removal) == expected


def test_remove_words_rejects_an_unknown_removal():
    with pytest.raises(ValueError, match="removal.*delete.*unk"):
        attribution.remove_words(["a"], [True], "mask")


@pytest.mark.parametrize("removal", ["delete", "unk"])
def test_values_sum_to_the_score_of_the_full_bio_minus_the_score_of_the_empty_bio(removal):
    # Efficiency on arbitrary nonlinear scores, with repeated words, for several bios.
    rng = np.random.default_rng(0)
    pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    texts = [" ".join(rng.choice(pool, size=n)) for n in (1, 2, 5, 8, 11)]
    classifier = Hashed()
    results = attribution.shapley(classifier, texts, ["nurse"] * 5, 4, removal, seed=1)
    for text, result in zip(texts, results, strict=True):
        empty = "" if removal == "delete" else " ".join(["[UNK]"] * len(text.split()))
        expected = _log_odds_of(classifier, text, "nurse") - _log_odds_of(
            classifier, empty, "nurse"
        )
        assert result.values.sum() == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("removal", ["delete", "unk"])
def test_on_a_linear_bag_of_words_every_word_gets_its_weight_with_repeats_counted_each(removal):
    weights = {"alpha": 2.0, "beta": -1.5, "gamma": 0.25}
    classifier = BagOfWords({"nurse": weights})
    text = "alpha beta alpha gamma beta alpha other"
    (result,) = attribution.shapley(classifier, [text], ["nurse"], 3, removal, seed=0)
    assert result.words == text.split()
    np.testing.assert_allclose(result.values, [2.0, -1.5, 2.0, 0.25, -1.5, 2.0, 0.0], atol=1e-6)


def test_words_are_the_whitespace_separated_tokens_of_the_bio():
    classifier = BagOfWords({"nurse": {}})
    (result,) = attribution.shapley(classifier, [" a  b\nc\td "], ["nurse"], 1, "delete", seed=0)
    assert result.words == ["a", "b", "c", "d"]


def test_orders_come_in_antithetic_pairs_and_each_is_a_permutation_of_the_words():
    classifier = Hashed()
    (result,) = attribution.shapley(classifier, ["a b c d e f g"], ["nurse"], 5, "delete", seed=3)
    assert result.orders.shape == (10, 7)
    for row in result.orders:
        assert sorted(row) == list(range(7))
    np.testing.assert_array_equal(result.orders[1::2], result.orders[0::2, ::-1])
    assert len({tuple(row) for row in result.orders[0::2]}) > 1


def test_orders_follow_the_seed():
    classifier = Hashed()

    def orders(seed):
        (result,) = attribution.shapley(classifier, ["a b c d e f g h"], ["nurse"], 4, "unk", seed)
        return result.orders

    np.testing.assert_array_equal(orders(5), orders(5))
    assert not np.array_equal(orders(5), orders(6))


def test_a_bio_without_words_has_no_values():
    (result,) = attribution.shapley(Hashed(), ["  "], ["nurse"], 2, "delete", seed=0)
    assert result.words == []
    assert result.values.shape == (0,)


def test_an_unknown_removal_or_target_is_rejected_before_any_scoring():
    classifier = Hashed()
    with pytest.raises(ValueError, match="removal"):
        attribution.shapley(classifier, ["a b"], ["nurse"], 2, "mask", seed=0)
    with pytest.raises(ValueError):
        attribution.shapley(classifier, ["a b"], ["plumber"], 2, "delete", seed=0)
    assert classifier.calls == []


def test_the_target_list_must_have_one_occupation_per_bio():
    with pytest.raises(ValueError, match="2.*1"):
        attribution.shapley(Hashed(), ["a b", "c d"], ["nurse"], 2, "delete", seed=0)
