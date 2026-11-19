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


def test_all_coalitions_of_a_bio_go_to_the_classifier_in_one_call_with_each_text_once():
    classifier = Hashed()
    (result,) = attribution.shapley(classifier, ["a b a c"], ["nurse"], 6, "delete", seed=0)
    (texts,) = classifier.calls
    assert len(texts) == len(set(texts)) == len(result.logits)
    # 12 walks of 5 coalitions each, but at most 2^4 = 16 distinct ones.
    assert result.index.size == 60
    assert len(texts) <= 16
    assert "" in texts
    assert "a b a c" in texts


def test_each_bio_gets_its_own_classifier_call():
    classifier = Hashed()
    attribution.shapley(classifier, ["a b", "c d e"], ["nurse", "nurse"], 2, "delete", seed=0)
    assert len(classifier.calls) == 2


def test_with_delete_the_orders_walk_from_the_empty_bio_to_the_full_bio():
    classifier = Hashed()
    (result,) = attribution.shapley(classifier, ["a b c"], ["nurse"], 3, "delete", seed=0)
    (texts,) = classifier.calls
    assert result.index.shape == (6, 4)
    assert len(set(result.index[:, 0])) == 1
    assert texts[result.index[0, 0]] == ""
    assert len(set(result.index[:, -1])) == 1
    assert texts[result.index[0, -1]] == "a b c"
    for k, order in enumerate(result.orders):
        for step in range(1, 4):
            kept = sorted(order[:step])
            assert texts[result.index[k, step]] == " ".join("abc"[i] for i in kept)


def test_with_unk_every_removed_word_becomes_one_unk_in_its_place():
    classifier = Hashed()
    (result,) = attribution.shapley(classifier, ["a b c"], ["nurse"], 3, "unk", seed=0)
    (texts,) = classifier.calls
    assert texts[result.index[0, 0]] == "[UNK] [UNK] [UNK]"
    assert texts[result.index[0, -1]] == "a b c"
    for k, order in enumerate(result.orders):
        kept = set(order[:1])
        expected = " ".join("abc"[i] if i in kept else "[UNK]" for i in range(3))
        assert texts[result.index[k, 1]] == expected


def test_the_stored_logits_are_the_float32_logits_of_the_coalition_texts_in_call_order():
    classifier = Hashed()
    (result,) = attribution.shapley(classifier, ["a b c d"], ["nurse"], 2, "delete", seed=0)
    (texts,) = classifier.calls
    assert result.logits.dtype == np.float32
    assert result.logits.shape == (len(texts), 28)
    np.testing.assert_array_equal(result.logits, classifier.logits(texts))


def test_the_stored_result_carries_the_target_removal_and_time():
    (result,) = attribution.shapley(Hashed(), ["a b"], ["surgeon"], 2, "unk", seed=0)
    assert (result.target, result.removal) == (SURGEON, "unk")
    assert result.seconds > 0


def test_the_values_for_another_occupation_equal_a_run_with_that_target_and_the_same_seed():
    # Two occupations with word weights: the log-odds of one depend on the logit of the other,
    # so the two sets of values differ.
    classifier = BagOfWords({"nurse": {"a": 1.0, "b": -2.0}, "surgeon": {"b": 3.0, "c": 0.5}})
    text = "a b c b"
    (as_nurse,) = attribution.shapley(classifier, [text], ["nurse"], 3, "delete", seed=7)
    (as_surgeon,) = attribution.shapley(classifier, [text], ["surgeon"], 3, "delete", seed=7)
    np.testing.assert_array_equal(as_nurse.orders, as_surgeon.orders)
    np.testing.assert_allclose(as_nurse.values_for(SURGEON), as_surgeon.values)
    np.testing.assert_allclose(as_surgeon.values_for(NURSE), as_nurse.values)
    assert not np.allclose(as_nurse.values, as_surgeon.values, atol=0.1)


def test_values_for_another_occupation_on_nonlinear_scores_match_a_run_with_that_target():
    classifier = Hashed()
    texts = ["a b c d e", "f g h"]
    nurse = attribution.shapley(classifier, texts, ["nurse"] * 2, 4, "unk", seed=2)
    surgeon = attribution.shapley(classifier, texts, ["surgeon"] * 2, 4, "unk", seed=2)
    for as_nurse, as_surgeon in zip(nurse, surgeon, strict=True):
        np.testing.assert_allclose(as_nurse.values_for(SURGEON), as_surgeon.values, atol=1e-9)
