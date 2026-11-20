import dataclasses

import numpy as np
import pytest
from fakes import BagOfWords, Hashed

from hiring_bias_xai import attribution, data
from hiring_bias_xai.models import log_odds

NURSE, SURGEON = data.OCCUPATIONS.index("nurse"), data.OCCUPATIONS.index("surgeon")


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


def test_an_unknown_removal_is_rejected_even_without_bios():
    with pytest.raises(ValueError, match="removal"):
        attribution.shapley(Hashed(), [], [], 2, "mask", seed=0)


def test_an_unknown_removal_or_target_is_rejected_before_any_scoring():
    classifier = Hashed()
    with pytest.raises(ValueError, match="removal"):
        attribution.shapley(classifier, ["a b"], ["nurse"], 2, "mask", seed=0)
    with pytest.raises(ValueError):
        attribution.shapley(classifier, ["a b"], ["plumber"], 2, "delete", seed=0)
    assert classifier.calls == []


def test_the_target_list_must_have_one_occupation_per_bio():
    with pytest.raises(ValueError, match="2 bios but 1 targets"):
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


def test_efficiency_error_is_the_gap_between_the_sum_of_the_values_and_the_score_difference():
    (result,) = attribution.shapley(Hashed(), ["a b c d e"], ["nurse"], 3, "delete", seed=0)
    assert result.efficiency_error() < 1e-9
    index = result.index.copy()
    index[0, -1] = index[0, 0]  # the first order now ends on the empty bio
    wrong = dataclasses.replace(result, index=index)
    assert wrong.efficiency_error() > 0.01


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("HBX_ARTIFACTS", str(tmp_path))
    return tmp_path


def _one(text="alpha beta gamma delta", removal="unk", target="surgeon"):
    (result,) = attribution.shapley(Hashed(), [text], [target], 3, removal, seed=4)
    return result


def test_an_attribution_lives_under_the_artifacts_directory_by_screener_and_bio_id(artifacts):
    expected = artifacts / "attributions" / "deberta" / "test-1234.npz"
    assert attribution.path("deberta", "test-1234") == expected


@pytest.mark.parametrize(
    "bio_id", ["../x", "test-1/../2", "test-12.npz", "x-1", "test-", "", "test-1\n"]
)
def test_a_bio_id_that_is_not_split_dash_number_is_rejected_so_it_cannot_leave_the_directory(
    artifacts, bio_id
):
    with pytest.raises(ValueError, match="bio ID"):
        attribution.path("deberta", bio_id)


@pytest.mark.parametrize("screener", ["../deberta", "a/b", "", "Deberta"])
def test_a_screener_name_must_be_lowercase_letters(artifacts, screener):
    with pytest.raises(ValueError, match="screener"):
        attribution.path(screener, "test-1")


def test_a_saved_attribution_loads_back_with_every_field_unchanged(artifacts):
    result = _one()
    target = attribution.path("deberta", "test-1")
    attribution.save(target, result)
    loaded = attribution.load(target)
    assert loaded.words == result.words
    assert (loaded.target, loaded.removal, loaded.seconds) == (
        result.target,
        result.removal,
        result.seconds,
    )
    for name in ("orders", "index", "logits"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
        assert getattr(loaded, name).dtype == getattr(result, name).dtype
    np.testing.assert_array_equal(loaded.values, result.values)


def test_words_with_accents_and_emoji_and_a_bio_without_words_survive_a_round_trip(artifacts):
    for text in ["café señor 日本語 🙂 naïve", " "]:
        result = _one(text)
        attribution.save(attribution.path("baseline", "dev-3"), result)
        loaded = attribution.load(attribution.path("baseline", "dev-3"))
        assert loaded.words == result.words
        assert loaded.values.shape == result.values.shape


def test_saving_writes_through_a_temporary_file_and_leaves_only_the_final_file(artifacts):
    target = attribution.path("deberta", "test-1")
    attribution.save(target, _one())
    attribution.save(target, _one("other words here"))
    assert [file.name for file in target.parent.iterdir()] == ["test-1.npz"]
    assert attribution.load(target).words == ["other", "words", "here"]


def test_a_write_that_fails_halfway_leaves_the_previous_file_and_no_temporary_file(
    artifacts, monkeypatch
):
    target = attribution.path("deberta", "test-1")
    attribution.save(target, _one())
    before = target.read_bytes()

    def broken(file, **arrays):
        file.write(b"PK partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    with pytest.raises(OSError, match="disk full"):
        attribution.save(target, _one("other words here"))
    assert target.read_bytes() == before
    assert [file.name for file in target.parent.iterdir()] == ["test-1.npz"]


def test_a_missing_file_is_not_reported_as_corrupt(artifacts):
    with pytest.raises(FileNotFoundError):
        attribution.load(attribution.path("deberta", "test-9"))


@pytest.mark.parametrize("damage", ["truncated", "garbage", "empty"])
def test_a_damaged_file_raises_a_value_error_naming_it(artifacts, damage):
    target = attribution.path("deberta", "test-1")
    attribution.save(target, _one())
    content = target.read_bytes()
    target.write_bytes(
        {"truncated": content[: len(content) // 2], "garbage": b"x" * 99, "empty": b""}[damage]
    )
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_a_file_with_missing_or_inconsistent_arrays_raises_a_value_error(artifacts):
    target = attribution.path("deberta", "test-1")
    result = _one()
    arrays = {
        "words": np.array(result.words),
        "target": result.target,
        "removal": result.removal,
        "orders": result.orders,
        "index": result.index,
        "logits": result.logits,
        "seconds": result.seconds,
    }
    target.parent.mkdir(parents=True)
    np.savez(target, **{key: value for key, value in arrays.items() if key != "logits"})
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)
    np.savez(target, **arrays | {"index": result.index[:, :-1]})
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)
    np.savez(target, **arrays | {"index": result.index + len(result.logits)})
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_a_file_that_holds_pickled_objects_is_refused_not_unpickled(artifacts):
    # np.load(allow_pickle=True) would run code from such a file.
    target = attribution.path("deberta", "test-1")
    result = _one()
    target.parent.mkdir(parents=True)
    np.savez(
        target,
        words=np.array(result.words, dtype=object),
        target=result.target,
        removal=result.removal,
        orders=result.orders,
        index=result.index,
        logits=result.logits,
        seconds=result.seconds,
    )
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_zero_permutations_are_rejected():
    with pytest.raises(ValueError, match="n_permutations"):
        attribution.shapley(Hashed(), ["a b"], ["nurse"], 0, "delete", seed=0)


def test_a_file_with_a_target_column_outside_the_logits_is_refused(artifacts):
    target = attribution.path("deberta", "test-1")
    attribution.save(target, dataclasses.replace(_one(), target=28))
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_a_file_whose_index_points_one_past_the_last_logits_row_is_refused(artifacts):
    target = attribution.path("deberta", "test-1")
    result = _one()
    index = result.index.copy()
    index[0, 0] = len(result.logits)
    attribution.save(target, dataclasses.replace(result, index=index))
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_attributions_of_train_and_dev_bios_have_paths_too(artifacts):
    assert attribution.path("baseline", "train-7").name == "train-7.npz"
    assert attribution.path("baseline", "dev-7").name == "dev-7.npz"


def _saved_with(artifacts, **changes):
    target = attribution.path("deberta", "test-1")
    attribution.save(target, dataclasses.replace(_one(), **changes))
    return target


def test_a_file_with_an_unknown_removal_is_refused(artifacts):
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(_saved_with(artifacts, removal="mask"))


def test_a_file_with_a_negative_index_is_refused(artifacts):
    index = _one().index.copy()
    index[0, 0] = -1
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(_saved_with(artifacts, index=index))


def test_a_file_whose_orders_do_not_match_the_words_is_refused(artifacts):
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(_saved_with(artifacts, orders=_one().orders[:, :-1]))


def test_a_file_with_one_dimensional_logits_is_refused(artifacts):
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(_saved_with(artifacts, logits=_one().logits[:, 0]))


def test_a_path_that_cannot_be_read_as_a_file_is_a_value_error(artifacts):
    target = attribution.path("deberta", "test-1")
    target.mkdir(parents=True)  # a directory where the file should be
    with pytest.raises(ValueError, match="test-1.npz"):
        attribution.load(target)


def test_a_screener_name_may_carry_a_variant_after_a_dash(artifacts):
    expected = artifacts / "attributions" / "deberta-seed7" / "test-1.npz"
    assert attribution.path("deberta-seed7", "test-1") == expected


@pytest.mark.parametrize(
    "screener", ["deberta-", "-seed7", "deberta--seed7", "deberta-Seed7", "deberta-../x", "a-b/c"]
)
def test_a_variant_must_be_lowercase_letters_or_digits_after_one_dash(artifacts, screener):
    with pytest.raises(ValueError, match="screener"):
        attribution.path(screener, "test-1")


def _saved_one(artifacts):
    result = _one(removal="delete")
    target = attribution.path("deberta", "test-1")
    attribution.save(target, result)
    return target, result


def test_a_file_is_current_when_words_target_removal_and_order_count_match(artifacts):
    target, result = _saved_one(artifacts)
    assert attribution.is_current(target, result.words, result.target, "delete", 3)


@pytest.mark.parametrize(
    "changed",
    [
        {"words": ["alpha", "beta"]},
        {"target": 0},
        {"removal": "unk"},
        {"n_permutations": 4},
    ],
)
def test_a_file_is_not_current_when_one_setting_differs(artifacts, changed):
    target, result = _saved_one(artifacts)
    settings = {
        "words": result.words,
        "target": result.target,
        "removal": "delete",
        "n_permutations": 3,
        **changed,
    }
    assert not attribution.is_current(target, **settings)


def test_a_missing_or_damaged_file_is_not_current(artifacts):
    target = attribution.path("deberta", "test-1")
    assert not attribution.is_current(target, ["alpha"], 0, "delete", 3)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not an npz")
    assert not attribution.is_current(target, ["alpha"], 0, "delete", 3)
