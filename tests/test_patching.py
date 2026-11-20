import numpy as np
import pytest

from hiring_bias_xai import models, patching
from hiring_bias_xai.models import DebertaModel


def test_restored_share_is_the_patched_change_over_the_swap_change():
    clean = np.array([2.0, 1.0, -1.0])
    corrupted = np.array([0.0, 3.0, 1.0])
    patched = np.array([1.5, 3.0, 1.0])
    # (1.5 - 0) / (2 - 0); no change at all; no change at all
    np.testing.assert_allclose(patching.restored_share(clean, corrupted, patched), [0.75, 0, 0])


def test_restored_share_is_one_at_the_clean_value_zero_at_the_corrupted_and_can_overshoot():
    clean, corrupted = np.array([1.0, -2.0]), np.array([-1.0, 2.0])
    np.testing.assert_allclose(patching.restored_share(clean, corrupted, clean), [1, 1])
    np.testing.assert_allclose(patching.restored_share(clean, corrupted, corrupted), [0, 0])
    np.testing.assert_allclose(patching.restored_share(clean, corrupted, 3 * clean), [2, 2])


def test_restored_share_refuses_a_pair_with_no_difference():
    with pytest.raises(ValueError, match="differ"):
        patching.restored_share(np.array([1.0, 2.0]), np.array([1.0, 3.0]), np.array([1.0, 3.0]))


def test_eligible_bios_have_a_swap_change_above_the_95th_percentile_of_the_removal_change():
    # |removal| is 0, 1, ..., 100; its 95th percentile is 95.0.
    removal = np.arange(101.0) * np.where(np.arange(101) % 2, -1, 1)
    swap = np.array([-96.0, 95.0, 95.5, -95.5, 0.0, 200.0])
    mask, threshold = patching.eligible(swap, removal)
    assert threshold == 95.0
    assert list(mask) == [True, False, True, True, False, True]


def test_the_eligibility_quantile_can_be_set():
    mask, threshold = patching.eligible(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), 0.5)
    assert threshold == 1.0
    assert list(mask) == [False, True, True]


def test_summarise_gives_mean_and_median_with_seeded_bootstrap_intervals_over_bios():
    shares = np.array([0.0, 0.2, 0.4, 0.6, 1.0, 1.4])
    summary = patching.summarise(shares, n_boot=200, seed=3)
    drawn = np.random.default_rng(3).integers(0, len(shares), size=(200, len(shares)))
    for name, function in (("mean", np.mean), ("median", np.median)):
        low, high = np.percentile(function(shares[drawn], axis=1), [2.5, 97.5])
        assert summary[name] == pytest.approx(function(shares))
        assert summary[f"{name}_ci_low"] == pytest.approx(low)
        assert summary[f"{name}_ci_high"] == pytest.approx(high)
    assert summary["n"] == 6
    assert summary["mean"] == pytest.approx(0.6) and summary["median"] == pytest.approx(0.5)


def test_summarise_is_repeatable_and_the_seed_changes_the_intervals():
    shares = np.random.default_rng(0).normal(size=50)
    assert patching.summarise(shares, 100, 1) == patching.summarise(shares, 100, 1)
    assert patching.summarise(shares, 100, 1) != patching.summarise(shares, 100, 2)


def test_summarise_uses_the_project_seed_and_1000_resamples_by_default():
    shares = np.random.default_rng(0).normal(size=30)
    assert patching.summarise(shares) == patching.summarise(shares, 1000, 42)


def test_true_class_log_odds_uses_each_bios_own_occupation():
    logits = np.random.default_rng(0).normal(size=(5, 28))
    index = np.array([3, 3, 27, 0, 13])
    got = patching.true_class_log_odds(logits, index)
    for row, column in enumerate(index):
        np.testing.assert_allclose(got[row], models.log_odds(logits[row : row + 1], column)[0])


CLEAN = ["alpha beta gamma delta", "zeta eta", "alpha beta gamma delta epsilon theta", "beta gamma"]
CORRUPTED = [
    "alpha zeta gamma eta",
    "eta zeta",
    "theta beta gamma delta epsilon alpha",
    "gamma beta",
]
TRUE = np.array([3, 13, 27, 0])  # the true occupation of each pair, as a column of the 28 logits


@pytest.fixture
def model(tiny_network, tiny_tokenizer):
    return DebertaModel(tiny_network(spread=3.0), tiny_tokenizer(64), batch_size=3)


def test_equal_length_marks_the_pairs_with_the_same_number_of_tokens(model):
    # "alpha beta" and "alpha beta gamma" are 4 and 5 tokens with the two special tokens.
    clean = ["alpha beta", "alpha beta gamma", "zeta"]
    corrupted = ["gamma delta", "alpha beta", "eta"]
    assert list(patching.equal_length(model.tokenizer, clean, corrupted)) == [True, False, True]


def test_equal_length_counts_tokens_after_the_cut_to_the_maximum_length(tiny_tokenizer, model):
    model.tokenizer = tiny_tokenizer(5)  # [CLS], three words, [SEP]
    same = patching.equal_length(
        model.tokenizer, ["alpha beta gamma delta"], ["alpha beta gamma delta epsilon zeta"]
    )
    assert list(same) == [True]


def test_patch_pairs_returns_clean_and_corrupted_log_odds_in_input_order(model):
    result = patching.patch_pairs(model, CLEAN, CORRUPTED, TRUE)
    for texts, key in ((CLEAN, "clean"), (CORRUPTED, "corrupted")):
        alone = [
            patching.true_class_log_odds(model.logits([text]), TRUE[i : i + 1])[0]
            for i, text in enumerate(texts)
        ]
        np.testing.assert_allclose(result[key], alone, atol=1e-4)


def test_patch_pairs_has_one_log_odds_per_bio_layer_and_position_set(model):
    result = patching.patch_pairs(model, CLEAN, CORRUPTED, TRUE)
    assert patching.POSITION_SETS == ("differing", "all")
    assert result["patched"].shape == (4, 3, 2)  # the tiny network has 2 layers


def test_patching_all_positions_at_the_embeddings_restores_the_whole_effect(model):
    # The sanity check of the method: with every position replaced nothing of the corrupted run
    # is left, so the share is 100%.
    result = patching.patch_pairs(model, CLEAN, CORRUPTED, TRUE)
    shares = patching.restored_share(
        result["clean"], result["corrupted"], result["patched"][:, 0, 1]
    )
    np.testing.assert_allclose(shares, 1.0, atol=1e-4)


def test_patching_the_differing_positions_at_the_embeddings_also_restores_everything(model):
    # Before any layer a position's state depends on its own token only.
    result = patching.patch_pairs(model, CLEAN, CORRUPTED, TRUE)
    np.testing.assert_allclose(result["patched"][:, 0, 0], result["clean"], atol=1e-4)


def test_patching_all_positions_at_the_last_layer_restores_everything_but_the_differing_ones_do_not(
    model,
):
    # The classifier reads the [CLS] state of the last layer, and [CLS] never differs.
    result = patching.patch_pairs(model, CLEAN, CORRUPTED, TRUE)
    np.testing.assert_allclose(result["patched"][:, -1, 1], result["clean"], atol=1e-4)
    np.testing.assert_allclose(result["patched"][:, -1, 0], result["corrupted"], atol=1e-4)


def test_patch_pairs_refuses_pairs_of_different_token_length(model):
    with pytest.raises(ValueError, match="length"):
        patching.patch_pairs(model, ["alpha beta"], ["alpha beta gamma"], TRUE[:1])


def test_patch_pairs_refuses_pairs_whose_lengths_differ_even_when_the_padded_batches_match(model):
    # Each batch pads to 5 tokens, but the two texts of a batch swap their lengths.
    with pytest.raises(ValueError, match="length"):
        patching.patch_pairs(
            model, ["alpha beta", "alpha beta gamma"], ["alpha beta gamma", "alpha beta"], TRUE[:2]
        )
