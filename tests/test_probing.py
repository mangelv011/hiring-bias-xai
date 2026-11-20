import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hiring_bias_xai import probing


def test_shuffle_within_keeps_every_groups_labels_and_moves_them_inside_the_group():
    groups = np.array(["a"] * 40 + ["b"] * 40)
    labels = np.array(["f"] * 20 + ["m"] * 20 + ["f"] * 5 + ["m"] * 35)
    shuffled = probing.shuffle_within(labels, groups, np.random.default_rng(0))
    for group in ("a", "b"):
        rows = groups == group
        assert sorted(shuffled[rows]) == sorted(labels[rows])
    assert (shuffled != labels).any()
    assert (shuffled[:40] == "f").sum() == 20


def test_shuffle_within_is_repeatable_for_a_seed_and_differs_between_draws():
    groups = np.repeat(["a", "b"], 50)
    labels = np.random.default_rng(1).choice(["f", "m"], size=100)
    first = probing.shuffle_within(labels, groups, np.random.default_rng(7))
    again = probing.shuffle_within(labels, groups, np.random.default_rng(7))
    other = probing.shuffle_within(labels, groups, np.random.default_rng(8))
    assert (first == again).all()
    assert (first != other).any()


def test_the_majority_class_reference_predicts_the_most_common_training_gender():
    assert list(probing.majority_class(["f", "m", "m"], 4)) == ["m"] * 4


def test_the_occupation_only_reference_predicts_the_majority_gender_of_each_occupation():
    train_occ = ["nurse"] * 3 + ["surgeon"] * 3
    train_gender = ["f", "f", "m", "m", "m", "f"]
    got = probing.occupation_only(train_occ, train_gender, ["surgeon", "nurse", "surgeon"])
    assert list(got) == ["m", "f", "m"]


def test_the_occupation_only_reference_gives_a_tie_to_the_first_gender_in_sorted_order():
    got = probing.occupation_only(["nurse"] * 2, ["m", "f"], ["nurse"])
    assert list(got) == ["f"]


def test_accuracy_with_a_seeded_bootstrap_interval_over_bios():
    correct = np.array([True] * 30 + [False] * 10)
    got = probing.accuracy_ci(correct, n_boot=200, seed=5)
    drawn = np.random.default_rng(5).integers(0, 40, size=(200, 40))
    low, high = np.percentile(correct[drawn].mean(axis=1), [2.5, 97.5])
    assert got == {
        "accuracy": 0.75,
        "ci_low": pytest.approx(low),
        "ci_high": pytest.approx(high),
        "n": 40,
    }


def test_accuracy_ci_defaults_to_1000_resamples_and_the_project_seed():
    correct = np.random.default_rng(0).random(60) < 0.7
    assert probing.accuracy_ci(correct) == probing.accuracy_ci(correct, 1000, 42)


def _noisy(n=120, d=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, d))
    x[:, 0] += 0.8 * (y - 0.5)
    return x, y


def test_select_c_takes_the_grid_value_with_the_best_5_fold_accuracy_on_standardised_features():
    x, y = _noisy()
    scores = []
    for c in probing.C_GRID:
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        folds = StratifiedKFold(5, shuffle=True, random_state=42)
        scores.append(cross_val_score(probe, x, y, cv=folds).mean())
    assert len(set(scores)) > 1  # the grid matters on these data
    assert probing.select_c(x, y) == probing.C_GRID[int(np.argmax(scores))]


def test_select_c_gives_a_tie_to_the_smallest_c():
    x = np.array([[-1.0], [1.0]] * 20)
    y = np.array([0, 1] * 20)
    assert probing.select_c(x, y) == min(probing.C_GRID)


def _bios(n, seed, reads_gender):
    """Two occupations whose bios are 80% and 20% female. The first feature tells the occupation;
    the second tells the gender only if `reads_gender`."""
    rng = np.random.default_rng(seed)
    occupation = rng.integers(0, 2, n)
    female = rng.random(n) < np.where(occupation == 0, 0.8, 0.2)
    x = rng.normal(size=(n, 6))
    x[:, 0] += 3 * occupation
    if reads_gender:
        x[:, 1] += 3 * female
    return x, np.where(female, "f", "m"), np.where(occupation == 0, "a", "b")


def _run(reads_gender, n_permutations=20, **kwargs):
    train_x, train_gender, train_occ = _bios(600, 0, reads_gender)
    test_x, test_gender, _ = _bios(300, 1, reads_gender)
    flagged = np.arange(300) % 3 == 0
    return probing.probe(
        train_x, train_gender, train_occ, test_x, test_gender, ~flagged,
        n_permutations=n_permutations, n_boot=50, **kwargs,
    )  # fmt: skip


def test_a_probe_that_reads_gender_beats_every_shuffle_within_occupation():
    result = _run(reads_gender=True)
    row = result["all"]
    assert row["accuracy"] > 0.9
    assert row["p_value"] == pytest.approx(1 / 21)  # (1 + 0 shuffles as good) / (1 + 20)
    assert row["null_mean"] < 0.85  # the null can only learn the occupation


def test_a_probe_that_only_reads_the_occupation_is_not_beyond_the_shuffles():
    # Shuffling gender within occupation keeps what the occupation says about gender, so a probe
    # that reads only the occupation is as good as its shuffled versions.
    row = _run(reads_gender=False)["all"]
    assert row["p_value"] > 0.2
    assert abs(row["accuracy"] - row["null_mean"]) < 0.05


def test_probe_reports_both_subsets_with_their_sizes_the_chosen_c_and_the_shuffle_count():
    result = _run(reads_gender=True, n_permutations=7)
    assert result["n_permutations"] == 7
    assert result["C"] in probing.C_GRID
    assert result["all"]["n"] == 300
    assert result["unflagged"]["n"] == 200
    assert set(result["all"]) == {"accuracy", "ci_low", "ci_high", "n", "p_value", "null_mean"}


def test_the_unflagged_row_scores_the_same_probe_on_the_unflagged_bios_only():
    result = _run(reads_gender=True)
    assert result["unflagged"]["accuracy"] != result["all"]["accuracy"]
    assert result["unflagged"]["p_value"] >= 1 / 21


def test_probe_gives_the_same_result_with_one_worker_or_two():
    assert _run(reads_gender=True, n_permutations=6, n_jobs=1) == _run(
        reads_gender=True, n_permutations=6, n_jobs=2
    )


def _texts(n, seed):
    """Bios of 'knitting' words for women and 'welding' words for men, 90% of the time."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["f", "m"], size=n)
    own = {"f": "knitting yarn", "m": "welding steel"}
    other = {"f": "welding steel", "m": "knitting yarn"}
    texts = [f"works with {(own if rng.random() < 0.9 else other)[g]} every day" for g in gender]
    return texts, gender


def test_the_tfidf_reference_reads_the_words_of_the_same_scrubbed_text():
    train_texts, train_gender = _texts(300, 0)
    test_texts, test_gender = _texts(200, 1)
    predicted = probing.tfidf_predictions(train_texts, train_gender, test_texts)
    assert predicted.shape == (200,)
    assert 0.85 < (predicted == test_gender).mean() < 0.95


def test_the_tfidf_reference_chooses_its_c_from_its_own_grid():
    train_texts, train_gender = _texts(300, 0)
    assert probing.TFIDF_C_GRID != probing.C_GRID
    chosen = probing.select_c(
        np.asarray(train_texts), train_gender, probing.TFIDF_C_GRID, make=probing.new_tfidf
    )
    assert chosen in probing.TFIDF_C_GRID
