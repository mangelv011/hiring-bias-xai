import pytest

from hiring_bias_xai import fairness

N, S = "nurse", "surgeon"

# Hand-built example: 2 occupations x 2 genders, 12 bios.
#   true     gender  count  predicted
#   nurse    female  4      nurse, nurse, nurse, surgeon
#   nurse    male    2      nurse, surgeon
#   surgeon  female  2      surgeon, nurse
#   surgeon  male    4      surgeon x4
ROWS = (
    [(N, "female", p) for p in (N, N, N, S)]
    + [(N, "male", p) for p in (N, S)]
    + [(S, "female", p) for p in (S, N)]
    + [(S, "male", S)] * 4
)
SHARE_WOMEN = {N: 0.9, S: 0.1}


def _report(**kwargs):
    y_true, gender, y_pred = zip(*ROWS, strict=True)
    return fairness.evaluate(y_true, y_pred, gender, SHARE_WOMEN, **kwargs)


def test_accuracy_is_the_share_of_bios_with_the_right_occupation():
    # Correct: nurse 3 + 1, surgeon 1 + 4 = 9 of 12.
    assert _report()["accuracy"]["value"] == pytest.approx(9 / 12)


def test_macro_f1_averages_the_f1_of_each_occupation():
    # F1 = 2 TP / (2 TP + FP + FN).
    # nurse: TP 4, FP 1 (a female surgeon), FN 2 (a female and a male nurse) -> 8 / 11.
    # surgeon: TP 5, FP 2, FN 1 -> 10 / 13.
    assert _report()["macro_f1"]["value"] == pytest.approx((8 / 11 + 10 / 13) / 2)


def test_tpr_is_the_share_of_true_bios_of_an_occupation_predicted_as_it():
    per_occupation = _report()["per_occupation"]
    # nurse: 3 of 4 women, 1 of 2 men. surgeon: 1 of 2 women, 4 of 4 men.
    assert per_occupation[N]["tpr_female"]["value"] == pytest.approx(3 / 4)
    assert per_occupation[N]["tpr_male"]["value"] == pytest.approx(1 / 2)
    assert per_occupation[S]["tpr_female"]["value"] == pytest.approx(1 / 2)
    assert per_occupation[S]["tpr_male"]["value"] == pytest.approx(4 / 4)


def test_per_occupation_counts_the_true_bios_of_each_gender():
    per_occupation = _report()["per_occupation"]
    assert [per_occupation[N]["n_female"], per_occupation[N]["n_male"]] == [4, 2]
    assert [per_occupation[S]["n_female"], per_occupation[S]["n_male"]] == [2, 4]


def test_tpr_gap_is_the_female_tpr_minus_the_male_tpr():
    per_occupation = _report()["per_occupation"]
    assert per_occupation[N]["tpr_gap"]["value"] == pytest.approx(3 / 4 - 1 / 2)
    assert per_occupation[S]["tpr_gap"]["value"] == pytest.approx(1 / 2 - 4 / 4)
