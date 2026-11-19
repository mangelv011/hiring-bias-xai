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
