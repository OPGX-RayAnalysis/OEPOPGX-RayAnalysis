import pytest

from opg import fdi as F


@pytest.mark.parametrize(
    "fdi, universal, palmer",
    [
        (18, "1", "UR8"),
        (11, "8", "UR1"),
        (21, "9", "UL1"),
        (28, "16", "UL8"),
        (38, "17", "LL8"),
        (31, "24", "LL1"),
        (41, "25", "LR1"),
        (46, "30", "LR6"),  # textbook example: FDI 46 = Universal 30 = Palmer LR6
        (48, "32", "LR8"),
        (55, "A", "URE"),
        (65, "J", "ULE"),
        (75, "K", "LLE"),
        (85, "T", "LRE"),
        (51, "E", "URA"),
    ],
)
def test_known_conversions(fdi, universal, palmer):
    assert F.to_universal(fdi) == universal
    assert F.to_palmer(fdi) == palmer
    assert F.from_universal(universal) == fdi
    assert F.from_palmer(palmer) == fdi


def test_round_trip_all_teeth():
    for tooth in F.ALL_PERMANENT + F.ALL_PRIMARY:
        assert F.from_universal(F.to_universal(tooth)) == tooth
        assert F.from_palmer(F.to_palmer(tooth)) == tooth


def test_universal_covers_1_to_32_and_a_to_t_once():
    assert sorted(int(F.to_universal(t)) for t in F.ALL_PERMANENT) == list(range(1, 33))
    assert sorted(F.to_universal(t) for t in F.ALL_PRIMARY) == list("ABCDEFGHIJKLMNOPQRST")


@pytest.mark.parametrize("bad", [0, 10, 19, 49, 56, 90, 91])
def test_invalid_numbers_rejected(bad):
    with pytest.raises(ValueError):
        F.split(bad)


def test_describe_and_sides():
    assert F.describe(46) == "lower right first molar"
    assert F.describe(28) == "upper left third molar (wisdom tooth)"
    assert F.is_upper(17) and not F.is_upper(37)
    assert F.is_patient_right(45) and not F.is_patient_right(35)
