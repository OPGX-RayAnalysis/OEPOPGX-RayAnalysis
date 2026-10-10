import pytest
from opg import fdi as F

# Independent chart for all 32 permanent teeth.
# (FDI, Universal, Palmer, name)
# Source chart: https://www.dentclosers.com/tooth-chart
CHART = [
    (18, "1", "UR8", "upper right third molar (wisdom tooth)"),
    (17, "2", "UR7", "upper right second molar"),
    (16, "3", "UR6", "upper right first molar"),
    (15, "4", "UR5", "upper right second premolar"),
    (14, "5", "UR4", "upper right first premolar"),
    (13, "6", "UR3", "upper right canine"),
    (12, "7", "UR2", "upper right lateral incisor"),
    (11, "8", "UR1", "upper right central incisor"),

    (21, "9", "UL1", "upper left central incisor"),
    (22, "10", "UL2", "upper left lateral incisor"),
    (23, "11", "UL3", "upper left canine"),
    (24, "12", "UL4", "upper left first premolar"),
    (25, "13", "UL5", "upper left second premolar"),
    (26, "14", "UL6", "upper left first molar"),
    (27, "15", "UL7", "upper left second molar"),
    (28, "16", "UL8", "upper left third molar (wisdom tooth)"),

    (38, "17", "LL8", "lower left third molar (wisdom tooth)"),
    (37, "18", "LL7", "lower left second molar"),
    (36, "19", "LL6", "lower left first molar"),
    (35, "20", "LL5", "lower left second premolar"),
    (34, "21", "LL4", "lower left first premolar"),
    (33, "22", "LL3", "lower left canine"),
    (32, "23", "LL2", "lower left lateral incisor"),
    (31, "24", "LL1", "lower left central incisor"),

    (41, "25", "LR1", "lower right central incisor"),
    (42, "26", "LR2", "lower right lateral incisor"),
    (43, "27", "LR3", "lower right canine"),
    (44, "28", "LR4", "lower right first premolar"),
    (45, "29", "LR5", "lower right second premolar"),
    (46, "30", "LR6", "lower right first molar"),
    (47, "31", "LR7", "lower right second molar"),
    (48, "32", "LR8", "lower right third molar (wisdom tooth)"),
]


def test_chart_has_all_32_teeth():
    assert len(CHART) == 32
    assert len({row[0] for row in CHART}) == 32


@pytest.mark.parametrize("fdi, universal, palmer, name", CHART)
def test_fdi_matches_chart(fdi, universal, palmer, name):
    assert F.to_universal(fdi) == universal
    assert F.to_palmer(fdi) == palmer
    assert F.from_universal(universal) == fdi
    assert F.from_palmer(palmer) == fdi
    assert F.describe(fdi) == name