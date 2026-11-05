# Reference:
# Primary Teeth Numbering Systems Illustrated.
# Dental Numbering System Press, Rev. 903.
# Source chart supplied by the team.

import pytest
from opg import fdi as F

# Independent chart for all 20 primary teeth.
# (FDI, Universal, Palmer, name)
# Source chart: transcribe from the university dental anatomy reference.
CHART = [
    # Upper right
    (51, "E", "URA", "upper right primary central incisor"),
    (52, "D", "URB", "upper right primary lateral incisor"),
    (53, "C", "URC", "upper right primary canine"),
    (54, "B", "URD", "upper right primary first molar"),
    (55, "A", "URE", "upper right primary second molar"),

    # Upper left
    (61, "F", "ULA", "upper left primary central incisor"),
    (62, "G", "ULB", "upper left primary lateral incisor"),
    (63, "H", "ULC", "upper left primary canine"),
    (64, "I", "ULD", "upper left primary first molar"),
    (65, "J", "ULE", "upper left primary second molar"),

    # Lower left
    (71, "O", "LLA", "lower left primary central incisor"),
    (72, "N", "LLB", "lower left primary lateral incisor"),
    (73, "M", "LLC", "lower left primary canine"),
    (74, "L", "LLD", "lower left primary first molar"),
    (75, "K", "LLE", "lower left primary second molar"),

    # Lower right
    (81, "P", "LRA", "lower right primary central incisor"),
    (82, "Q", "LRB", "lower right primary lateral incisor"),
    (83, "R", "LRC", "lower right primary canine"),
    (84, "S", "LRD", "lower right primary first molar"),
    (85, "T", "LRE", "lower right primary second molar"),
]


def test_chart_has_all_20_teeth():
    assert len(CHART) == 20
    assert len({row[0] for row in CHART}) == 20


@pytest.mark.parametrize("fdi, universal, palmer, name", CHART)
def test_fdi_matches_primary_chart(fdi, universal, palmer, name):
    assert F.to_universal(fdi) == universal
    assert F.to_palmer(fdi) == palmer
    assert F.from_universal(universal) == fdi
    assert F.from_palmer(palmer) == fdi
    assert F.describe(fdi) == name
