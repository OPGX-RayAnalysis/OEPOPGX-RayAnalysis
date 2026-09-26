"""Tooth numbering: FDI (ISO 3950) <-> Universal <-> Palmer, plus tooth names.

FDI is the internal standard everywhere in this project. Convert only at the edges
(reports, UI, Q&A), so every model, dataset and JSON file speaks FDI.

FDI: two digits, quadrant then position from the midline.
    Permanent quadrants 1-4 (UR, UL, LL, LR), positions 1-8.
    Primary quadrants 5-8 (UR, UL, LL, LR), positions 1-5.
Universal: permanent 1-32 starting at the upper-right third molar, clockwise
    from the dentist's view; primary teeth are letters A-T in the same order.
Palmer: quadrant (UR/UL/LL/LR) + position 1-8, or A-E for primary teeth.

On an OPG the patient's RIGHT side is on the LEFT of the image.
"""

from __future__ import annotations

PERMANENT_QUADRANTS = (1, 2, 3, 4)
PRIMARY_QUADRANTS = (5, 6, 7, 8)

QUADRANT_CODE = {1: "UR", 2: "UL", 3: "LL", 4: "LR", 5: "UR", 6: "UL", 7: "LL", 8: "LR"}
QUADRANT_NAME = {
    "UR": "upper right",
    "UL": "upper left",
    "LL": "lower left",
    "LR": "lower right",
}

PERMANENT_TOOTH_NAMES = {
    1: "central incisor",
    2: "lateral incisor",
    3: "canine",
    4: "first premolar",
    5: "second premolar",
    6: "first molar",
    7: "second molar",
    8: "third molar (wisdom tooth)",
}
PRIMARY_TOOTH_NAMES = {
    1: "primary central incisor",
    2: "primary lateral incisor",
    3: "primary canine",
    4: "primary first molar",
    5: "primary second molar",
}

ALL_PERMANENT = tuple(q * 10 + n for q in PERMANENT_QUADRANTS for n in range(1, 9))
ALL_PRIMARY = tuple(q * 10 + n for q in PRIMARY_QUADRANTS for n in range(1, 6))
WISDOM_TEETH = (18, 28, 38, 48)

_UNIVERSAL_LETTERS = "ABCDEFGHIJKLMNOPQRST"


def split(fdi: int) -> tuple[int, int]:
    """Return (quadrant, position) and validate the FDI number."""
    q, n = divmod(int(fdi), 10)
    if q in PERMANENT_QUADRANTS and 1 <= n <= 8:
        return q, n
    if q in PRIMARY_QUADRANTS and 1 <= n <= 5:
        return q, n
    raise ValueError(f"Not a valid FDI tooth number: {fdi}")


def is_primary(fdi: int) -> bool:
    return split(fdi)[0] in PRIMARY_QUADRANTS


def is_upper(fdi: int) -> bool:
    return QUADRANT_CODE[split(fdi)[0]] in ("UR", "UL")


def is_patient_right(fdi: int) -> bool:
    return QUADRANT_CODE[split(fdi)[0]] in ("UR", "LR")


def to_universal(fdi: int) -> str:
    q, n = split(fdi)
    if q == 1:
        return str(9 - n)
    if q == 2:
        return str(8 + n)
    if q == 3:
        return str(25 - n)
    if q == 4:
        return str(24 + n)
    # Primary teeth: A (55) ... J (65), K (75) ... T (85)
    index = {5: 5 - n, 6: 4 + n, 7: 15 - n, 8: 14 + n}[q]
    return _UNIVERSAL_LETTERS[index]


def from_universal(code: str | int) -> int:
    code = str(code).strip().upper()
    if code.isdigit():
        u = int(code)
        if not 1 <= u <= 32:
            raise ValueError(f"Universal number out of range: {code}")
        if u <= 8:
            return 10 + (9 - u)
        if u <= 16:
            return 20 + (u - 8)
        if u <= 24:
            return 30 + (25 - u)
        return 40 + (u - 24)
    if len(code) == 1 and code in _UNIVERSAL_LETTERS:
        i = _UNIVERSAL_LETTERS.index(code)
        if i <= 4:
            return 50 + (5 - i)
        if i <= 9:
            return 60 + (i - 4)
        if i <= 14:
            return 70 + (15 - i)
        return 80 + (i - 14)
    raise ValueError(f"Not a valid Universal tooth code: {code}")


def to_palmer(fdi: int) -> str:
    q, n = split(fdi)
    position = "ABCDE"[n - 1] if q in PRIMARY_QUADRANTS else str(n)
    return f"{QUADRANT_CODE[q]}{position}"


def from_palmer(code: str) -> int:
    code = code.strip().upper()
    quad, position = code[:2], code[2:]
    base = {"UR": 1, "UL": 2, "LL": 3, "LR": 4}.get(quad)
    if base is None or len(position) != 1:
        raise ValueError(f"Not a valid Palmer code: {code}")
    if position.isdigit() and 1 <= int(position) <= 8:
        return base * 10 + int(position)
    if position in "ABCDE":
        return (base + 4) * 10 + "ABCDE".index(position) + 1
    raise ValueError(f"Not a valid Palmer code: {code}")


def convert(fdi: int, system: str = "fdi") -> str:
    """Format an FDI tooth number in the requested system: fdi, universal or palmer."""
    system = system.lower()
    if system == "fdi":
        split(fdi)
        return str(fdi)
    if system == "universal":
        return to_universal(fdi)
    if system == "palmer":
        return to_palmer(fdi)
    raise ValueError(f"Unknown numbering system: {system}")


def describe(fdi: int) -> str:
    """Human-readable name, e.g. 46 -> 'lower right first molar'."""
    q, n = split(fdi)
    names = PRIMARY_TOOTH_NAMES if q in PRIMARY_QUADRANTS else PERMANENT_TOOTH_NAMES
    return f"{QUADRANT_NAME[QUADRANT_CODE[q]]} {names[n]}"
