"""Anatomy rules on top of the raw tooth detector (plan task 2.5).

The detector predicts an FDI number per box, but it can use a number twice, skip one,
or put teeth out of order next to a gap. These rules fix that:

1. Drop near-duplicate boxes in the same quadrant (keep the most confident one).
2. Put each tooth on the correct side of the midline (patient right = image left).
3. Inside a quadrant, teeth sorted outward from the midline must get strictly
   increasing positions 1..8. We choose the ordered assignment that changes confident
   predictions the least (a small dynamic program), so each number is used once and
   gaps become missing teeth.

Only permanent teeth for now; primary/mixed dentition is stretch goal 3.11.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from . import fdi as F
from .geometry import center_x, iou, width
from .schema import Finding, Summary, Tooth


@dataclass
class RawTooth:
    fdi: int  # the detector's predicted FDI number
    bbox: list[float]
    confidence: float


_SIDE_SWAP = {1: 2, 2: 1, 3: 4, 4: 3}


def estimate_midline(teeth: list[RawTooth]) -> float:
    """Image x-coordinate of the dental midline, from the central incisors if found."""
    xs = {t.fdi: center_x(t.bbox) for t in teeth}
    pairs = [(11, 21), (41, 31)]
    mids = [(xs[a] + xs[b]) / 2 for a, b in pairs if a in xs and b in xs]
    if mids:
        return sum(mids) / len(mids)
    return median(center_x(t.bbox) for t in teeth)


def _dedupe(teeth: list[RawTooth], iou_threshold: float) -> list[RawTooth]:
    kept: list[RawTooth] = []
    for t in sorted(teeth, key=lambda t: -t.confidence):
        if all(iou(t.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(t)
    return kept


def _fix_side(t: RawTooth, midline: float, margin: float) -> int:
    q, n = F.split(t.fdi)
    cx = center_x(t.bbox)
    patient_right = q in (1, 4)
    # Patient-right teeth belong on the image left. Only flip when clearly on the wrong side.
    if (patient_right and cx > midline + margin) or (not patient_right and cx < midline - margin):
        q = _SIDE_SWAP[q]
    return q * 10 + n


def _assign_positions(pred: list[int], conf: list[float]) -> list[int]:
    """Strictly increasing positions in 1..8 minimising sum(conf * |pred - pos|)."""
    k, big = len(pred), float("inf")
    # dp[i][j]: best cost placing the first i teeth using positions <= j
    dp = [[0.0] * 9] + [[big] * 9 for _ in range(k)]
    take = [[False] * 9 for _ in range(k + 1)]
    for i in range(1, k + 1):
        for j in range(1, 9):
            skip = dp[i][j - 1]
            use = dp[i - 1][j - 1] + conf[i - 1] * abs(pred[i - 1] - j) + 1e-6 * j
            dp[i][j], take[i][j] = (use, True) if use < skip else (skip, False)
    positions, j = [0] * k, 8
    for i in range(k, 0, -1):
        while not take[i][j]:
            j -= 1
        positions[i - 1] = j
        j -= 1
    return positions


def apply_anatomy_rules(raw: list[RawTooth], dedupe_iou: float = 0.6) -> list[Tooth]:
    """Turn raw permanent-tooth detections into a consistent, uniquely numbered set."""
    raw = [t for t in raw if not F.is_primary(t.fdi)]
    if not raw:
        return []
    midline = estimate_midline(raw)
    margin = 0.5 * median(width(t.bbox) for t in raw)

    by_quadrant: dict[int, list[tuple[RawTooth, int]]] = {1: [], 2: [], 3: [], 4: []}
    for t in raw:
        fixed = _fix_side(t, midline, margin)
        by_quadrant[fixed // 10].append((t, fixed))

    result: list[Tooth] = []
    for q, items in by_quadrant.items():
        kept_raw = _dedupe([t for t, _ in items], dedupe_iou)
        fixed_of = {id(t): f for t, f in items}
        # Keep at most 8 teeth per quadrant, dropping the least confident extras.
        kept_raw = sorted(kept_raw, key=lambda t: -t.confidence)[:8]
        kept_raw.sort(key=lambda t: abs(center_x(t.bbox) - midline))
        preds = [fixed_of[id(t)] % 10 for t in kept_raw]
        positions = _assign_positions(preds, [t.confidence for t in kept_raw])
        for t, pos in zip(kept_raw, positions):
            new = q * 10 + pos
            result.append(
                Tooth(
                    fdi=new,
                    bbox=list(t.bbox),
                    confidence=t.confidence,
                    renumbered_from=t.fdi if new != t.fdi else None,
                )
            )
    return sorted(result, key=lambda t: t.fdi)


def summarize(teeth: list[Tooth], unassigned: list[Finding] | None = None) -> Summary:
    present = {t.fdi for t in teeth}
    wisdom = {}
    for w in F.WISDOM_TEETH:
        tooth = next((t for t in teeth if t.fdi == w), None)
        if tooth is None:
            wisdom[str(w)] = "missing"
        elif any(f.label == "impacted" for f in tooth.findings):
            wisdom[str(w)] = "impacted"
        else:
            wisdom[str(w)] = "present"
    counts: dict[str, int] = {}
    for f in [f for t in teeth for f in t.findings] + list(unassigned or []):
        counts[f.label] = counts.get(f.label, 0) + 1
    return Summary(
        tooth_count=len([t for t in teeth if not F.is_primary(t.fdi)]),
        missing_permanent=sorted(set(F.ALL_PERMANENT) - present),
        wisdom_teeth=wisdom,
        finding_counts=dict(sorted(counts.items())),
    )
