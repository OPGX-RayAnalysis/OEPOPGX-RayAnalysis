"""Link each detected finding to the tooth it belongs to (plan task 2.8).

A finding goes to the tooth that covers the largest share of the finding's box,
as long as that share is at least `min_overlap`. Otherwise it stays unassigned
(e.g. a lesion in the bone, or a tooth the tooth detector missed).
"""

from __future__ import annotations

from .geometry import area, intersection, iou
from .schema import Finding, Tooth


def assign_findings(
    teeth: list[Tooth], findings: list[Finding], min_overlap: float = 0.3
) -> list[Finding]:
    """Attach findings to `teeth` in place. Returns the findings that could not be assigned."""
    unassigned = []
    for f in findings:
        f_area = area(f.bbox) or 1.0
        best, best_key = None, (0.0, 0.0)
        for t in teeth:
            key = (intersection(f.bbox, t.bbox) / f_area, iou(f.bbox, t.bbox))
            if key > best_key:
                best, best_key = t, key
        if best is not None and best_key[0] >= min_overlap:
            f.tooth_fdi = best.fdi
            best.findings.append(f)
        else:
            f.tooth_fdi = None
            unassigned.append(f)
    return unassigned
