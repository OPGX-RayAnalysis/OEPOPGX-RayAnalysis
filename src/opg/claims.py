"""Claims and the hallucination log (task 2.18). Pure functions, no model needed.

A claim is one statement about one tooth (or one finding on a tooth), the unit a report is
made of. Hallucination rate = unsupported claims / claims made. Two layers:

    detector layer  the model's output vs the expert labels (measurable now)
    text layer      a report's claims vs the Findings JSON (the LLM's fault) and vs the labels
                    (end to end); from January, once the report generator exists (task 3.3)

Detector layer, numbering. For every permanent tooth number N (and any other number the
model used), the output says either "N present" or "N missing", as a report would:
    "N present" is supported if a box the model numbered N overlaps the real tooth N
    (IoU >= 0.5). If not, the model's most confident N box decides the error:
        wrong_side      it sits on the same tooth on the other side (36 called 46)
        wrong_number    it sits on another real tooth
        invented_tooth  it sits on no real tooth
    "N missing" is supported if tooth N is not in the labels, else: negation_flip.
A second box with the same number does not add a claim: a report says "36 present" once.

Detector layer, findings. Every predicted finding is a claim "this tooth has <finding>"
(two boxes of one class on the same tooth = one claim). Supported if it overlaps a real
finding of the same class, else:
    severity_inflation  deep_caries said where the label is caries
    wrong_finding       it overlaps a real finding of another class
    invented_finding    it overlaps no real finding
Real findings no prediction of their class overlaps are omissions: not claims made, so not
in the rate, but counted (omission rate = omissions / real findings).

Balel 2026's four groups are given beside ours: true = supported; false = wrong number, side,
finding or severity; hallucination = invented tooth or finding; missing = negation flips and
omissions (something real left out or denied).
"""

from __future__ import annotations

import random

from . import fdi as F
from .geometry import iou

CLAIM_FIELDS = ["image", "claim", "about", "supported", "error"]
ERRORS = [
    "invented_tooth", "wrong_number", "wrong_side", "negation_flip",
    "invented_finding", "wrong_finding", "severity_inflation",
]
BALEL = {
    "false": ["wrong_number", "wrong_side", "wrong_finding", "severity_inflation"],
    "hallucination": ["invented_tooth", "invented_finding"],
}


def mirror(fdi: int) -> int:
    """The same tooth on the other side (left <-> right): 36 -> 46, 14 -> 24, 85 -> 75."""
    quadrant, position = divmod(fdi, 10)
    other = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}[quadrant]
    return other * 10 + position


def _best_overlap(box, truth):
    best, best_t = 0.0, None
    for t in truth:
        overlap = iou(box, t.bbox)
        if overlap > best:
            best, best_t = overlap, t
    return best, best_t


def numbering_claims(truth: list, pred: list, image: str = "", iou_threshold: float = 0.5) -> list[dict]:
    """truth, pred: objects with .fdi, .bbox (pred also .conf), e.g. scoring.Tooth."""
    rows = []
    real = {t.fdi for t in truth}
    for n in sorted(set(F.ALL_PERMANENT) | {p.fdi for p in pred}):
        said = [p for p in pred if p.fdi == n]
        if not said:
            ok = n not in real
            rows.append({"image": image, "claim": "missing", "about": n, "supported": int(ok),
                         "error": "" if ok else "negation_flip"})
            continue
        if any(iou(p.bbox, t.bbox) >= iou_threshold for p in said for t in truth if t.fdi == n):
            rows.append({"image": image, "claim": "present", "about": n, "supported": 1, "error": ""})
            continue
        overlap, t = _best_overlap(max(said, key=lambda p: p.conf).bbox, truth)
        if overlap < iou_threshold:
            error = "invented_tooth"
        else:
            error = "wrong_side" if t.fdi == mirror(n) else "wrong_number"
        rows.append({"image": image, "claim": "present", "about": n, "supported": 0, "error": error})
    return rows


def findings_claims(truth: list, pred: list, image: str = "", iou_threshold: float = 0.5) -> list[dict]:
    """truth, pred: objects with .name, .bbox (pred also .conf), e.g. scoring_findings.Finding.
    Pass them after any class merge and filtering, as the scorer scores them."""
    said = []
    for p in sorted(pred, key=lambda p: -p.conf):
        if not any(q.name == p.name and iou(q.bbox, p.bbox) >= iou_threshold for q in said):
            said.append(p)
    rows = []
    for p in said:
        same = [t for t in truth if t.name == p.name]
        if _best_overlap(p.bbox, same)[0] >= iou_threshold:
            rows.append({"image": image, "claim": "finding", "about": p.name, "supported": 1, "error": ""})
            continue
        overlap, t = _best_overlap(p.bbox, truth)
        if overlap < iou_threshold:
            error = "invented_finding"
        elif p.name == "deep_caries" and t.name == "caries":
            error = "severity_inflation"
        else:
            error = "wrong_finding"
        rows.append({"image": image, "claim": "finding", "about": p.name, "supported": 0, "error": error})
    for t in truth:
        if not any(p.name == t.name and iou(p.bbox, t.bbox) >= iou_threshold for p in said):
            rows.append({"image": image, "claim": "omitted", "about": t.name, "supported": 0, "error": "omission"})
    return rows


def summarize_claims(rows: list[dict]) -> dict:
    """Claims made, unsupported, rate, count per error type, omissions and Balel's four groups."""
    made = [r for r in rows if r["claim"] != "omitted"]
    omitted = len(rows) - len(made)
    real_findings = omitted + sum(r["supported"] for r in made if r["claim"] == "finding")
    errors = {e: sum(r["error"] == e for r in made) for e in ERRORS}
    unsupported = sum(not r["supported"] for r in made)
    return {
        "claims": len(made),
        "unsupported": unsupported,
        "rate": unsupported / len(made) if made else None,
        "errors": errors,
        "omissions": omitted,
        "omission_rate": omitted / real_findings if real_findings else None,
        "balel": {
            "true": len(made) - unsupported,
            "false": sum(errors[e] for e in BALEL["false"]),
            "hallucination": sum(errors[e] for e in BALEL["hallucination"]),
            "missing": errors["negation_flip"] + omitted,
        },
    }


def bootstrap_rate_ci(rows: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0) -> list[float] | None:
    """95% interval of the hallucination rate, resampling images (one patient = one image in
    DENTEX; group by patient instead where a dataset has several images per patient)."""
    by_image: dict = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)
    names = sorted(by_image)
    if not names:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(n):
        rate = summarize_claims([r for x in rng.choices(names, k=len(names)) for r in by_image[x]])["rate"]
        if rate is not None:
            rates.append(rate)
    if not rates:
        return None
    rates.sort()
    tail = (1 - level) / 2
    return [rates[int(tail * (len(rates) - 1))], rates[int(round((1 - tail) * (len(rates) - 1)))]]


def paired_rate_difference(a: list[dict], b: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0) -> dict:
    """B's hallucination rate minus A's on the images both were scored on, resampling the same
    images for both. {"images", "diff", "ci": [low, high]}; an interval containing 0 = not shown
    to differ."""
    a_by, b_by = {}, {}
    for rows, by in ((a, a_by), (b, b_by)):
        for r in rows:
            by.setdefault(r["image"], []).append(r)
    names = sorted(a_by.keys() & b_by.keys())
    if not names:
        raise ValueError("the two runs share no image names")

    def diff(sample: list[str]) -> float | None:
        ra = summarize_claims([r for x in sample for r in a_by[x]])["rate"]
        rb = summarize_claims([r for x in sample for r in b_by[x]])["rate"]
        return None if ra is None or rb is None else rb - ra

    rng = random.Random(seed)
    draws = sorted(d for d in (diff(rng.choices(names, k=len(names))) for _ in range(n)) if d is not None)
    tail = (1 - level) / 2
    ci = [draws[int(tail * (len(draws) - 1))], draws[int(round((1 - tail) * (len(draws) - 1)))]] if draws else None
    return {"images": len(names), "diff": diff(names), "ci": ci}
