"""Score a findings detector (caries, lesions, treatments) against labels (task 2.16). Pure functions.

Each image gives real findings (class name, box) and predicted findings (class name, box,
confidence). Classes are compared by NAME, never by YOLO class index, so a model and a
dataset whose data.yaml lists classes in a different order, or only some of them, still
score correctly. Within each class, predictions are matched one-to-one to real findings by
overlap (IoU >= 0.5, best overlap first, scoring.match). Then, per class:

    tp          matched predictions
    fp          predictions matching no real finding of that class (invented findings)
    fn          real findings that no prediction matched (missed findings)
    precision   tp / (tp + fp)
    recall      tp / (tp + fn)
    F1          2 tp / (2 tp + fp + fn)
    AP50        average precision at IoU 0.5 from the low-threshold raw predictions

    mean F1     plain average of the per-class F1s (the headline: every class counts the same,
                however rare)
    pooled      precision, recall and F1 over all findings together (common classes dominate)

match_rule="cover" (--match cover) matches by how much of the REAL box lies inside the predicted
box (geometry.cover >= 0.5) instead of IoU: for datasets that box the lesion while our model
boxes the whole tooth (OralXrays-9), where IoU calls even a right tooth a miss. It asks "did
the model flag the tooth that has it?" Still one-to-one. Report it beside the IoU score.

A prediction with the right box but the wrong class counts twice: an fp for the class it
said and an fn for the class it should have said. `merge` renames classes before anything
else, e.g. {"deep_caries": "caries"} (--merge-caries) when deep vs shallow is not the question.

The scorer keeps one row of counts per image per class (FINDING_IMAGE_FIELDS), so, as in
scoring.py, summarize_findings() recomputes everything on any subset of images,
bootstrap_findings_ci() gives 95% intervals by resampling images, and
paired_findings_difference() compares two models on the same images. It also keeps one row per
scored prediction (FINDING_PRED_FIELDS) for calibration (2.14), and one row per claim
(claims.CLAIM_FIELDS) for the hallucination log (2.18).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from .claims import findings_claims
from .geometry import cover, iou
from .scoring import _percentiles, _ratio, ap_hits, average_precision, match

FINDING_IMAGE_FIELDS = ["image", "finding", "truth", "tp", "fp", "fn"]
FINDING_PRED_FIELDS = ["image", "finding", "confidence", "matched"]
PRF = ["precision", "recall", "f1"]
MERGE_CARIES = {"deep_caries": "caries"}
MATCH_RULES = {"iou": iou, "cover": cover}


def normalize(name: str) -> str:
    """'Deep Caries' / 'deep-caries' / 'deep_caries' -> 'deep_caries'."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class Finding:
    name: str
    bbox: list[float]  # x1, y1, x2, y2 in pixels
    conf: float = 1.0


@dataclass
class FindingScorer:
    """Feed it images one at a time with add(), then call result(). classes: the class names
    to score (after merging); findings of any other class are ignored."""

    classes: list[str]
    iou_threshold: float = 0.5
    merge: dict = field(default_factory=dict)
    match_rule: str = "iou"  # or "cover", see the module docstring
    images: list = field(default_factory=list)  # one row per image per class, see FINDING_IMAGE_FIELDS
    preds: list = field(default_factory=list)  # one row per scored prediction, see FINDING_PRED_FIELDS
    claims: list = field(default_factory=list)  # one row per claim, see claims.CLAIM_FIELDS (task 2.18)
    hits: dict = field(default_factory=lambda: defaultdict(list))
    n_images: int = 0

    def _keep(self, findings: list[Finding]) -> list[Finding]:
        renamed = (Finding(self.merge.get(f.name, f.name), f.bbox, f.conf) for f in findings)
        return [f for f in renamed if f.name in self.classes]

    def add(
        self, truth: list[Finding], pred: list[Finding], pred_all: list[Finding] | None = None, name: str = ""
    ) -> None:
        """pred: the predictions being scored. pred_all: low-threshold raw predictions for AP50
        (defaults to pred). name: the image file name, kept in the per-image rows."""
        self.n_images += 1
        name = name or str(self.n_images)
        truth, pred = self._keep(truth), self._keep(pred)
        pred_all = pred if pred_all is None else self._keep(pred_all)
        overlap = MATCH_RULES[self.match_rule]
        self.claims += findings_claims(truth, pred, name, self.iou_threshold, overlap)
        for c in self.classes:
            t = [f for f in truth if f.name == c]
            p = [f for f in pred if f.name == c]
            matched = {j for _, j in match(t, p, self.iou_threshold, overlap)}
            for j, f in enumerate(p):
                self.preds.append({"image": name, "finding": c, "confidence": round(f.conf, 4), "matched": int(j in matched)})
            self.images.append({
                "image": name, "finding": c, "truth": len(t),
                "tp": len(matched), "fp": len(p) - len(matched), "fn": len(t) - len(matched),
            })
            self.hits[c] += ap_hits(t, [f for f in pred_all if f.name == c], self.iou_threshold, overlap)

    def result(self) -> dict:
        s = summarize_findings(self.images)
        for c, row in s["per_class"].items():
            row["ap50"] = average_precision(self.hits[c], row["truth"]) if row["truth"] else None
        aps = [row["ap50"] for row in s["per_class"].values() if row["ap50"] is not None]
        return {**s, "map50": sum(aps) / len(aps) if aps else None}


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def summarize_findings(rows: list[dict]) -> dict:
    """Per-class counts and precision/recall/F1, mean F1 and pooled numbers from per-image rows."""
    counts: dict = {}
    for r in rows:
        c = counts.setdefault(r["finding"], {"truth": 0, "tp": 0, "fp": 0, "fn": 0})
        for k in c:
            c[k] += r[k]
    per_class = {name: {**c, **_prf(c["tp"], c["fp"], c["fn"])} for name, c in counts.items()}
    f1s = [c["f1"] for c in per_class.values() if c["f1"] is not None]
    total = {k: sum(c[k] for c in counts.values()) for k in ("truth", "tp", "fp", "fn")}
    return {
        "images": len({r["image"] for r in rows}),
        "findings": total["truth"],
        "mean_f1": sum(f1s) / len(f1s) if f1s else None,
        "pooled": {**total, **_prf(total["tp"], total["fp"], total["fn"])},
        "per_class": per_class,
    }


def _by_image(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["image"]].append(r)
    return out


def _resample(by_image: dict, names: list[str], rng: random.Random) -> list[dict]:
    return [r for name in rng.choices(names, k=len(names)) for r in by_image[name]]


def bootstrap_findings_ci(rows: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap over images (all of an image's findings move together).
    {"mean_f1": [lo, hi], "pooled": {metric: [lo, hi]}, "per_class": {class: {metric: [lo, hi]}}}."""
    by_image = _by_image(rows)
    names = sorted(by_image)
    rng = random.Random(seed)
    draws = [summarize_findings(_resample(by_image, names, rng)) for _ in range(n)]
    classes = list(summarize_findings(rows)["per_class"])
    return {
        "mean_f1": _percentiles([d["mean_f1"] for d in draws], level),
        "pooled": {m: _percentiles([d["pooled"][m] for d in draws], level) for m in PRF},
        "per_class": {
            c: {m: _percentiles([d["per_class"][c][m] for d in draws], level) for m in PRF} for c in classes
        },
    }


def _f1s(summary: dict) -> dict:
    return {"mean_f1": summary["mean_f1"], "pooled_f1": summary["pooled"]["f1"],
            **{f"{c}_f1": v["f1"] for c, v in summary["per_class"].items()}}


def paired_findings_difference(
    a: list[dict], b: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0
) -> dict:
    """B minus A in mean F1, pooled F1 and each shared class's F1 (keys "<class>_f1"), on the
    images both were scored on, resampling the same images for both.
    {"images": n, metric: {"diff": observed, "ci": [low, high]}}."""
    a_by, b_by = _by_image(a), _by_image(b)
    names = sorted(a_by.keys() & b_by.keys())
    if not names:
        raise ValueError("the two runs share no image names")
    shared = {r["finding"] for r in a} & {r["finding"] for r in b}
    a_by = {k: [r for r in v if r["finding"] in shared] for k, v in a_by.items()}
    b_by = {k: [r for r in v if r["finding"] in shared] for k, v in b_by.items()}

    def diff(sample: list[str]) -> dict:
        fa = _f1s(summarize_findings([r for x in sample for r in a_by[x]]))
        fb = _f1s(summarize_findings([r for x in sample for r in b_by[x]]))
        return {m: None if fa.get(m) is None or fb.get(m) is None else fb[m] - fa[m] for m in fa}

    rng = random.Random(seed)
    observed = diff(names)
    draws = [diff(rng.choices(names, k=len(names))) for _ in range(n)]
    return {
        "images": len(names),
        **{m: {"diff": observed[m], "ci": _percentiles([d.get(m) for d in draws], level)} for m in observed},
    }
