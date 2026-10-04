"""Score tooth numbering against labels (plan task 1.11). Pure functions, no model needed.

Each image gives real teeth (FDI, box) and predicted teeth (FDI, box, confidence).
Predicted boxes are matched to real teeth one-to-one by overlap (IoU >= 0.5), ignoring
the number. Then, over all real teeth:

    detection rate       matched / all real teeth
    numbering accuracy   matched AND right number / all real teeth   (the headline number)
    accuracy if found    right number / matched                      (numbering alone)
    missing-tooth F1     per image, each of the 32 permanent teeth is "missing" or not, in the
                         labels and in the predictions; F1 of the "missing" class
    mAP50                class-aware average precision at IoU 0.5 (all-point interpolation),
                         a little above what Ultralytics prints (its 101-point method gave
                         30.0% where this gave 32.2% on the 5-epoch smoke model); compare
                         mAP50 only between runs scored by this script
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import fdi as F
from .geometry import iou


@dataclass
class Tooth:
    fdi: int
    bbox: list[float]  # x1, y1, x2, y2 in pixels
    conf: float = 1.0


def match(truth: list[Tooth], pred: list[Tooth], iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one (truth index, pred index) pairs, best overlap first, number ignored."""
    pairs = sorted(
        ((iou(t.bbox, p.bbox), i, j) for i, t in enumerate(truth) for j, p in enumerate(pred)),
        reverse=True,
    )
    used_t, used_p, out = set(), set(), []
    for overlap, i, j in pairs:
        if overlap < iou_threshold:
            break
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def average_precision(hits: list[tuple[float, bool]], n_truth: int) -> float:
    """AP from (confidence, is_true_positive) pairs, all-point interpolation."""
    if n_truth == 0:
        return float("nan")
    hits = sorted(hits, key=lambda h: -h[0])
    tp = fp = 0
    points = []  # (recall, precision)
    for _, ok in hits:
        tp, fp = tp + ok, fp + (not ok)
        points.append((tp / n_truth, tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for k, (recall, _) in enumerate(points):
        best_precision = max(p for _, p in points[k:])
        ap += (recall - prev_recall) * best_precision
        prev_recall = recall
    return ap


@dataclass
class Scorer:
    """Feed it images one at a time with add(), then call result()."""

    iou_threshold: float = 0.5
    per_tooth: dict = field(default_factory=lambda: defaultdict(lambda: {"count": 0, "detected": 0, "correct": 0}))
    missing: dict = field(default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0})
    ap_hits: dict = field(default_factory=lambda: defaultdict(list))
    n_images: int = 0

    def add(self, truth: list[Tooth], pred: list[Tooth], pred_all: list[Tooth] | None = None) -> None:
        """pred: the predictions being scored. pred_all: low-threshold raw predictions for mAP
        (defaults to pred)."""
        self.n_images += 1
        pairs = dict(match(truth, pred, self.iou_threshold))
        for i, t in enumerate(truth):
            row = self.per_tooth[t.fdi]
            row["count"] += 1
            if i in pairs:
                row["detected"] += 1
                row["correct"] += pred[pairs[i]].fdi == t.fdi

        truth_present = {t.fdi for t in truth}
        pred_present = {p.fdi for p in pred}
        for tooth in F.ALL_PERMANENT:
            truly_missing, said_missing = tooth not in truth_present, tooth not in pred_present
            if truly_missing and said_missing:
                self.missing["tp"] += 1
            elif said_missing:
                self.missing["fp"] += 1
            elif truly_missing:
                self.missing["fn"] += 1

        self._add_ap(truth, pred if pred_all is None else pred_all)

    def _add_ap(self, truth: list[Tooth], pred: list[Tooth]) -> None:
        # Standard detection matching per class: confident predictions claim truths first.
        for tooth in {t.fdi for t in truth} | {p.fdi for p in pred}:
            truths = [t for t in truth if t.fdi == tooth]
            claimed = set()
            for p in sorted((p for p in pred if p.fdi == tooth), key=lambda p: -p.conf):
                best, best_i = 0.0, None
                for i, t in enumerate(truths):
                    overlap = iou(t.bbox, p.bbox)
                    if i not in claimed and overlap > best:
                        best, best_i = overlap, i
                ok = best_i is not None and best >= self.iou_threshold
                if ok:
                    claimed.add(best_i)
                self.ap_hits[tooth].append((p.conf, ok))

    def result(self) -> dict:
        rows = []
        for tooth in F.ALL_PERMANENT:
            r = self.per_tooth.get(tooth, {"count": 0, "detected": 0, "correct": 0})
            rows.append({"fdi": tooth, **r, "accuracy": _ratio(r["correct"], r["count"])})
        total = sum(r["count"] for r in self.per_tooth.values())
        detected = sum(r["detected"] for r in self.per_tooth.values())
        correct = sum(r["correct"] for r in self.per_tooth.values())
        m = self.missing
        aps = [
            average_precision(self.ap_hits[t], self.per_tooth[t]["count"])
            for t in self.per_tooth
            if self.per_tooth[t]["count"]
        ]
        return {
            "images": self.n_images,
            "teeth": total,
            "numbering_accuracy": _ratio(correct, total),
            "accuracy_if_found": _ratio(correct, detected),
            "detection_rate": _ratio(detected, total),
            "missing_tooth_f1": _ratio(2 * m["tp"], 2 * m["tp"] + m["fp"] + m["fn"]),
            "missing_counts": dict(m),
            "map50": sum(aps) / len(aps) if aps else None,
            "per_tooth": rows,
        }


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None
