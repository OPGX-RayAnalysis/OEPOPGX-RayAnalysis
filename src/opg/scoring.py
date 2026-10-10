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

The Scorer also keeps one row of counts per image (Scorer.images). From those rows,
summarize() recomputes the first four numbers on any subset of images, bootstrap_ci() gives
a 95% confidence interval by resampling images (task 2.13), and paired_difference_ci() says
whether two models really differ on the same images. mAP50 has no interval: it needs every
raw prediction, and recomputing it 1,000 times is too slow.
"""

from __future__ import annotations

import random
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
    images: list = field(default_factory=list)  # one row of counts per image, see IMAGE_FIELDS
    n_images: int = 0

    def add(self, truth: list[Tooth], pred: list[Tooth], pred_all: list[Tooth] | None = None, name: str = "") -> None:
        """pred: the predictions being scored. pred_all: low-threshold raw predictions for mAP
        (defaults to pred). name: the image file name, kept in the per-image row."""
        self.n_images += 1
        image = {"image": name or str(self.n_images), "teeth": len(truth), "detected": 0, "correct": 0}
        pairs = dict(match(truth, pred, self.iou_threshold))
        for i, t in enumerate(truth):
            row = self.per_tooth[t.fdi]
            row["count"] += 1
            if i in pairs:
                ok = pred[pairs[i]].fdi == t.fdi
                row["detected"] += 1
                row["correct"] += ok
                image["detected"] += 1
                image["correct"] += ok

        truth_present = {t.fdi for t in truth}
        pred_present = {p.fdi for p in pred}
        missing = {"tp": 0, "fp": 0, "fn": 0}
        for tooth in F.ALL_PERMANENT:
            truly_missing, said_missing = tooth not in truth_present, tooth not in pred_present
            if truly_missing and said_missing:
                missing["tp"] += 1
            elif said_missing:
                missing["fp"] += 1
            elif truly_missing:
                missing["fn"] += 1
        for k, v in missing.items():
            self.missing[k] += v
            image[f"missing_{k}"] = v
        self.images.append(image)

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


IMAGE_FIELDS = ["image", "teeth", "detected", "correct", "missing_tp", "missing_fp", "missing_fn"]
CI_METRICS = ["numbering_accuracy", "accuracy_if_found", "detection_rate", "missing_tooth_f1"]


def summarize(images: list[dict]) -> dict:
    """The four per-image-additive metrics (CI_METRICS) from per-image rows."""
    total = {k: sum(r[k] for r in images) for k in IMAGE_FIELDS[1:]}
    tp, fp, fn = total["missing_tp"], total["missing_fp"], total["missing_fn"]
    return {
        "images": len(images),
        "teeth": total["teeth"],
        "numbering_accuracy": _ratio(total["correct"], total["teeth"]),
        "accuracy_if_found": _ratio(total["correct"], total["detected"]),
        "detection_rate": _ratio(total["detected"], total["teeth"]),
        "missing_tooth_f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _percentiles(values: list[float], level: float) -> list[float] | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    tail = (1 - level) / 2
    lo = values[int(tail * (len(values) - 1))]
    hi = values[int(round((1 - tail) * (len(values) - 1)))]
    return [lo, hi]


def bootstrap_ci(images: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap over images: draw len(images) images with replacement, n times,
    recompute each metric, take the middle `level` of the results. {metric: [low, high]}."""
    rng = random.Random(seed)
    draws = [summarize(rng.choices(images, k=len(images))) for _ in range(n)]
    return {m: _percentiles([d[m] for d in draws], level) for m in CI_METRICS}


def paired_difference_ci(
    a: list[dict], b: list[dict], n: int = 1000, level: float = 0.95, seed: int = 0
) -> dict:
    """B minus A on the images both were scored on, resampling the same images for both.
    {metric: {"diff": observed, "ci": [low, high]}}. If the interval contains 0, the two
    models are not shown to differ."""
    a_by, b_by = {r["image"]: r for r in a}, {r["image"]: r for r in b}
    names = sorted(a_by.keys() & b_by.keys())
    if not names:
        raise ValueError("the two runs share no image names")

    def diff(sample: list[str]) -> dict:
        sa, sb = summarize([a_by[x] for x in sample]), summarize([b_by[x] for x in sample])
        return {m: None if sa[m] is None or sb[m] is None else sb[m] - sa[m] for m in CI_METRICS}

    rng = random.Random(seed)
    observed = diff(names)
    draws = [diff(rng.choices(names, k=len(names))) for _ in range(n)]
    return {
        "images": len(names),
        **{m: {"diff": observed[m], "ci": _percentiles([d[m] for d in draws], level)} for m in CI_METRICS},
    }
