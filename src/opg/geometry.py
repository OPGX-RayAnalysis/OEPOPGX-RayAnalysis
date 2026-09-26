"""Box helpers. Boxes are [x1, y1, x2, y2] in pixels."""

from __future__ import annotations


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b) -> float:
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center_x(b) -> float:
    return (b[0] + b[2]) / 2


def width(b) -> float:
    return b[2] - b[0]


def xywh_to_xyxy(b) -> list[float]:
    """COCO [x, y, w, h] -> [x1, y1, x2, y2]."""
    x, y, w, h = b
    return [x, y, x + w, y + h]
