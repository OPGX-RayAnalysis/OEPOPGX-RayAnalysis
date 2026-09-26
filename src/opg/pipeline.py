"""End-to-end inference: OPG image -> FindingsReport + annotated image (plan task 2.8).

Model conventions:
- Tooth model: class names are FDI numbers as strings ("11" ... "48").
- Findings model: class names are labels from opg.schema.FINDING_LABELS.
Both are produced by scripts/dentex_to_yolo.py + scripts/train_yolo.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import preprocess as P
from .associate import assign_findings
from .postprocess import RawTooth, apply_anatomy_rules, summarize
from .schema import Finding, FindingsReport, ImageInfo


def _detect(model, img: np.ndarray, conf: float, offset: tuple[int, int]):
    """Run an Ultralytics model and yield (class_name, box_in_original_px, confidence)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = model.predict(rgb, conf=conf, verbose=False)[0]
    dx, dy = offset
    for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = box
        yield result.names[int(cls)], [x1 + dx, y1 + dy, x2 + dx, y2 + dy], float(score)


def analyze(
    image_path: str | Path,
    teeth_weights: str | Path,
    findings_weights: str | Path | None = None,
    conf: float = 0.25,
    use_clahe: bool = True,
) -> FindingsReport:
    from ultralytics import YOLO  # imported lazily so the rest of the package works without torch

    original = P.load_gray(image_path)
    pre = P.preprocess(image_path, use_clahe=use_clahe)

    raw = [RawTooth(int(name), box, score) for name, box, score in _detect(YOLO(str(teeth_weights)), pre["image"], conf, pre["offset"])]
    teeth = apply_anatomy_rules(raw)

    findings: list[Finding] = []
    if findings_weights:
        findings = [
            Finding(label=name, bbox=box, confidence=score)
            for name, box, score in _detect(YOLO(str(findings_weights)), pre["image"], conf, pre["offset"])
        ]
    unassigned = assign_findings(teeth, findings)

    return FindingsReport(
        image=ImageInfo(file=Path(image_path).name, width=original.shape[1], height=original.shape[0]),
        teeth=teeth,
        summary=summarize(teeth, unassigned),
        unassigned_findings=unassigned,
        models={"teeth": Path(teeth_weights).name, "findings": Path(findings_weights).name if findings_weights else ""},
    )


def draw(image_path: str | Path, report: FindingsReport, out_path: str | Path) -> None:
    """Save an annotated copy of the OPG: tooth boxes with FDI numbers, findings in red."""
    img = cv2.imread(str(image_path))
    for t in report.teeth:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = (0, 200, 255) if t.findings else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(t.fdi), (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for f in [f for t in report.teeth for f in t.findings] + report.unassigned_findings:
        x1, y1, x2, y2 = map(int, f.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f.label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)
