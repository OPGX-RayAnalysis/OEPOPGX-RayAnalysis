"""Findings JSON: the one file every pipeline stage after detection reads (plan task 1.5).

Changing this schema needs the lead's sign-off because every stage depends on it.
Versioning: "<major>.<minor>". Adding an optional field bumps the minor version and old files
still load. Renaming, removing or changing the meaning of a field bumps the major version, and
files from another major version are rejected.

Coordinates are pixels in the ORIGINAL image, boxes are [x1, y1, x2, y2].
Tooth numbers are always FDI (see opg.fdi for conversions).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import fdi as F

SCHEMA_VERSION = "1.0"

# Final finding labels (v1). Detectors and dataset converters must map their class names onto
# these; a dataset class with no match is dropped, not added here without sign-off.
DIAGNOSIS_LABELS = (
    "caries",
    "deep_caries",
    "periapical_lesion",  # also: periapical infection / abscess classes in other datasets
    "impacted",
    "fracture",
    "residual_root",  # also: "broken root" / retained root classes
    "bone_loss",
)
TREATMENT_LABELS = (
    "crown",
    "filling",
    "root_canal_filling",
    "implant",
    "bridge",
)
FINDING_LABELS = DIAGNOSIS_LABELS + TREATMENT_LABELS
# Trained in Phase 2 from DENTEX; the rest need the extra datasets (task 2.2).
DENTEX_LABELS = ("caries", "deep_caries", "periapical_lesion", "impacted")

Box = list[float]  # [x1, y1, x2, y2]


def _check_box(bbox: Box, confidence: float, what: str) -> None:
    if len(bbox) != 4 or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        raise ValueError(f"{what}: bbox must be [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2, got {bbox}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{what}: confidence must be between 0 and 1, got {confidence}")


@dataclass
class Finding:
    label: str
    bbox: Box
    confidence: float
    tooth_fdi: int | None = None  # None = could not be linked to a tooth
    source: str = "findings_model"  # which model or rule produced it

    def __post_init__(self):
        if self.label not in FINDING_LABELS:
            raise ValueError(f"Unknown finding label '{self.label}'. Add it to FINDING_LABELS first.")
        _check_box(self.bbox, self.confidence, f"finding '{self.label}'")
        if self.tooth_fdi is not None:
            F.split(self.tooth_fdi)


@dataclass
class Tooth:
    fdi: int
    bbox: Box
    confidence: float
    findings: list[Finding] = field(default_factory=list)
    renumbered_from: int | None = None  # set when post-processing changed the model's number

    def __post_init__(self):
        F.split(self.fdi)  # raises on anything that isn't a real FDI number
        _check_box(self.bbox, self.confidence, f"tooth {self.fdi}")


@dataclass
class ImageInfo:
    file: str
    width: int
    height: int
    patient_id: str | None = None  # anonymised id; links X-rays of one patient (history, task 3.5)
    study_date: str | None = None  # ISO date the X-ray was taken, e.g. "2026-10-04"


@dataclass
class Summary:
    tooth_count: int
    missing_permanent: list[int]
    wisdom_teeth: dict[str, str]  # "18" -> "present" | "missing" | "impacted"
    dentition: str = "permanent"  # permanent | mixed | primary (stretch goal 3.11)
    finding_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class FindingsReport:
    image: ImageInfo
    teeth: list[Tooth]
    summary: Summary
    unassigned_findings: list[Finding] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)  # e.g. {"teeth": "yolo11s-teeth-v1"}
    schema_version: str = SCHEMA_VERSION

    def tooth(self, fdi: int) -> Tooth | None:
        return next((t for t in self.teeth if t.fdi == fdi), None)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "FindingsReport":
        version = str(d.get("schema_version", ""))
        if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"Schema version {version or 'missing'} is not compatible with {SCHEMA_VERSION}")
        return cls(
            image=ImageInfo(**d["image"]),
            teeth=[
                Tooth(
                    fdi=t["fdi"],
                    bbox=t["bbox"],
                    confidence=t["confidence"],
                    findings=[Finding(**f) for f in t["findings"]],
                    renumbered_from=t.get("renumbered_from"),
                )
                for t in d["teeth"]
            ],
            summary=Summary(**d["summary"]),
            unassigned_findings=[Finding(**f) for f in d.get("unassigned_findings", [])],
            models=d.get("models", {}),
            schema_version=d["schema_version"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "FindingsReport":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
