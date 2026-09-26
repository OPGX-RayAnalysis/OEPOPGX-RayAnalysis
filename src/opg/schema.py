"""Findings JSON: the one file every pipeline stage after detection reads (plan task 1.5).

Changing this schema needs the lead's sign-off because every stage depends on it.
Bump SCHEMA_VERSION on any breaking change.

Coordinates are pixels in the ORIGINAL image, boxes are [x1, y1, x2, y2].
Tooth numbers are always FDI (see opg.fdi for conversions).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "1.0"

# Finding labels the pipeline may emit. Detectors must map their class names onto these.
FINDING_LABELS = (
    # DENTEX diagnoses
    "caries",
    "deep_caries",
    "periapical_lesion",
    "impacted",
    # Restorations / treatments (Zhou et al., Six-class, etc.)
    "crown",
    "filling",
    "root_canal_filling",
    "implant",
    "residual_root",
    "bridge",
    # Other
    "fracture",
    "bone_loss",
)

Box = list[float]  # [x1, y1, x2, y2]


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


@dataclass
class Tooth:
    fdi: int
    bbox: Box
    confidence: float
    findings: list[Finding] = field(default_factory=list)
    renumbered_from: int | None = None  # set when post-processing changed the model's number


@dataclass
class ImageInfo:
    file: str
    width: int
    height: int


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
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Schema version {d.get('schema_version')} != {SCHEMA_VERSION}")
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
