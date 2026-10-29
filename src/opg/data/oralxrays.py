"""Read OralXrays-9 (Chen et al., CVPR 2025; the authors' own release) into the record format
of data/dentex.py, so dentex.write_yolo can write it (task 2.17).

Release: annotations/instances_{train2017,val2017}.json (COCO) + train2017/, val2017/ images,
all 1991 x 1127 JPG. Boxes are drawn around the lesion or the restoration itself, not the
whole tooth as in DENTEX (see docs/datasets/oralxrays.md and score_findings.py --match cover).

Terms (authors' written permission, 24 Oct 2026): non-commercial research, cite the CVPR 2025
paper, official split for benchmark comparisons, no redistribution outside the team.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..geometry import xywh_to_xyxy

# Their class names -> ours (schema.FINDING_LABELS). Wisdom Tooth boxes every third molar,
# not only impacted ones, so it is not DENTEX's "impacted"; Missing Tooth is a numbering fact.
CLASS_MAP = {
    "Decay": "caries",
    "Apical Periodontitis": "periapical_lesion",
    "Dental Filling": "filling",
    "Root Canal Filling": "root_canal_filling",
    "Porcelain Crown": "crown",
    "Ceramic Bridge": "bridge",
    "Implant": "implant",
}
DROPPED = {"Wisdom Tooth", "Missing Tooth"}
CLASSES = list(CLASS_MAP.values())  # YOLO class order
SPLITS = {"train2017": "train2017", "val2017": "val2017"}  # release split -> image folder


def load_split(raw: Path, split: str) -> tuple[list[dict], Counter]:
    """Records for one release split, plus a count of the boxes left out, by reason."""
    data = json.loads((Path(raw) / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    records = {
        img["id"]: {
            "file": img["file_name"],
            "path": Path(raw) / SPLITS[split] / img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "boxes": [],
        }
        for img in data["images"]
    }
    dropped = Counter()
    for a in data["annotations"]:
        name = names.get(a["category_id"])
        if name is None:
            dropped[f"unknown category id {a['category_id']}"] += 1
        elif name in DROPPED:
            dropped[name] += 1
        elif name not in CLASS_MAP:
            raise KeyError(f"Unknown OralXrays-9 class '{name}'; add it to CLASS_MAP or DROPPED")
        else:
            records[a["image_id"]]["boxes"].append({"bbox": xywh_to_xyxy(a["bbox"]), "label": CLASS_MAP[name]})
    return list(records.values()), dropped
