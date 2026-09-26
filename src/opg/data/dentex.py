"""Read DENTEX annotations into one simple record format, and write YOLO datasets.

Record = {"file": str, "path": Path, "width": int, "height": int,
          "boxes": [{"bbox": [x1, y1, x2, y2], "fdi": int, "label": str | None}]}

Two source formats:
- Training JSONs (COCO-like): category_id_1 = quadrant, _2 = tooth position,
  _3 = diagnosis. Ids are indices into categories_1/2/3, so we map through the
  category NAMES, never assume id == number.
- Test labels (LabelMe, one JSON per image): shape label "<n>-<diagnosis in Turkish>-<FDI>".
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from ..fdi import ALL_PERMANENT
from ..geometry import xywh_to_xyxy

DIAGNOSIS_MAP = {
    "Impacted": "impacted",
    "Caries": "caries",
    "Deep Caries": "deep_caries",
    "Periapical Lesion": "periapical_lesion",
}
# Turkish labels in the official test set. Run scripts/dentex_to_yolo.py --list-test-labels
# after downloading to confirm this covers every label actually present.
TURKISH_DIAGNOSIS_MAP = {
    "çürük": "caries",
    "derin çürük": "deep_caries",
    "gömülü": "impacted",
    "gömülü diş": "impacted",
    "periapikal lezyon": "periapical_lesion",
    "lezyon": "periapical_lesion",
}

TEETH_CLASSES = [str(f) for f in ALL_PERMANENT]  # "11" ... "48", 32 classes
FINDING_CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted"]


def load_coco_hierarchical(json_path: Path, images_dir: Path) -> list[dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    names = {
        level: {c["id"]: c["name"] for c in data.get(f"categories_{level}", [])} for level in (1, 2, 3)
    }
    records = {
        img["id"]: {
            "file": img["file_name"],
            "path": Path(images_dir) / img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "boxes": [],
        }
        for img in data["images"]
    }
    for a in data["annotations"]:
        quadrant = int(names[1][a["category_id_1"]])
        position = int(names[2][a["category_id_2"]]) if "category_id_2" in a else None
        diagnosis = names[3][a["category_id_3"]] if "category_id_3" in a else None
        records[a["image_id"]]["boxes"].append(
            {
                "bbox": xywh_to_xyxy(a["bbox"]),
                "fdi": quadrant * 10 + position if position else None,
                "label": DIAGNOSIS_MAP[diagnosis] if diagnosis else None,
            }
        )
    return list(records.values())


def parse_test_label(label: str) -> tuple[str, int]:
    """'1-çürük-33' -> ('çürük', 33)."""
    parts = label.split("-")
    return "-".join(parts[1:-1]).strip().lower(), int(parts[-1])


def load_labelme_test(label_dir: Path, images_dir: Path, strict: bool = True) -> list[dict]:
    records = []
    for path in sorted(Path(label_dir).glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        boxes = []
        for shape in d["shapes"]:
            name, fdi = parse_test_label(shape["label"])
            if name not in TURKISH_DIAGNOSIS_MAP:
                if strict:
                    raise KeyError(f"Unknown test label '{shape['label']}' in {path.name}; extend TURKISH_DIAGNOSIS_MAP")
                continue
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            boxes.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "fdi": fdi, "label": TURKISH_DIAGNOSIS_MAP[name]})
        records.append(
            {
                "file": d["imagePath"],
                "path": Path(images_dir) / d["imagePath"],
                "width": d["imageWidth"],
                "height": d["imageHeight"],
                "boxes": boxes,
            }
        )
    return records


def split_records(records: list[dict], fractions: dict[str, float], seed: int = 42) -> dict[str, list[dict]]:
    """Random image-level split. DENTEX has no patient ids, so image level is the best we can do."""
    shuffled = sorted(records, key=lambda r: r["file"])
    random.Random(seed).shuffle(shuffled)
    out, start = {}, 0
    names = list(fractions)
    for i, name in enumerate(names):
        end = len(shuffled) if i == len(names) - 1 else start + round(fractions[name] * len(shuffled))
        out[name] = shuffled[start:end]
        start = end
    return out


def yolo_line(box: list[float], cls: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_yolo(splits: dict[str, list[dict]], out_dir: Path, classes: list[str], class_key: str) -> Path:
    """Write images/labels/<split>/ plus data.yaml. class_key is 'fdi' or 'label'."""
    out_dir = Path(out_dir)
    index = {c: i for i, c in enumerate(classes)}
    for split, records in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        manifest = []
        for r in records:
            target = out_dir / "images" / split / r["file"]
            if not target.exists():
                shutil.copy2(r["path"], target)
            lines = [
                yolo_line(b["bbox"], index[str(b[class_key])], r["width"], r["height"])
                for b in r["boxes"]
                if b[class_key] is not None and str(b[class_key]) in index
            ]
            (out_dir / "labels" / split / (Path(r["file"]).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            manifest.append({**r, "path": str(r["path"])})
        (out_dir / f"annotations_{split}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    yaml_lines = [f"path: {out_dir.resolve().as_posix()}"]
    yaml_lines += [f"{split}: images/{split}" for split in splits]
    yaml_lines += ["names:"] + [f"  {i}: '{c}'" for i, c in enumerate(classes)]
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return data_yaml
