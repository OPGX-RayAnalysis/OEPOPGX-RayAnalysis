"""Read the Zhou et al. dual-labelled dataset into the same record format as Tufts (plan task 2.2).

Record = {"file": str, "path": Path, "width": int, "height": int,
          "boxes": [{"bbox": [x1, y1, x2, y2], "fdi": int, "labels": [str, ...]}]}
"labels" is [] for a healthy tooth, two labels for a tooth with a root canal filling and a crown.

Download layout (data/raw/zhou/, Kaggle "a-dual-labeled-dataset"):
    images1/1000.png ...         the images (the Kaggle copy has 500 of them)
    labels/1000.json ...         LabelMe, one file per image
    labels/cw1000.json ...       LabelMe files for a second image set ("cw*.png") that is not in
                                 the download; they are ignored (no image to go with them)

Each shape: {"label": "36", "group_id": 1, "points": [[x, y], ...], "shape_type": "polygon"}.
"label" is the tooth's FDI number; "group_id" is the tooth state (None = 0, no anomaly).
"label" 91 is not a tooth number and is skipped. See docs/datasets/zhou.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2

from ..fdi import ALL_PERMANENT

LABELS_DIR = Path("labels")
IMAGES_GLOB = "images*/*.png"
# Zhou state (group_id) -> our finding labels.
STATE_LABELS = {
    0: [],
    1: ["filling"],
    2: ["root_canal_filling"],
    3: ["crown"],
    4: ["caries"],
    5: ["residual_root"],
    6: ["root_canal_filling", "crown"],
}
FINDING_CLASSES = ["filling", "root_canal_filling", "crown", "caries", "residual_root"]


def tooth_fdi(label: str) -> int | None:
    """Shape label -> FDI number, or None if it isn't a permanent tooth (11-48)."""
    try:
        fdi = int(label)
    except ValueError:
        return None
    return fdi if fdi in ALL_PERMANENT else None


def state_labels(group_id: int | None, where: str = "") -> list[str]:
    state = group_id or 0
    if state not in STATE_LABELS:
        raise KeyError(f"Unknown Zhou tooth state {group_id!r} {where}; extend STATE_LABELS")
    return list(STATE_LABELS[state])


def outline_to_box(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    height, width = img.shape[:2]
    return width, height


def index_images(raw_folder: Path) -> list[Path]:
    return sorted(Path(raw_folder).glob(IMAGES_GLOB), key=lambda p: p.name)


def read_annotation(raw_folder: Path, image: Path) -> dict:
    path = Path(raw_folder) / LABELS_DIR / (image.stem + ".json")
    if not path.exists():
        raise FileNotFoundError(f"No annotation {path} for image {image}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_zhou(raw_folder: Path) -> list[dict]:
    """One record per image. Shapes whose label isn't a tooth number 11-48 are skipped
    (count_labels shows which labels exist)."""
    raw_folder = Path(raw_folder)
    records = []
    for image in index_images(raw_folder):
        d = read_annotation(raw_folder, image)
        width, height = image_size(image)
        boxes = []
        for shape in d["shapes"]:
            fdi = tooth_fdi(shape["label"])
            if fdi is None or not shape.get("points"):
                continue
            labels = state_labels(shape.get("group_id"), f"in {image.stem}.json")
            boxes.append({"bbox": outline_to_box(shape["points"]), "fdi": fdi, "labels": labels})
        records.append({"file": image.name, "path": image, "width": width, "height": height, "boxes": boxes})
    return records


def finding_records(records: list[dict]) -> list[dict]:
    """Same images, one box per finding label: a healthy tooth gives none, RCT + crown gives two."""
    return [
        {**r, "boxes": [{"bbox": b["bbox"], "fdi": b["fdi"], "label": label} for b in r["boxes"] for label in b["labels"]]}
        for r in records
    ]


def count_labels(raw_folder: Path) -> Counter:
    """How often each shape label appears in the annotations that have an image."""
    raw_folder = Path(raw_folder)
    return Counter(
        str(shape["label"]) for image in index_images(raw_folder) for shape in read_annotation(raw_folder, image)["shapes"]
    )
