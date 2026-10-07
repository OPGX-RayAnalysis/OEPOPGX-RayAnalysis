"""Read the Tufts Dental Database into the same record format as DENTEX (plan task 1.14).

Record = {"file": str, "path": Path, "width": int, "height": int,
          "boxes": [{"bbox": [x1, y1, x2, y2], "fdi": int, "label": None}]}

Download layout (data/raw/tufts/):
    Radiographs/1.JPG ... 1000.JPG
    Segmentation/teeth_polygon.json    Labelbox export, one entry per image

Each entry: {"External ID": "1.jpg",
             "Label": {"objects": [{"title": "1", "bounding box": [...], "polygons": [[[x, y], ...], ...]}]}}.
"title" is the tooth's Universal number (1-32, or A-T for primary teeth); it is converted to FDI
with opg.fdi.from_universal.

"polygons" holds several outlines per tooth, traced from a mask: the tooth itself plus many
specks of a few pixels, some far from the tooth. The box is taken over every outline at least
SPECK_FRACTION the area of the largest one, so specks can't stretch it. We ignore the supplied
"bounding box": it is [y1, x1, y2, x2] and some are broken (e.g. 1x1 px for a whole tooth).
See docs/datasets/tufts.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .. import fdi as F

ANNOTATION_FILE = Path("Segmentation") / "teeth_polygon.json"
IMAGES_DIR = Path("Radiographs")
# On the real data 5%, 10% and 20% give the same boxes; any of them removes every far-away speck.
SPECK_FRACTION = 0.1


def outline_area(outline: list[list[float]]) -> float:
    return cv2.contourArea(np.asarray(outline, dtype=np.float32)) if len(outline) >= 3 else 0.0


def tooth_points(obj: dict) -> list[tuple[float, float]]:
    """(x, y) points of the outlines that belong to the tooth, without the specks."""
    outlines = [o for o in obj.get("polygons", []) if o]
    if not outlines:
        return []
    areas = [outline_area(o) for o in outlines]
    largest = max(areas)
    kept = [o for o, a in zip(outlines, areas) if a >= SPECK_FRACTION * largest] if largest > 0 else outlines
    return [(p[0], p[1]) for o in kept for p in o]


def outline_to_box(points: list[tuple[float, float]]) -> list[float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def tooth_fdi(title: str) -> int | None:
    """Universal title -> FDI, or None if the title isn't a tooth number."""
    try:
        return F.from_universal(title)
    except ValueError:
        return None


def index_images(images_dir: Path) -> dict[str, Path]:
    """Lower-case stem -> real file. Labelbox ids are lower case ('1.jpg') while the files are
    '1.JPG'; matching on the real listing keeps the true file name on any OS."""
    return {path.stem.lower(): path for path in Path(images_dir).iterdir() if path.is_file()}


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image {path}")
    height, width = img.shape[:2]
    return width, height


def read_entries(raw_folder: Path, annotation_file: Path | None = None) -> list[dict]:
    path = Path(annotation_file) if annotation_file else Path(raw_folder) / ANNOTATION_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def load_tufts(raw_folder: Path, annotation_file: Path | None = None) -> list[dict]:
    """One record per image, including images with no teeth (edentulous patients). Objects without
    an outline or a tooth-number title are skipped (count_titles shows which titles exist)."""
    raw_folder = Path(raw_folder)
    images_dir = raw_folder / IMAGES_DIR
    images = index_images(images_dir)
    records = []
    for entry in read_entries(raw_folder, annotation_file):
        image = images.get(Path(entry["External ID"]).stem.lower())
        if image is None:
            raise FileNotFoundError(f"No image for {entry['External ID']} in {images_dir}")
        width, height = image_size(image)
        boxes = []
        for obj in entry.get("Label", {}).get("objects", []):
            fdi, points = tooth_fdi(obj.get("title", "")), tooth_points(obj)
            if fdi is not None and points:
                boxes.append({"bbox": outline_to_box(points), "fdi": fdi, "label": None})
        records.append({"file": image.name, "path": image, "width": width, "height": height, "boxes": boxes})
    return records


def count_titles(raw_folder: Path, annotation_file: Path | None = None) -> Counter:
    """How often each object title appears, to spot titles that aren't Universal tooth numbers."""
    return Counter(
        str(obj.get("title"))
        for entry in read_entries(raw_folder, annotation_file)
        for obj in entry.get("Label", {}).get("objects", [])
    )
