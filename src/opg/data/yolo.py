"""Read YOLO datasets (data.yaml + images/ + labels/), shared by view_labels and score_numbering.

Works on any dataset in the Ultralytics layout: DENTEX teeth, Shubh's cleaned copy, Krisha's Tufts set.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Box:
    name: str  # class name, e.g. "46"
    xyxy: tuple[float, float, float, float]  # pixels
    conf: float = 1.0

    @property
    def fdi(self) -> int | None:
        """The FDI number if the class name is one, else None (e.g. a findings class)."""
        return int(self.name) if self.name.isdigit() else None


def _parse_simple_yaml(text: str) -> dict:
    """Fallback for when PyYAML isn't installed. Handles the flat files dentex_to_yolo.py writes."""
    data: dict = {}
    names: dict[int, str] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip().strip("'\"")
        if names is not None and raw.startswith((" ", "\t")):
            names[int(key)] = value
            continue
        names = None
        if key == "names" and not value:
            names = data["names"] = {}
        elif key == "names":
            data["names"] = ast.literal_eval(line.strip().partition(":")[2].strip())
        else:
            data[key] = value
    return data


def load_data_yaml(yaml_path: str | Path) -> dict:
    """Load data.yaml, resolve the dataset root and return {'root', 'names', <split>: dir, ...}."""
    yaml_path = Path(yaml_path)
    text = yaml_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        data = _parse_simple_yaml(text)

    names = data["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    data["names"] = {int(k): str(v) for k, v in names.items()}

    # Same rule as Ultralytics: no "path" means the yaml's own folder is the root.
    root = Path(data["path"]) if data.get("path") else yaml_path.parent
    if not root.is_absolute() and not root.exists():
        root = yaml_path.parent / root
    data["root"] = root
    return data


def split_images(data: dict, split: str) -> list[Path]:
    """Sorted image paths for one split (only folder-style splits are supported)."""
    if split not in data:
        available = [k for k in ("train", "val", "test") if k in data]
        raise SystemExit(f"Split '{split}' is not in data.yaml. Available: {', '.join(available)}")
    folder = data["root"] / str(data[split])
    if not folder.is_dir():
        raise SystemExit(f"Image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path(image_path: Path) -> Path:
    """images/<split>/x.png -> labels/<split>/x.txt (the Ultralytics convention)."""
    parts = list(image_path.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image_path: Path, names: dict[int, str], width: int, height: int) -> list[Box]:
    """Read the YOLO label file for an image as pixel boxes. A missing file means no boxes."""
    path = label_path(image_path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append(Box(names.get(int(cls), cls), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes
