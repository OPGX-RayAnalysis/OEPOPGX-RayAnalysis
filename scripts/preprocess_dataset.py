"""Build a crop + CLAHE copy of a YOLO dataset, with boxes shifted to match the crop.

Training still resizes every image to --imgsz (1024). This script does not resize the
saved files: resize_long_side is a later step (task 2.1), and resizing only the cleaned
copy would make the plain-vs-CLAHE comparison unfair. Normalised boxes are unchanged by
an aspect-preserving resize, so the crop offset is the only geometry change.

Usage:
    python scripts/preprocess_dataset.py --data data/yolo/teeth/data.yaml --out data/yolo/teeth_clahe
    python scripts/preprocess_dataset.py --data data/yolo/teeth/data.yaml --out data/yolo/teeth_clahe --preview 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from opg.data.yolo import label_path, load_data_yaml, split_images
from opg.preprocess import load_gray, preprocess, shift_yolo_labels


def write_data_yaml(src: dict, out_dir: Path, splits: list[str]) -> None:
    """Same split paths and class order as the source. No path: line, on purpose."""
    names = [src["names"][i] for i in sorted(src["names"])]
    lines = [f"{split}: images/{split}" for split in splits]
    lines.append("names:")
    lines.extend(f"  {i}: '{name}'" for i, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_image(src: Path, dst_img: Path, dst_label: Path) -> tuple[np.ndarray, np.ndarray]:
    """Crop, CLAHE, and rewrite the label. Returns (original gray, cleaned) for previews."""
    original = load_gray(src)
    cleaned = preprocess(src)
    image, offset = cleaned["image"], cleaned["offset"]
    label = label_path(src)
    text = label.read_text(encoding="utf-8") if label.exists() else ""
    shifted = shift_yolo_labels(text, original.shape[1], original.shape[0], offset, image.shape[1], image.shape[0])
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_img), image):
        raise OSError(f"could not write {dst_img}")
    dst_label.write_text(shifted, encoding="utf-8")
    return original, image


def _boxes_from_text(text: str, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        _, cx, cy, w, h = raw.split()[:5]
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))
    return boxes


def _panel(img: np.ndarray, label_text: str, title: str) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in _boxes_from_text(label_text, img.shape[1], img.shape[0]):
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
    cv2.putText(out, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def write_preview(before: np.ndarray, before_text: str, after: np.ndarray, after_text: str, path: Path) -> None:
    """Side-by-side, top-aligned. The after panel is padded so the crop is obvious."""
    left = _panel(before, before_text, "before")
    right = _panel(after, after_text, "crop + CLAHE")
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1] + 8

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] + 8 : left.shape[1] + 8 + right.shape[1]] = right
    path.parent.mkdir(parents=True, exist_ok=True)
    # Long OPGs make huge panels; slides only need enough detail to see the crop and boxes.
    long_side = max(canvas.shape[:2])
    if long_side > 1600:
        scale = 1600 / long_side
        canvas = cv2.resize(canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preview", type=int, default=0, help="write this many before/after images from train")
    args = parser.parse_args()

    src = load_data_yaml(args.data)
    splits = [name for name in ("train", "val", "test") if name in src]
    if not splits:
        raise SystemExit("data.yaml has no train/val/test split")

    preview_left = args.preview
    for split in splits:
        images = split_images(src, split)
        print(f"{split}: {len(images)} images", flush=True)
        for i, image_path in enumerate(images, 1):
            dst_img = args.out / "images" / split / image_path.name
            dst_label = args.out / "labels" / split / f"{image_path.stem}.txt"
            before, after = process_image(image_path, dst_img, dst_label)
            if split == "train" and preview_left > 0:
                src_label = label_path(image_path)
                before_text = src_label.read_text(encoding="utf-8") if src_label.exists() else ""
                write_preview(before, before_text, after, dst_label.read_text(encoding="utf-8"), args.out / "previews" / f"{image_path.stem}.png")
                preview_left -= 1
            if i % 50 == 0 or i == len(images):
                print(f"  {i}/{len(images)}", flush=True)

    write_data_yaml(src, args.out, splits)
    print(f"Wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
