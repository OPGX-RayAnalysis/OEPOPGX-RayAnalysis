"""Draw YOLO labels (and optionally model predictions) on OPGs for checking (plan task 1.12).

Each box shows its tooth number, coloured by quadrant: 1 red, 2 green, 3 blue, 4 orange.
Remember: the patient's right is on the image's LEFT, so tooth 18 is at the top far left.

Usage:
    python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split train --n 30 --random --out spotcheck/
    python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split test --pred runs/teeth-v1/weights/best.pt
    python scripts/view_labels.py --data data/yolo/teeth/data.yaml --system universal

With --pred, each output image has two panels: real labels on top, model predictions below.
Used by Faizan (label spot-check), Krisha (Tufts check) and Yash (looking at mistakes).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from opg import fdi as F
from opg.data import yolo as Y

# BGR colours per quadrant; primary-tooth quadrants 5-8 reuse 1-4.
QUADRANT_COLOURS = {1: (40, 40, 230), 2: (40, 180, 40), 3: (220, 120, 30), 4: (0, 150, 255)}
OTHER_COLOUR = (200, 0, 200)


def box_text(box: Y.Box, system: str) -> str:
    if box.fdi is None:
        return box.name
    try:
        return F.convert(box.fdi, system)
    except ValueError:
        return box.name


def box_colour(box: Y.Box) -> tuple[int, int, int]:
    if box.fdi is None:
        return OTHER_COLOUR
    quadrant = box.fdi // 10
    return QUADRANT_COLOURS.get((quadrant - 1) % 4 + 1, OTHER_COLOUR) if 1 <= quadrant <= 8 else OTHER_COLOUR


def draw_boxes(img: np.ndarray, boxes: list[Y.Box], system: str, title: str, show_conf: bool = False) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    scale = out.shape[0] / 1000  # text and lines grow with the image
    thick = max(1, round(2 * scale))
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy)
        colour = box_colour(box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        text = box_text(box, system) + (f" {box.conf:.2f}" if show_conf else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, thick)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), colour, -1)
        cv2.putText(out, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255), thick)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, thick + 1)
    cv2.rectangle(out, (0, 0), (tw + int(30 * scale), th + int(30 * scale)), (0, 0, 0), -1)
    cv2.putText(out, title, (int(15 * scale), th + int(15 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), thick + 1)
    return out


def predict(model, image_path: Path) -> list[Y.Box]:
    result = model.predict(str(image_path), verbose=False)[0]
    return [
        Y.Box(str(result.names[int(c)]), tuple(float(v) for v in xyxy), float(conf))
        for xyxy, c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]


def pick(images: list[Path], n: int, shuffle: bool, seed: int) -> list[Path]:
    if shuffle:
        images = random.Random(seed).sample(images, len(images))
    return images[:n] if n > 0 else images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=30, help="number of images (0 = all)")
    parser.add_argument("--random", action="store_true", help="pick images at random instead of the first N")
    parser.add_argument("--seed", type=int, default=0, help="same seed = same random pick")
    parser.add_argument("--system", default="fdi", choices=["fdi", "universal", "palmer"])
    parser.add_argument("--pred", default=None, help="model weights (best.pt) to also draw predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence for predictions")
    parser.add_argument("--out", type=Path, default=Path("spotcheck"))
    args = parser.parse_args()

    data = Y.load_data_yaml(args.data)
    images = pick(Y.split_images(data, args.split), args.n, args.random, args.seed)
    if not images:
        raise SystemExit(f"No images in split '{args.split}'.")

    model = None
    if args.pred:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("--pred needs ultralytics. Install PyTorch, then: pip install ultralytics")
        model = YOLO(args.pred)
        model.overrides["conf"] = args.conf

    args.out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  skip (can't read): {path.name}")
            continue
        h, w = img.shape[:2]
        labels = Y.read_labels(path, data["names"], w, h)
        panel = draw_boxes(img, labels, args.system, f"{path.name}  labels ({len(labels)})")
        if model is not None:
            preds = predict(model, path)
            panel = np.vstack([panel, draw_boxes(img, preds, args.system, f"predictions ({len(preds)})", True)])
        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {i}/{len(images)}  {path.name}  {len(labels)} labelled teeth")
    print(f"Done. {len(images)} images in {args.out.resolve()}")


if __name__ == "__main__":
    main()
