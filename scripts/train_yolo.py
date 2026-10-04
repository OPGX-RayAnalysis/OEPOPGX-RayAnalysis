"""Train a YOLO detector on a dataset made by dentex_to_yolo.py (plan tasks 1.7, 2.4, 2.6).

Usage:
    python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name teeth-v1
    python scripts/train_yolo.py --data data/yolo/findings/data.yaml --name findings-v1 --model yolo11m.pt

Notes for a 6 GB laptop GPU (RTX 4050): imgsz 1024 with batch 4 fits yolo11s/m.
On Colab or Kaggle (T4, 16 GB) use batch 8-16.
Horizontal flip is OFF by default: flipping an OPG swaps left and right, which would
silently corrupt FDI labels (18 would become 28).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    args = parser.parse_args()

    from ultralytics import YOLO

    YOLO(args.model).train(
        data=args.data,
        name=args.name,
        project=str(Path("runs").resolve()),  # absolute, or Ultralytics nests it under runs/detect/
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fliplr=0.0,  # never mirror: it breaks FDI left/right labels
        mosaic=0.5,
        patience=25,
        seed=42,
    )


if __name__ == "__main__":
    main()
