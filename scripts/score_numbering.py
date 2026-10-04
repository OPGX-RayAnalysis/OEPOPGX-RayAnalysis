"""Score a tooth-numbering model on a YOLO dataset split (plan task 1.11).

One fair way of measuring numbering for everyone: DENTEX test split, Shubh's cleaned
dataset, Krisha's Tufts set. See src/opg/scoring.py for exactly what each number means.

Usage:
    python scripts/score_numbering.py --weights runs/teeth-v1/weights/best.pt --data data/yolo/teeth/data.yaml --split test
    python scripts/score_numbering.py --weights best.pt --data data/yolo/teeth/data.yaml --split test --rules

--rules applies the anatomy rules (postprocess.apply_anatomy_rules) to the predictions first.
Writes score.json and score.csv (score_rules.* with --rules) next to the weights' run folder,
or into --out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from opg.data import yolo as Y
from opg.postprocess import RawTooth, apply_anatomy_rules
from opg.scoring import Scorer, Tooth

MAP_CONF = 0.001  # mAP needs (almost) every prediction, accuracy uses --conf


def default_out(weights: Path) -> Path:
    """runs/<name>/weights/best.pt -> runs/<name>/"""
    return weights.parent.parent if weights.parent.name == "weights" else weights.parent


def to_teeth(result) -> list[Tooth]:
    teeth = []
    for xyxy, c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        name = str(result.names[int(c)])
        if name.isdigit():
            teeth.append(Tooth(int(name), [float(v) for v in xyxy], float(conf)))
    return teeth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rules", action="store_true", help="apply anatomy rules before scoring")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for counting a tooth")
    parser.add_argument("--iou", type=float, default=0.5, help="overlap needed to match a real tooth")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Needs ultralytics. Install PyTorch, then: pip install ultralytics")

    data = Y.load_data_yaml(args.data)
    images = Y.split_images(data, args.split)
    if not images:
        raise SystemExit(f"No images in split '{args.split}'.")
    model = YOLO(str(args.weights))

    scorer = Scorer(iou_threshold=args.iou)
    for i, path in enumerate(images, 1):
        h, w = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape[:2]
        truth = [Tooth(b.fdi, list(b.xyxy)) for b in Y.read_labels(path, data["names"], w, h) if b.fdi is not None]
        result = model.predict(str(path), conf=MAP_CONF, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred_all = to_teeth(result)
        pred = [p for p in pred_all if p.conf >= args.conf]
        if args.rules:
            fixed = apply_anatomy_rules([RawTooth(p.fdi, p.bbox, p.conf) for p in pred])
            pred = [Tooth(t.fdi, t.bbox, t.confidence) for t in fixed]
        scorer.add(truth, pred, pred_all)
        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)}", flush=True)

    score = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "rules": args.rules,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        **scorer.result(),
    }
    out = args.out or default_out(args.weights)
    out.mkdir(parents=True, exist_ok=True)
    stem = "score_rules" if args.rules else "score"
    (out / f"{stem}.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fdi", "count", "detected", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(score["per_tooth"])

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    print(f"\n{args.split}: {score['images']} images, {score['teeth']} teeth, rules {'on' if args.rules else 'off'}")
    print(f"  numbering accuracy  {pct(score['numbering_accuracy'])}   (right number / all teeth)")
    print(f"  accuracy if found   {pct(score['accuracy_if_found'])}")
    print(f"  detection rate      {pct(score['detection_rate'])}")
    print(f"  missing-tooth F1    {pct(score['missing_tooth_f1'])}")
    print(f"  mAP50 (raw model)   {pct(score['map50'])}")
    worst = sorted((r for r in score["per_tooth"] if r["count"]), key=lambda r: r["accuracy"])[:5]
    print("  weakest teeth       " + ", ".join(f"{r['fdi']} {pct(r['accuracy'])}" for r in worst))
    print(f"Saved {out / stem}.json and .csv")


if __name__ == "__main__":
    main()
