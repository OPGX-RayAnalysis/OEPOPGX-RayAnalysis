"""Score a findings detector on a YOLO dataset split (task 2.16).

Same idea as score_numbering.py, for findings (DENTEX diagnoses, Zhou treatments,
OralXrays-9). See src/opg/scoring_findings.py for exactly what each number means.

Usage:
    python scripts/score_findings.py --weights runs/<run>/weights/best.pt --data data/yolo/findings/data.yaml --split val
    python scripts/score_findings.py --weights best.pt --data data/yolo/findings/data.yaml --split test --merge-caries
    python scripts/score_findings.py --weights best.pt --data data/yolo/oralxrays/data.yaml --split test

Classes are matched by name, so any findings data.yaml works. Only classes both the model and
the data.yaml know are scored; the rest are listed and left out.
--merge-caries counts deep caries as caries (in the labels and the predictions).
--match cover: a prediction matches if it covers at least half of the real box, instead of IoU.
For datasets that box the lesion while our model boxes the whole tooth (OralXrays-9): score
those with both rules and report both (docs/datasets/oralxrays.md).
--skip leaves out the images named in a text file (one file name per line).
Writes, next to the weights' run folder or into --out (score_findings_merged.* with --merge-caries,
_cover added with --match cover; compare_scores.py --findings takes any of these .json files):
    score_findings.json         all numbers, plus 95% bootstrap intervals (ci95)
    score_findings.csv          per class: truth, tp, fp, fn, precision, recall, F1, AP50
    score_findings_images.csv   per image per class, for compare_scores.py --findings
    score_findings_preds.csv    per scored prediction: image, finding, confidence, matched
    score_findings_claims.csv   per claim (and per omitted real finding), for log_hallucination.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from opg.claims import CLAIM_FIELDS
from opg.data import yolo as Y
from opg.scoring_findings import (
    FINDING_IMAGE_FIELDS, FINDING_PRED_FIELDS, MERGE_CARIES, Finding, FindingScorer, bootstrap_findings_ci, normalize,
)

MAP_CONF = 0.001  # AP50 needs (almost) every prediction, precision/recall/F1 use --conf


def default_out(weights: Path) -> Path:
    """runs/<name>/weights/best.pt -> runs/<name>/"""
    return weights.parent.parent if weights.parent.name == "weights" else weights.parent


def class_names(names: dict, merge: dict) -> list[str]:
    """data.yaml / model class names, normalised and merged, in order, without repeats."""
    out = []
    for n in names.values():
        n = normalize(str(n))
        n = merge.get(n, n)
        if n not in out:
            out.append(n)
    return out


def to_findings(result) -> list[Finding]:
    return [
        Finding(normalize(str(result.names[int(c)])), [float(v) for v in xyxy], float(conf))
        for xyxy, c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--merge-caries", action="store_true", help="count deep caries as caries")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold for counting a finding")
    parser.add_argument("--iou", type=float, default=0.5, help="overlap needed to match a real finding")
    parser.add_argument("--match", default="iou", choices=["iou", "cover"], help="how overlap is measured")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip", type=Path, default=None, help="text file of image names to leave out")
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the 95%% interval, 0 = none")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Needs ultralytics. Install PyTorch, then: pip install ultralytics")

    merge = MERGE_CARIES if args.merge_caries else {}
    data = Y.load_data_yaml(args.data)
    images = Y.split_images(data, args.split)
    if not images:
        raise SystemExit(f"No images in split '{args.split}'.")
    if args.skip:
        skip = {Path(line.strip()).stem.lower() for line in args.skip.read_text(encoding="utf-8").splitlines() if line.strip()}
        images = [p for p in images if p.stem.lower() not in skip]
        print(f"Skipping {len(skip)} images listed in {args.skip}: {len(images)} left")
    model = YOLO(str(args.weights))

    data_classes, model_classes = class_names(data["names"], merge), class_names(model.names, merge)
    classes = [c for c in data_classes if c in model_classes]
    if not classes:
        raise SystemExit(f"Model classes {model_classes} share no name with data.yaml classes {data_classes}.")
    for what, names in (("data.yaml", data_classes), ("model", model_classes)):
        left_out = [c for c in names if c not in classes]
        if left_out:
            print(f"Not scored, only the {what} has: {', '.join(left_out)}")

    scorer = FindingScorer(classes, iou_threshold=args.iou, merge=merge, match_rule=args.match)
    for i, path in enumerate(images, 1):
        h, w = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape[:2]
        truth = [Finding(normalize(b.name), list(b.xyxy)) for b in Y.read_labels(path, data["names"], w, h)]
        result = model.predict(str(path), conf=MAP_CONF, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred_all = to_findings(result)
        scorer.add(truth, [p for p in pred_all if p.conf >= args.conf], pred_all, name=path.name)
        if i % 10 == 0 or i == len(images):
            print(f"  {i}/{len(images)}", flush=True)

    score = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "merge_caries": args.merge_caries,
        "classes": classes,
        "conf": args.conf,
        "iou": args.iou,
        "match": args.match,
        "imgsz": args.imgsz,
        "skip": str(args.skip) if args.skip else None,
        **scorer.result(),
    }
    if args.bootstrap:
        score["ci95"] = bootstrap_findings_ci(scorer.images, n=args.bootstrap)
        score["bootstrap_resamples"] = args.bootstrap
    out = args.out or default_out(args.weights)
    out.mkdir(parents=True, exist_ok=True)
    stem = ("score_findings_merged" if args.merge_caries else "score_findings") + ("_cover" if args.match == "cover" else "")
    (out / f"{stem}.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    with open(out / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["finding", "truth", "tp", "fp", "fn", "precision", "recall", "f1", "ap50"])
        writer.writeheader()
        writer.writerows({"finding": c, **row} for c, row in score["per_class"].items())
    with open(out / f"{stem}_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINDING_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(scorer.images)
    with open(out / f"{stem}_preds.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FINDING_PRED_FIELDS)
        writer.writeheader()
        writer.writerows(scorer.preds)
    with open(out / f"{stem}_claims.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLAIM_FIELDS)
        writer.writeheader()
        writer.writerows(scorer.claims)

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def ci(bounds):
        return f"{pct(bounds[0])} to {pct(bounds[1])}" if bounds else ""

    ci95 = score.get("ci95", {})
    print(f"\n{args.split}: {score['images']} images, {score['findings']} real findings, "
          f"deep caries {'merged into caries' if args.merge_caries else 'kept separate'}, "
          f"matched by {'cover' if args.match == 'cover' else 'IoU'}")
    print(f"  {'class':<20}{'truth':>6}{'tp':>6}{'fp':>6}{'fn':>6}{'prec':>8}{'recall':>8}{'F1':>8}   F1 95% CI        AP50")
    for c, row in score["per_class"].items():
        bounds = ci95.get("per_class", {}).get(c, {}).get("f1")
        print(f"  {c:<20}{row['truth']:>6}{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}"
              f"{pct(row['precision']):>8}{pct(row['recall']):>8}{pct(row['f1']):>8}   {ci(bounds):<17}{pct(row['ap50'])}")
    p = score["pooled"]
    print(f"  mean F1         {pct(score['mean_f1'])}   95% CI {ci(ci95.get('mean_f1'))}   (average over classes, the headline)")
    print(f"  pooled          precision {pct(p['precision'])}, recall {pct(p['recall'])}, F1 {pct(p['f1'])}   (all findings together)")
    print(f"  invented        {p['fp']} findings matching no real one;  missed {p['fn']}")
    print(f"  mAP50 (raw)     {pct(score['map50'])}")
    print(f"Saved {out / stem}.json, {stem}.csv, {stem}_images.csv, {stem}_preds.csv and {stem}_claims.csv")


if __name__ == "__main__":
    main()
