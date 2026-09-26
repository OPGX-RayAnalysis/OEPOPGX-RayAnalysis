"""Convert downloaded DENTEX data into two YOLO datasets (plan task 1.6).

    data/yolo/teeth     32 FDI classes, from the enumeration subset (all teeth labelled)
                        split 85% train / 15% val
    data/yolo/findings  4 diagnosis classes, from the disease subset (abnormal teeth only)
                        split 90% train / 10% val, plus the official 250-image test set

Usage:
    python scripts/dentex_to_yolo.py
    python scripts/dentex_to_yolo.py --raw data/raw/dentex --out data/yolo
    python scripts/dentex_to_yolo.py --list-test-labels     # check the Turkish label map
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from opg.data import dentex as D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/dentex"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-test-labels", action="store_true")
    args = parser.parse_args()

    enum_dir = args.raw / "training_data" / "quadrant_enumeration"
    disease_dir = args.raw / "training_data" / "quadrant-enumeration-disease"
    test_dir = args.raw / "disease"

    if args.list_test_labels:
        labels = Counter(
            shape["label"].split("-", 1)[1].rsplit("-", 1)[0]
            for p in (test_dir / "label").glob("*.json")
            for shape in json.loads(p.read_text(encoding="utf-8"))["shapes"]
        )
        for name, count in labels.most_common():
            mapped = D.TURKISH_DIAGNOSIS_MAP.get(name.strip().lower(), "!! NOT MAPPED")
            print(f"{count:5d}  {name!r:30} -> {mapped}")
        return

    if enum_dir.exists():
        records = D.load_coco_hierarchical(enum_dir / "train_quadrant_enumeration.json", enum_dir / "xrays")
        splits = D.split_records(records, {"train": 0.85, "val": 0.15}, args.seed)
        path = D.write_yolo(splits, args.out / "teeth", D.TEETH_CLASSES, "fdi")
        print(f"teeth:    {', '.join(f'{k}={len(v)}' for k, v in splits.items())} -> {path}")
    else:
        print(f"skip teeth: {enum_dir} not found")

    if disease_dir.exists():
        records = D.load_coco_hierarchical(
            disease_dir / "train_quadrant_enumeration_disease.json", disease_dir / "xrays"
        )
        splits = D.split_records(records, {"train": 0.9, "val": 0.1}, args.seed)
        if test_dir.exists():
            splits["test"] = D.load_labelme_test(test_dir / "label", test_dir / "input")
        path = D.write_yolo(splits, args.out / "findings", D.FINDING_CLASSES, "label")
        print(f"findings: {', '.join(f'{k}={len(v)}' for k, v in splits.items())} -> {path}")
    else:
        print(f"skip findings: {disease_dir} not found")


if __name__ == "__main__":
    main()
