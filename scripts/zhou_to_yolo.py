"""Convert the Zhou et al. dual-labelled dataset into two YOLO datasets (plan task 2.2).

    data/yolo/zhou/teeth      32 FDI classes (same order as DENTEX)
    data/yolo/zhou/findings   5 treatment/state classes (zhou.FINDING_CLASSES), one box per label
Both use the same 70/15/15 image-level split (seed 42).

Non-commercial research only; never on GitHub. See docs/datasets/zhou.md.

Usage:
    python scripts/zhou_to_yolo.py --raw data/raw/zhou --out data/yolo/zhou
    python scripts/zhou_to_yolo.py --list-labels      # check every label is a tooth number
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from opg.data import dentex as D
from opg.data import zhou as Z

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/zhou"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo/zhou"))
    parser.add_argument("--list-labels", action="store_true")
    args = parser.parse_args()

    if args.list_labels:
        for label, count in sorted(Z.count_labels(args.raw).items()):
            fdi = Z.tooth_fdi(label)
            print(f"{count:6d}  {label!r:8} -> {fdi if fdi is not None else '!! SKIPPED (not 11-48)'}")
        return

    records = Z.load_zhou(args.raw)
    splits = D.split_records(records, FRACTIONS, 42)
    teeth_yaml = D.write_yolo(splits, args.out / "teeth", D.TEETH_CLASSES, "fdi")
    finding_splits = {split: Z.finding_records(rs) for split, rs in splits.items()}
    findings_yaml = D.write_yolo(finding_splits, args.out / "findings", Z.FINDING_CLASSES, "label")

    skipped = {label: n for label, n in Z.count_labels(args.raw).items() if Z.tooth_fdi(label) is None}
    teeth = sum(len(r["boxes"]) for r in records)
    print(f"zhou: {len(records)} images, {teeth} teeth -> {teeth_yaml}")
    print(f"zhou: findings -> {findings_yaml}")
    print(f"zhou: skipped labels (not a tooth 11-48): {skipped or 'none'}")
    print(f"zhou: tooth states: {dict(sorted(Counter(tuple(b['labels']) for r in records for b in r['boxes']).items()))}")
    print("\nsplit   images  teeth  " + "  ".join(f"{c:>18}" for c in Z.FINDING_CLASSES))
    for split, rs in finding_splits.items():
        found = Counter(b["label"] for r in rs for b in r["boxes"])
        n_teeth = sum(len(r["boxes"]) for r in splits[split])
        print(f"{split:6} {len(rs):7d} {n_teeth:6d}  " + "  ".join(f"{found[c]:18d}" for c in Z.FINDING_CLASSES))
    repeated = [r for r in records if D.repeated_numbers(r)]
    print(f"\nzhou: {len(repeated)} images repeat a tooth number (labelling mistakes; kept as labelled):")
    for r in repeated:
        print(f"    {r['file']}\trepeated: {' '.join(map(str, D.repeated_numbers(r)))}")


if __name__ == "__main__":
    main()
