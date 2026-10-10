"""Convert the Tufts Dental Database into a YOLO test set for tooth numbering (plan task 1.14).

    data/yolo/tufts_test   32 FDI classes (same order as DENTEX), one split: test.
                           External test set: never train or tune on it.
                           mixed_dentition.txt lists the images with primary teeth, for
                           score_numbering.py --skip (report all 1,000 and the rest).

Usage:
    python scripts/tufts_to_yolo.py --raw data/raw/tufts --out data/yolo/tufts_test
    python scripts/tufts_to_yolo.py --list-titles      # check every title is a Universal number
    python scripts/tufts_to_yolo.py --annotations data/raw/tufts/Segmentation/other.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opg import fdi as F
from opg.data import dentex as D
from opg.data import tufts as T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tufts"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo/tufts_test"))
    parser.add_argument("--annotations", type=Path, default=None, help=f"default: <raw>/{T.ANNOTATION_FILE.as_posix()}")
    parser.add_argument("--list-titles", action="store_true")
    args = parser.parse_args()

    if args.list_titles:
        for title, count in sorted(T.count_titles(args.raw, args.annotations).items()):
            fdi = T.tooth_fdi(title)
            print(f"{count:6d}  {title!r:12} -> {fdi if fdi is not None else '!! NOT A TOOTH NUMBER'}")
        return

    records = T.load_tufts(args.raw, args.annotations)
    path = D.write_yolo({"test": records}, args.out, D.TEETH_CLASSES, "fdi")

    teeth = [b["fdi"] for r in records for b in r["boxes"]]
    primary = sum(F.is_primary(f) for f in teeth)
    mixed = [r["file"] for r in records if any(F.is_primary(b["fdi"]) for b in r["boxes"])]
    (args.out / "mixed_dentition.txt").write_text("\n".join(mixed) + "\n", encoding="utf-8")
    repeated = [r for r in records if D.repeated_numbers(r)]
    print(f"tufts: {len(records)} images, {len(teeth) - primary} permanent teeth -> {path}")
    print(f"tufts: {sum(not r['boxes'] for r in records)} images with no teeth (edentulous; empty label files)")
    print(f"tufts: {primary} primary teeth in {len(mixed)} images left out (no class for them; DENTEX is adults only)")
    print(f"tufts: those {len(mixed)} images are listed in {args.out / 'mixed_dentition.txt'}")
    print(f"tufts: {len(repeated)} images repeat a tooth number (labelling mistakes):")
    for r in repeated:
        print(f"    {r['file']}\trepeated: {' '.join(map(str, D.repeated_numbers(r)))}")


if __name__ == "__main__":
    main()
