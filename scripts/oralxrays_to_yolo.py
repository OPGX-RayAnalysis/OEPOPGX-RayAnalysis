"""Convert OralXrays-9 into a YOLO findings dataset (task 2.17).

    data/yolo/oralxrays   7 classes mapped onto ours: caries, periapical_lesion, filling,
                          root_canal_filling, crown, bridge, implant (Wisdom Tooth and
                          Missing Tooth are left out; see src/opg/data/oralxrays.py)
                          train / val = the authors' train2017 split 90/10 (seed 42),
                          test = the authors' val2017 split (their benchmark split), untouched.
                          Images that also appear in DENTEX are left out (excluded_oralxrays.txt).
                          test_seen_in_train.txt lists test images that repeat a train image.

Uses: outside test of the DENTEX findings model (test split, caries + periapical lesion,
--merge-caries, scored with --match iou AND --match cover), and training data for treatments
and implants (task 2.7).

Usage (after unzipping the authors' release into data/raw/oralxrays with their password):
    python scripts/oralxrays_to_yolo.py
    python scripts/oralxrays_to_yolo.py --raw data/raw/oralxrays --out data/yolo/oralxrays --dentex data/raw/dentex
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from opg.data import dentex as D
from opg.data import oralxrays as O
from opg.data.yolo import IMAGE_SUFFIXES, fingerprint

SAME_IMAGE = 3  # fingerprints differing in at most this many of 128 places = the same X-ray


def nearest(a: np.ndarray, b: np.ndarray, chunk: int = 100) -> np.ndarray:
    """For each fingerprint in a, the fewest places in which it differs from any in b."""
    out = []
    for i in range(0, len(a), chunk):
        out.append((a[i:i + chunk, None, :] != b[None, :, :]).sum(axis=2).min(axis=1))
    return np.concatenate(out) if out else np.array([], dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/oralxrays"))
    parser.add_argument("--out", type=Path, default=Path("data/yolo/oralxrays"))
    parser.add_argument("--dentex", type=Path, default=Path("data/raw/dentex"), help="raw DENTEX, for the overlap check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, dropped_train = O.load_split(args.raw, "train2017")
    test, dropped_test = O.load_split(args.raw, "val2017")
    dropped = dropped_train + dropped_test
    print(f"oralxrays: left out {sum(dropped.values())} boxes: "
          + ", ".join(f"{k} {v}" for k, v in dropped.most_common()))
    splits = D.split_records(train, {"train": 0.9, "val": 0.1}, args.seed)
    splits["test"] = test

    print("oralxrays: fingerprinting images for the DENTEX overlap check (a few minutes)", flush=True)
    dentex = sorted(p for p in args.dentex.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    dentex_fp = np.array([np.frombuffer(fingerprint(p), dtype=np.uint8) for p in dentex])
    excluded = []
    for name, records in splits.items():
        fps = np.array([np.frombuffer(fingerprint(r["path"]), dtype=np.uint8) for r in records])
        dist = nearest(fps, dentex_fp) if len(dentex_fp) else np.full(len(records), 128)
        excluded += [(name, r, dentex[int(np.argmin((dentex_fp != f).sum(axis=1)))], int(d))
                     for r, f, d in zip(records, fps, dist) if d <= SAME_IMAGE]
        splits[name] = [r for r, d in zip(records, dist) if d > SAME_IMAGE]
        if name == "train":
            train_fp = fps
        if name == "test":
            seen = [r["file"] for r, d in zip(records, nearest(fps, train_fp)) if d <= SAME_IMAGE]
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "test_seen_in_train.txt").write_text("".join(f + "\n" for f in seen), encoding="utf-8")
            print(f"oralxrays: {len(seen)} test images look the same as a train image (authors' split, kept;"
                  f" listed in test_seen_in_train.txt for --skip when a model trained on OralXrays-9 is scored)")
    lines = [f"{name}\t{r['file']}\tsame image as DENTEX {p.name} ({d} of 128 differ)\n" for name, r, p, d in excluded]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "excluded_oralxrays.txt").write_text("".join(lines), encoding="utf-8")
    print(f"oralxrays: checked against {len(dentex)} DENTEX images, left out {len(excluded)} (excluded_oralxrays.txt)")

    path = D.write_yolo(splits, args.out, O.CLASSES, "label")
    print(f"oralxrays: {', '.join(f'{k}={len(v)}' for k, v in splits.items())} -> {path}")
    for name, records in splits.items():
        counts = Counter(b["label"] for r in records for b in r["boxes"])
        print(f"  {name:5} " + ", ".join(f"{c} {counts[c]}" for c in O.CLASSES)
              + f"; {sum(not r['boxes'] for r in records)} images with no box")


if __name__ == "__main__":
    main()
