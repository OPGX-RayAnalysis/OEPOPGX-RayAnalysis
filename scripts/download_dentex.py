"""Download DENTEX from Hugging Face, fetching only the subsets we need (plan task 1.6).

training_data.zip is ~10.9 GB, but most of that is unlabelled images. This script reads
the zip's index over HTTP range requests and extracts only the chosen folders:

    enumeration  quadrant + tooth-number labels, all teeth   (634 imgs, ~1.7 GB)  -> tooth model
    disease      quadrant + number + diagnosis, abnormal teeth (705 imgs, ~2.2 GB) -> findings model
    test         official 250-image test set with labels     (~0.8 GB)            -> final evaluation
    quadrant     quadrant-only labels                         (693 imgs, ~1.9 GB)  -> optional
    unlabelled   no labels                                    (1571 imgs, ~5.2 GB) -> optional pre-training

Usage:
    python scripts/download_dentex.py                      # enumeration + disease + test
    python scripts/download_dentex.py --subsets enumeration
    python scripts/download_dentex.py --out data/raw/dentex

Re-running skips files that already exist, so an interrupted download can be resumed.
Dataset licence: CC BY-NC-SA 4.0 (non-commercial, cite the DENTEX papers).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/"
SUBSETS = {
    "enumeration": ("training_data.zip", "training_data/quadrant_enumeration/"),
    "disease": ("training_data.zip", "training_data/quadrant-enumeration-disease/"),
    "quadrant": ("training_data.zip", "training_data/quadrant/"),
    "unlabelled": ("training_data.zip", "training_data/unlabelled/"),
    "test": ("test_data.zip", "disease/"),
}


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file backed by HTTP range requests."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(request, timeout=120).read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def extract(zip_name: str, prefix: str, out: Path) -> None:
    archive = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(BASE + zip_name), buffer_size=4 << 20))
    members = [m for m in archive.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
    members = [m for m in members if ".ipynb_checkpoints" not in m.filename]
    total = sum(m.file_size for m in members)
    done = 0
    print(f"{zip_name}:{prefix}  {len(members)} files, {total / 1e9:.2f} GB")
    for i, m in enumerate(members, 1):
        target = out / m.filename
        done += m.file_size
        if target.exists() and target.stat().st_size == m.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(m))
        if i % 25 == 0 or i == len(members):
            print(f"  {i}/{len(members)}  {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", nargs="+", default=["enumeration", "disease", "test"], choices=SUBSETS)
    parser.add_argument("--out", type=Path, default=Path("data/raw/dentex"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.subsets:
        zip_name, prefix = SUBSETS[name]
        try:
            extract(zip_name, prefix, args.out)
        except Exception as e:  # network hiccups: re-running resumes
            sys.exit(f"Failed on subset '{name}': {e}\nRe-run the same command to resume.")
    print(f"Done. Data in {args.out.resolve()}")


if __name__ == "__main__":
    main()
