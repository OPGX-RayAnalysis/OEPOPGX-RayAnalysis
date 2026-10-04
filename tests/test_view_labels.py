import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from opg.data import yolo as Y

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(tmp_path: Path, with_path_line: bool = False) -> Path:
    """A tiny YOLO dataset: one 200x100 image with tooth 18 (class 7) on the left."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images" / "train" / "a.png"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels" / "train" / "a.txt").write_text("7 0.100000 0.250000 0.100000 0.200000\n")
    names = "\n".join(f"  {i}: '{q * 10 + n}'" for i, (q, n) in enumerate((q, n) for q in (1, 2, 3, 4) for n in range(1, 9)))
    head = f"path: {tmp_path.as_posix()}\n" if with_path_line else ""
    (tmp_path / "data.yaml").write_text(f"{head}train: images/train\nnames:\n{names}\n")
    return tmp_path / "data.yaml"


def test_reads_yaml_and_labels(tmp_path):
    data = Y.load_data_yaml(make_dataset(tmp_path))
    assert data["root"] == tmp_path and data["names"][7] == "18" and len(data["names"]) == 32
    [image] = Y.split_images(data, "train")
    [box] = Y.read_labels(image, data["names"], 200, 100)
    assert box.fdi == 18
    assert [round(v) for v in box.xyxy] == [10, 15, 30, 35]


def test_absolute_path_line_still_works(tmp_path):
    data = Y.load_data_yaml(make_dataset(tmp_path, with_path_line=True))
    assert len(Y.split_images(data, "train")) == 1


def test_simple_yaml_fallback_matches(tmp_path):
    text = make_dataset(tmp_path).read_text()
    parsed = Y._parse_simple_yaml(text)
    assert parsed["train"] == "images/train" and parsed["names"][31] == "48"
    assert Y._parse_simple_yaml("names: ['11', '12']\n")["names"] == ["11", "12"]


def test_script_writes_images(tmp_path):
    yaml_path = make_dataset(tmp_path / "ds")
    out = tmp_path / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "view_labels.py"), "--data", str(yaml_path),
           "--out", str(out), "--system", "universal"]
    subprocess.run(cmd, check=True, capture_output=True)
    img = cv2.imread(str(out / "a.jpg"))
    assert img is not None and img.shape[:2] == (100, 200)
