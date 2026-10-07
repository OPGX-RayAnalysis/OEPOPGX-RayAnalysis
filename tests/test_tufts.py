import json

import cv2
import numpy as np

from opg.data import dentex as D
from opg.data import tufts as T


SPECK_FAR_AWAY = [[180, 90], [181, 90], [181, 91], [180, 91]]  # a few pixels, far from the tooth
TOOTH_IN_TWO_PIECES = [[[12, 60], [35, 55], [25, 75]], [[15, 78], [30, 78], [25, 95]]]


def make_fake_tufts(root):
    """Two 200x100 images in the real Tufts format. 7.JPG: Universal 1 (FDI 18, image left) with a
    stray speck, and 32 (FDI 48) traced in two pieces. 8.JPG: no teeth (edentulous patient)."""
    (root / "Radiographs").mkdir(parents=True)
    (root / "Segmentation").mkdir()
    for name in ("7.JPG", "8.JPG"):
        cv2.imwrite(str(root / "Radiographs" / name), np.zeros((100, 200), np.uint8))
    entries = [
        {
            "External ID": "7.jpg",  # lower case on purpose, like the real export
            "Label": {
                "objects": [
                    # "bounding box" is [y1, x1, y2, x2] and unreliable; the reader must not use it.
                    {"title": "1", "bounding box": [0, 0, 1, 1], "polygons": [[[10, 5], [30, 8], [20, 40]], SPECK_FAR_AWAY]},
                    {"title": "32", "bounding box": [0, 0, 1, 1], "polygons": TOOTH_IN_TWO_PIECES},
                    {"title": "not a tooth", "bounding box": [0, 0, 1, 1], "polygons": [[[0, 0], [5, 0], [5, 5]]]},
                ]
            },
        },
        {"External ID": "8.jpg", "Label": {}},
    ]
    (root / "Segmentation" / "teeth_polygon.json").write_text(json.dumps(entries))


def test_load_tufts_boxes_and_fdi(tmp_path):
    make_fake_tufts(tmp_path)
    record, edentulous = T.load_tufts(tmp_path)
    assert record["file"] == "7.JPG" and (record["width"], record["height"]) == (200, 100)
    assert record["boxes"] == [
        {"bbox": [10, 5, 30, 40], "fdi": 18, "label": None},  # speck ignored
        {"bbox": [12, 55, 35, 95], "fdi": 48, "label": None},  # both pieces kept
    ]
    assert edentulous["file"] == "8.JPG" and edentulous["boxes"] == []


def test_universal_converts_with_patient_right_on_image_left():
    assert [T.tooth_fdi(u) for u in ("1", "8", "9", "16", "17", "24", "25", "32")] == [18, 11, 21, 28, 38, 31, 41, 48]
    assert T.tooth_fdi("A") == 55 and T.tooth_fdi("not a tooth") is None


def test_count_titles_shows_unknown_titles(tmp_path):
    make_fake_tufts(tmp_path)
    assert T.count_titles(tmp_path) == {"1": 1, "32": 1, "not a tooth": 1}


def test_written_yolo_uses_dentex_class_order(tmp_path):
    make_fake_tufts(tmp_path / "raw")
    D.write_yolo({"test": T.load_tufts(tmp_path / "raw")[:1]}, tmp_path / "out", D.TEETH_CLASSES, "fdi")
    lines = (tmp_path / "out" / "labels" / "test" / "7.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in lines] == [D.TEETH_CLASSES.index("18"), D.TEETH_CLASSES.index("48")]
