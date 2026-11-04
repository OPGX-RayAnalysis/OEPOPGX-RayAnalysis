import json

import cv2
import numpy as np
import pytest

from opg.data import dentex as D
from opg.data import zhou as Z


def make_fake_zhou(root):
    """One blank 200x100 image in the real Zhou layout with three teeth: 18 healthy (image left),
    36 with a filling, 46 with a root canal filling and a crown. Plus a '91' shape (not a tooth)
    and a cw*.json with no image, both of which the reader must ignore."""
    (root / "images1").mkdir(parents=True)
    (root / "labels").mkdir()
    cv2.imwrite(str(root / "images1" / "0001.png"), np.zeros((100, 200), np.uint8))
    shapes = [
        {"label": "18", "group_id": None, "points": [[10, 5], [30, 8], [20, 40]], "shape_type": "polygon"},
        {"label": "36", "group_id": 1, "points": [[120, 60], [140, 62], [130, 90]], "shape_type": "polygon"},
        {"label": "46", "group_id": 6, "points": [[50, 55], [75, 58], [60, 95]], "shape_type": "polygon"},
        {"label": "91", "group_id": None, "points": [[0, 0], [5, 0], [5, 5]], "shape_type": "polygon"},
    ]
    annotation = {"shapes": shapes, "imagePath": "0001.png", "imageWidth": 200, "imageHeight": 100}
    (root / "labels" / "0001.json").write_text(json.dumps(annotation))
    (root / "labels" / "cw0001.json").write_text(json.dumps({**annotation, "imagePath": "cw0001.png"}))


def test_load_zhou_boxes_fdi_and_labels(tmp_path):
    make_fake_zhou(tmp_path)
    [record] = Z.load_zhou(tmp_path)
    assert (record["file"], record["width"], record["height"]) == ("0001.png", 200, 100)
    assert record["boxes"] == [
        {"bbox": [10, 5, 30, 40], "fdi": 18, "labels": []},
        {"bbox": [120, 60, 140, 90], "fdi": 36, "labels": ["filling"]},
        {"bbox": [50, 55, 75, 95], "fdi": 46, "labels": ["root_canal_filling", "crown"]},
    ]


def test_teeth_and_findings_sets(tmp_path):
    make_fake_zhou(tmp_path / "raw")
    records = Z.load_zhou(tmp_path / "raw")
    splits = {"train": records}
    D.write_yolo(splits, tmp_path / "teeth", D.TEETH_CLASSES, "fdi")
    D.write_yolo({s: Z.finding_records(rs) for s, rs in splits.items()}, tmp_path / "findings", Z.FINDING_CLASSES, "label")

    teeth = (tmp_path / "teeth" / "labels" / "train" / "0001.txt").read_text().splitlines()
    assert sorted(D.TEETH_CLASSES[int(line.split()[0])] for line in teeth) == ["18", "36", "46"]

    findings = (tmp_path / "findings" / "labels" / "train" / "0001.txt").read_text().splitlines()
    assert sorted(Z.FINDING_CLASSES[int(line.split()[0])] for line in findings) == ["crown", "filling", "root_canal_filling"]


def test_unknown_state_is_an_error(tmp_path):
    make_fake_zhou(tmp_path)
    path = tmp_path / "labels" / "0001.json"
    d = json.loads(path.read_text())
    d["shapes"][0]["group_id"] = 9
    path.write_text(json.dumps(d))
    with pytest.raises(KeyError, match="9"):
        Z.load_zhou(tmp_path)


def test_count_labels_ignores_cw_files(tmp_path):
    make_fake_zhou(tmp_path)
    assert Z.count_labels(tmp_path) == {"18": 1, "36": 1, "46": 1, "91": 1}
