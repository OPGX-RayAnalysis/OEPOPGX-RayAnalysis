import json

from opg.data import dentex as D

COCO = {
    "images": [{"id": 7, "file_name": "train_1.png", "width": 2000, "height": 1000}],
    # Category ids are indices, names carry the meaning. Ids deliberately shuffled.
    "categories_1": [{"id": 0, "name": "3"}, {"id": 1, "name": "1"}],
    "categories_2": [{"id": 0, "name": "8"}, {"id": 1, "name": "6"}],
    "categories_3": [{"id": 0, "name": "Deep Caries"}, {"id": 1, "name": "Impacted"}],
    "annotations": [
        {"image_id": 7, "bbox": [100, 200, 50, 80], "category_id_1": 0, "category_id_2": 0, "category_id_3": 1},
        {"image_id": 7, "bbox": [400, 100, 60, 90], "category_id_1": 1, "category_id_2": 1, "category_id_3": 0},
    ],
}


def test_coco_hierarchical_uses_category_names(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(COCO))
    [record] = D.load_coco_hierarchical(tmp_path / "a.json", tmp_path)
    assert [(b["fdi"], b["label"]) for b in record["boxes"]] == [(38, "impacted"), (16, "deep_caries")]
    assert record["boxes"][0]["bbox"] == [100, 200, 150, 280]


def test_labelme_test_label():
    assert D.parse_test_label("1-çürük-33") == ("çürük", 33)


def test_yolo_line_normalised():
    assert D.yolo_line([100, 200, 150, 280], 3, 2000, 1000) == "3 0.062500 0.240000 0.025000 0.080000"


def test_split_is_deterministic_and_complete():
    records = [{"file": f"{i}.png"} for i in range(100)]
    a = D.split_records(records, {"train": 0.85, "val": 0.15}, seed=1)
    b = D.split_records(list(reversed(records)), {"train": 0.85, "val": 0.15}, seed=1)
    assert [r["file"] for r in a["val"]] == [r["file"] for r in b["val"]]
    assert len(a["train"]) == 85 and len(a["val"]) == 15


def test_teeth_classes():
    assert len(D.TEETH_CLASSES) == 32 and D.TEETH_CLASSES[0] == "11" and D.TEETH_CLASSES[-1] == "48"
