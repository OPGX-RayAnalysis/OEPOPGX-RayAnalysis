import json

import pytest

from opg.data import oralxrays as O
from opg.schema import FINDING_LABELS


def write_split(root, categories, annotations):
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    doc = {"images": [{"id": 1, "file_name": "000001.jpg", "width": 1991, "height": 1127},
                      {"id": 2, "file_name": "000002.jpg", "width": 1991, "height": 1127}],
           "annotations": annotations, "categories": categories}
    (root / "annotations" / "instances_train2017.json").write_text(json.dumps(doc), encoding="utf-8")


CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(
    ["Apical Periodontitis", "Decay", "Wisdom Tooth", "Missing Tooth", "Dental Filling",
     "Root Canal Filling", "Implant", "Porcelain Crown", "Ceramic Bridge"])]


def test_classes_are_ours():
    assert set(O.CLASSES) <= set(FINDING_LABELS)
    assert set(O.CLASS_MAP) | O.DROPPED == {c["name"] for c in CATEGORIES}


def test_load_split_maps_drops_and_counts(tmp_path):
    ann = [{"image_id": 1, "category_id": 2, "bbox": [10, 20, 30, 40]},   # Decay -> caries
           {"image_id": 1, "category_id": 3, "bbox": [0, 0, 5, 5]},      # Wisdom Tooth: left out
           {"image_id": 1, "category_id": 7, "bbox": [50, 60, 10, 80]},  # Implant
           {"image_id": 2, "category_id": 0, "bbox": [0, 0, 5, 5]}]      # id 0: not a class
    write_split(tmp_path, CATEGORIES, ann)
    records, dropped = O.load_split(tmp_path, "train2017")
    assert [(b["label"], b["bbox"]) for b in records[0]["boxes"]] == [("caries", [10, 20, 40, 60]), ("implant", [50, 60, 60, 140])]
    assert records[1]["boxes"] == [] and records[0]["path"] == tmp_path / "train2017" / "000001.jpg"
    assert dropped == {"Wisdom Tooth": 1, "unknown category id 0": 1}


def test_new_class_name_is_an_error(tmp_path):
    write_split(tmp_path, CATEGORIES + [{"id": 10, "name": "Fracture"}], [{"image_id": 1, "category_id": 10, "bbox": [0, 0, 1, 1]}])
    with pytest.raises(KeyError):
        O.load_split(tmp_path, "train2017")
