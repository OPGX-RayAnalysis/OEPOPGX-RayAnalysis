import csv
import importlib.util
import math
from pathlib import Path

from opg import fdi as F
from opg.scoring import (
    CI_METRICS, IMAGE_FIELDS, PRED_FIELDS, Scorer, Tooth, average_precision, bootstrap_ci, match, paired_difference_ci,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_scores", ROOT / "scripts" / "compare_scores.py")
compare_scores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_scores)


def row(teeth: list[int]) -> list[Tooth]:
    """Non-overlapping 10x10 boxes side by side, one per tooth."""
    return [Tooth(t, [20 * i, 0, 20 * i + 10, 10]) for i, t in enumerate(teeth)]


def test_perfect_predictions_score_100():
    truth = row(list(F.ALL_PERMANENT))
    s = Scorer()
    s.add(truth, row(list(F.ALL_PERMANENT)))
    r = s.result()
    assert r["numbering_accuracy"] == r["detection_rate"] == r["accuracy_if_found"] == 1.0
    assert math.isclose(r["map50"], 1.0)
    assert r["missing_tooth_f1"] is None  # nothing missing anywhere: F1 undefined
    assert len(r["per_tooth"]) == 32


def test_swapped_number_and_missed_tooth():
    truth = row([11, 12, 13, 14])
    pred = row([11, 13, 13])  # 12 called 13, 14 not found
    s = Scorer()
    s.add(truth, pred)
    r = s.result()
    assert r["detection_rate"] == 0.75
    assert r["numbering_accuracy"] == 0.5
    assert math.isclose(r["accuracy_if_found"], 2 / 3)
    per = {x["fdi"]: x for x in r["per_tooth"]}
    assert per[12] == {"fdi": 12, "count": 1, "detected": 1, "correct": 0, "accuracy": 0.0}
    assert per[14]["detected"] == 0


def test_missing_tooth_f1():
    # 28 truly missing teeth; the model also says 12 and 14 are missing (12 wrongly numbered as 13).
    truth = row([11, 12, 13, 14])
    s = Scorer()
    s.add(truth, row([11, 13, 13]))
    m = s.result()["missing_counts"]
    assert m == {"tp": 28, "fp": 2, "fn": 0}
    assert math.isclose(s.result()["missing_tooth_f1"], 56 / 58)


def test_match_is_one_to_one_and_ignores_numbers():
    truth = [Tooth(11, [0, 0, 10, 10])]
    pred = [Tooth(21, [0, 0, 10, 10]), Tooth(11, [1, 0, 11, 10])]
    assert match(truth, pred) == [(0, 0)]
    assert match(truth, [Tooth(11, [8, 0, 18, 10])]) == []  # too little overlap


def test_average_precision():
    assert average_precision([(0.9, True), (0.8, True)], 2) == 1.0
    assert average_precision([(0.9, False), (0.8, True)], 1) == 0.5
    assert average_precision([(0.9, True)], 2) == 0.5
    assert math.isnan(average_precision([], 0))


def image_row(name, teeth, correct, detected=None, extra=0):
    detected = teeth if detected is None else detected
    return {"image": name, "teeth": teeth, "detected": detected, "correct": correct,
            "missing_tp": 0, "missing_fp": 0, "missing_fn": 0, "extra": extra}


def test_per_image_rows_add_up_to_result():
    s = Scorer()
    s.add(row([11, 12, 13, 14]), row([11, 13, 13]), name="a.png")
    s.add(row([21, 22]), row([21, 22]), name="b.png")
    assert [r["image"] for r in s.images] == ["a.png", "b.png"]
    assert s.images[0] == {"image": "a.png", "teeth": 4, "detected": 3, "correct": 2,
                           "missing_tp": 28, "missing_fp": 2, "missing_fn": 0, "extra": 0}
    whole, rows = s.result(), summarize(s.images)
    for m in CI_METRICS:
        assert math.isclose(whole[m], rows[m])


def test_bootstrap_ci_brackets_the_estimate_and_is_repeatable():
    images = [image_row(f"{i}.png", 30, 30 - i % 4) for i in range(93)]
    ci = bootstrap_ci(images, n=300)
    acc = summarize(images)["numbering_accuracy"]
    lo, hi = ci["numbering_accuracy"]
    assert lo < acc < hi
    assert ci == bootstrap_ci(images, n=300)  # fixed seed
    assert ci["missing_tooth_f1"] is None  # nothing missing anywhere


def test_bootstrap_ci_is_zero_width_when_every_image_is_the_same():
    images = [image_row(f"{i}.png", 10, 9) for i in range(20)]
    assert bootstrap_ci(images, n=100)["numbering_accuracy"] == [0.9, 0.9]


def test_paired_difference_uses_shared_images_only():
    a = [image_row(f"{i}.png", 10, 8) for i in range(30)]
    b = [image_row(f"{i}.png", 10, 9) for i in range(30)] + [image_row("extra.png", 10, 0)]
    d = paired_difference_ci(a, b, n=200)
    assert d["images"] == 30
    assert math.isclose(d["numbering_accuracy"]["diff"], 0.1)
    lo, hi = d["numbering_accuracy"]["ci"]
    assert math.isclose(lo, 0.1) and math.isclose(hi, 0.1)


def test_whole_scan_needs_every_tooth_right_and_nothing_extra():
    s = Scorer()
    s.add(row([11, 12]), row([11, 12]), name="perfect.png")
    s.add(row([11, 12]), row([11, 12, 13]), name="extra.png")  # 13 sits where no real tooth is
    s.add(row([11, 12]), row([11, 21]), name="wrong.png")
    s.add(row([11, 12]), row([11]), name="missed.png")
    s.add([], [], name="edentulous.png")  # nothing to find, nothing claimed: right
    assert [r["extra"] for r in s.images] == [0, 1, 0, 0, 0]
    r = s.result()
    assert math.isclose(r["whole_scan_accuracy"], 2 / 5)
    assert r["extra_teeth"] == 1
    assert math.isclose(summarize(s.images)["whole_scan_accuracy"], 2 / 5)


def test_one_pred_row_per_scored_prediction():
    s = Scorer()
    truth = row([11, 12, 13])
    pred = [Tooth(11, [0, 0, 10, 10], 0.91234), Tooth(13, [20, 0, 30, 10], 0.8), Tooth(14, [200, 0, 210, 10], 0.3)]
    s.add(truth, pred, pred_all=pred + [Tooth(15, [300, 0, 310, 10], 0.01)], name="a.png")
    assert list(s.preds[0]) == PRED_FIELDS
    assert s.preds == [
        {"image": "a.png", "fdi": 11, "confidence": 0.9123, "matched": 1, "correct": 1},
        {"image": "a.png", "fdi": 13, "confidence": 0.8, "matched": 1, "correct": 0},  # on the real 12
        {"image": "a.png", "fdi": 14, "confidence": 0.3, "matched": 0, "correct": 0},  # extra
    ]  # pred_all (mAP only) adds no rows


def test_rows_without_extra_give_no_whole_scan_but_the_rest_still_works():
    old = [{k: v for k, v in image_row(f"{i}.png", 10, 9).items() if k != "extra"} for i in range(20)]
    s = summarize(old)
    assert s["whole_scan_accuracy"] is None
    assert math.isclose(s["numbering_accuracy"], 0.9)
    assert bootstrap_ci(old, n=50)["whole_scan_accuracy"] is None
    assert paired_difference_ci(old, old, n=50)["whole_scan_accuracy"] == {"diff": None, "ci": None}


def test_compare_scores_reads_old_and_new_images_csv(tmp_path):
    for name, fields in [("old", IMAGE_FIELDS[:-1]), ("new", IMAGE_FIELDS)]:
        (tmp_path / name).mkdir()
        with open(tmp_path / name / "score_images.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow({k: v for k, v in image_row("a.png", 10, 10).items() if k in fields})
    _, old_rows = compare_scores.read_images(tmp_path / "old", rules=False)
    _, new_rows = compare_scores.read_images(tmp_path / "new", rules=False)
    assert summarize(old_rows)["whole_scan_accuracy"] is None
    assert summarize(new_rows)["whole_scan_accuracy"] == 1.0
