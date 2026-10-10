import math

from opg import fdi as F
from opg.scoring import CI_METRICS, Scorer, Tooth, average_precision, bootstrap_ci, match, paired_difference_ci, summarize


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


def image_row(name, teeth, correct, detected=None):
    detected = teeth if detected is None else detected
    return {"image": name, "teeth": teeth, "detected": detected, "correct": correct,
            "missing_tp": 0, "missing_fp": 0, "missing_fn": 0}


def test_per_image_rows_add_up_to_result():
    s = Scorer()
    s.add(row([11, 12, 13, 14]), row([11, 13, 13]), name="a.png")
    s.add(row([21, 22]), row([21, 22]), name="b.png")
    assert [r["image"] for r in s.images] == ["a.png", "b.png"]
    assert s.images[0] == {"image": "a.png", "teeth": 4, "detected": 3, "correct": 2,
                           "missing_tp": 28, "missing_fp": 2, "missing_fn": 0}
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
