import csv
import importlib.util
import math
from pathlib import Path

from opg.scoring_findings import (
    FINDING_IMAGE_FIELDS, MERGE_CARIES, Finding, FindingScorer, bootstrap_findings_ci, normalize, paired_findings_difference,
    summarize_findings,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_findings", ROOT / "scripts" / "score_findings.py")
score_findings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score_findings)

CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted"]


def at(name: str, i: int, conf: float = 1.0) -> Finding:
    """A 10x10 box in slot i (slots never overlap)."""
    return Finding(name, [20 * i, 0, 20 * i + 10, 10], conf)


def test_perfect_predictions():
    truth = [at("caries", 0), at("impacted", 1)]
    s = FindingScorer(CLASSES)
    s.add(truth, [at("caries", 0, 0.9), at("impacted", 1, 0.8)])
    r = s.result()
    assert r["per_class"]["caries"]["f1"] == r["per_class"]["impacted"]["f1"] == 1.0
    assert r["per_class"]["deep_caries"]["f1"] is None  # never present, never predicted
    assert r["mean_f1"] == 1.0 and r["pooled"]["f1"] == 1.0
    assert math.isclose(r["map50"], 1.0)


def test_invented_missed_and_wrong_class():
    truth = [at("caries", 0), at("caries", 1), at("impacted", 2)]
    pred = [at("caries", 0), at("deep_caries", 1), at("periapical_lesion", 3)]
    s = FindingScorer(CLASSES)
    s.add(truth, pred)
    r = s.result()["per_class"]
    # Slot 1: right box, wrong class -> fn for caries AND fp for deep_caries.
    assert {k: r["caries"][k] for k in ("truth", "tp", "fp", "fn")} == {"truth": 2, "tp": 1, "fp": 0, "fn": 1}
    assert (r["deep_caries"]["fp"], r["periapical_lesion"]["fp"], r["impacted"]["fn"]) == (1, 1, 1)
    assert math.isclose(r["caries"]["f1"], 2 / 3)
    assert r["caries"]["precision"] == 1.0 and r["caries"]["recall"] == 0.5


def test_mean_f1_weights_classes_equally_pooled_does_not():
    truth = [at("caries", i) for i in range(9)] + [at("impacted", 9)]
    pred = [at("caries", i) for i in range(9)]  # all caries found, the one impacted tooth missed
    s = FindingScorer(CLASSES)
    s.add(truth, pred)
    r = s.result()
    assert r["mean_f1"] == 0.5  # (1.0 + 0.0) / 2
    assert math.isclose(r["pooled"]["f1"], 18 / 19)


def test_merge_caries_counts_deep_as_caries():
    truth = [at("deep_caries", 0)]
    pred = [at("caries", 0)]
    s = FindingScorer(["caries", "periapical_lesion", "impacted"], merge=MERGE_CARIES)
    s.add(truth, pred)
    assert s.result()["per_class"]["caries"]["tp"] == 1


def test_unknown_classes_are_ignored_and_preds_rows_kept():
    s = FindingScorer(["caries"])
    s.add([at("caries", 0), at("filling", 1)], [at("caries", 0, 0.7), at("crown", 2, 0.6), at("caries", 3, 0.4)])
    assert s.result()["per_class"]["caries"] == {
        "truth": 1, "tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0, "f1": 2 / 3, "ap50": 1.0,
    }
    assert [(p["confidence"], p["matched"]) for p in s.preds] == [(0.7, 1), (0.4, 0)]


def test_every_image_gets_a_row_per_class():
    s = FindingScorer(CLASSES)
    s.add([], [], name="a.png")
    s.add([at("caries", 0)], [], name="b.png")
    assert len(s.images) == 8
    assert summarize_findings(s.images)["images"] == 2


def test_bootstrap_brackets_observed_and_same_images_same_interval():
    s = FindingScorer(CLASSES)
    for i in range(20):
        s.add([at("caries", 0)], [at("caries", 0)] if i % 4 else [], name=f"{i}.png")
    ci = bootstrap_findings_ci(s.images, n=200)
    lo, hi = ci["per_class"]["caries"]["recall"]
    assert lo <= 0.75 <= hi
    assert ci["per_class"]["caries"]["precision"] == [1.0, 1.0]
    assert ci == bootstrap_findings_ci(s.images, n=200)


def test_paired_difference_identical_runs_is_zero():
    s = FindingScorer(CLASSES)
    for i in range(10):
        s.add([at("caries", 0), at("impacted", 1)], [at("caries", 0)] if i % 2 else [], name=f"{i}.png")
    d = paired_findings_difference(s.images, s.images, n=100)
    assert d["images"] == 10
    assert d["mean_f1"] == {"diff": 0.0, "ci": [0.0, 0.0]}
    assert d["caries_f1"]["diff"] == 0.0


def test_paired_difference_better_model():
    a, b = FindingScorer(["caries"]), FindingScorer(["caries"])
    for i in range(30):
        a.add([at("caries", 0)], [], name=f"{i}.png")
        b.add([at("caries", 0)], [at("caries", 0)], name=f"{i}.png")
    d = paired_findings_difference(a.images, b.images, n=100)
    assert d["caries_f1"]["diff"] == 1.0 and d["caries_f1"]["ci"][0] > 0


def test_names_are_normalised_and_merged_in_order():
    assert normalize(" Deep Caries ") == normalize("deep-caries") == "deep_caries"
    names = {0: "Caries", 1: "Deep Caries", 2: "Periapical Lesion", 3: "Impacted"}
    assert score_findings.class_names(names, {}) == CLASSES
    assert score_findings.class_names(names, MERGE_CARIES) == ["caries", "periapical_lesion", "impacted"]


def test_compare_scores_findings_round_trip(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("compare_scores", ROOT / "scripts" / "compare_scores.py")
    compare_scores = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_scores)
    for run, found in [("a", 1), ("b", 2)]:
        s = FindingScorer(CLASSES)
        for i in range(6):
            s.add([at("caries", 0), at("impacted", 1)], [at("caries", 0), at("impacted", 1)][:found], name=f"{i}.png")
        (tmp_path / run).mkdir()
        with open(tmp_path / run / "score_findings_images.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FINDING_IMAGE_FIELDS)
            writer.writeheader()
            writer.writerows(s.images)
    runs = [compare_scores.read_images(tmp_path / r, "score_findings", FINDING_IMAGE_FIELDS) for r in "ab"]
    assert runs[0][1][0] == {"image": "0.png", "finding": "caries", "truth": 1, "tp": 1, "fp": 0, "fn": 0}
    compare_scores.compare_findings(runs, n=50)
    out = capsys.readouterr().out
    assert "b minus a (paired, same 6 images)" in out
    assert "impacted F1         +100.0 points" in out
