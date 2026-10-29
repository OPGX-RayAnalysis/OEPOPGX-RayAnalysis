import importlib.util
from pathlib import Path

from opg import fdi as F
from opg.claims import (
    bootstrap_rate_ci, findings_claims, mirror, numbering_claims, paired_rate_difference, summarize_claims,
)
from opg.scoring import Scorer, Tooth
from opg.scoring_findings import Finding, FindingScorer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("log_hallucination", ROOT / "scripts" / "log_hallucination.py")
log_hallucination = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_hallucination)


def box(i: int) -> list[float]:
    return [20 * i, 0, 20 * i + 10, 10]


def errors(rows: list[dict]) -> dict:
    return {r["about"]: r["error"] for r in rows if not r["supported"]}


def test_mirror():
    assert (mirror(36), mirror(14), mirror(41), mirror(85)) == (46, 24, 31, 75)


def test_every_permanent_tooth_gets_one_claim():
    truth = [Tooth(11, box(0)), Tooth(21, box(1))]
    rows = numbering_claims(truth, [Tooth(11, box(0)), Tooth(21, box(1))])
    assert len(rows) == 32 and all(r["supported"] for r in rows)
    assert {r["about"] for r in rows if r["claim"] == "present"} == {11, 21}


def test_numbering_error_types():
    truth = [Tooth(36, box(0)), Tooth(14, box(1)), Tooth(47, box(2)), Tooth(48, box(3))]
    pred = [
        Tooth(46, box(0)),  # 36 called its mirror 46
        Tooth(15, box(1)),  # 14 called 15
        Tooth(47, box(2)),  # right
        Tooth(11, box(9)),  # a tooth where there is none
    ]  # 48 not found at all
    e = errors(numbering_claims(truth, pred))
    assert e == {
        46: "wrong_side", 15: "wrong_number", 11: "invented_tooth",
        36: "negation_flip", 14: "negation_flip", 48: "negation_flip",
    }


def test_duplicate_number_is_one_claim_and_any_right_box_supports_it():
    truth = [Tooth(36, box(0)), Tooth(37, box(1))]
    pred = [Tooth(36, box(1), 0.9), Tooth(36, box(0), 0.5)]
    rows = [r for r in numbering_claims(truth, pred) if r["about"] == 36]
    assert rows == [{"image": "", "claim": "present", "about": 36, "supported": 1, "error": ""}]


def test_findings_error_types_and_omissions():
    truth = [Finding("caries", box(0)), Finding("impacted", box(1)), Finding("periapical_lesion", box(2))]
    pred = [
        Finding("deep_caries", box(0)),  # caries called deep: severity inflation
        Finding("impacted", box(1), 0.9),
        Finding("impacted", box(1), 0.8),  # same finding on the same tooth again: not a new claim
        Finding("caries", box(2)),  # lesion called caries
        Finding("caries", box(5)),  # nothing there
    ]
    rows = findings_claims(truth, pred)
    made = [r for r in rows if r["claim"] == "finding"]
    assert len(made) == 4
    assert sorted(r["error"] for r in made) == ["", "invented_finding", "severity_inflation", "wrong_finding"]
    assert sorted(r["about"] for r in rows if r["claim"] == "omitted") == ["caries", "periapical_lesion"]


def test_summary_rate_and_balel_groups():
    truth = [Tooth(36, box(0)), Tooth(14, box(1))]
    rows = numbering_claims(truth, [Tooth(46, box(0)), Tooth(11, box(5))])
    s = summarize_claims(rows)
    assert s["claims"] == 32 and s["unsupported"] == 4  # 46 wrong side, 11 invented, 36 and 14 denied
    assert s["balel"] == {"true": 28, "false": 1, "hallucination": 1, "missing": 2}
    assert s["omissions"] == 0 and s["omission_rate"] is None


def test_scorers_collect_claims():
    s = Scorer()
    s.add([Tooth(11, box(0))], [Tooth(11, box(0))], name="a.png")
    assert len(s.claims) == 32 and s.claims[0]["image"] == "a.png"
    fs = FindingScorer(["caries"], merge={"deep_caries": "caries"})
    fs.add([Finding("caries", box(0))], [Finding("deep_caries", box(0))], name="b.png")
    assert fs.claims == [{"image": "b.png", "claim": "finding", "about": "caries", "supported": 1, "error": ""}]


def test_bootstrap_rate_ci_brackets_rate():
    rows = []
    for i in range(20):
        pred = [Tooth(t, box(k)) for k, t in enumerate(F.ALL_PERMANENT)]
        if i % 2:
            pred[0] = Tooth(12, box(0))  # 11 called 12 on half the scans
        rows += numbering_claims([Tooth(t, box(k)) for k, t in enumerate(F.ALL_PERMANENT)], pred, image=f"{i}")
    rate = summarize_claims(rows)["rate"]
    lo, hi = bootstrap_rate_ci(rows, n=200)
    assert lo <= rate <= hi and lo < hi


def test_log_row_replaces_same_key(tmp_path, monkeypatch):
    claims = tmp_path / "score_claims.csv"
    import csv
    with open(claims, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "claim", "about", "supported", "error"])
        writer.writeheader()
        writer.writerows(numbering_claims([Tooth(11, box(0))], [Tooth(21, box(0))], image="a.png"))
    log = tmp_path / "log.csv"
    for setting in ["rules off", "rules off", "rules on"]:
        monkeypatch.setattr("sys.argv", ["x", str(claims), "--run", "v1", "--test-set", "t", "--setting", setting,
                                         "--log", str(log), "--bootstrap", "10"])
        log_hallucination.main()
    with open(log, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["setting"] for r in rows] == ["rules off", "rules on"]
    assert rows[0]["component"] == "numbering" and rows[0]["wrong_side"] == "1" and rows[0]["negation_flip"] == "1"


def test_paired_rate_difference():
    truth = [Tooth(t, box(k)) for k, t in enumerate(F.ALL_PERMANENT)]
    a, b = [], []
    for i in range(30):
        a += numbering_claims(truth, list(truth), image=f"{i}")
        b += numbering_claims(truth, [Tooth(12, box(0))] + truth[1:], image=f"{i}")  # 11 called 12 everywhere
    d = paired_rate_difference(a, b, n=100)
    assert d["images"] == 30 and d["diff"] == 1 / 32 and d["ci"][0] > 0  # only "11 missing" is wrong: real 12 is still boxed 12
    assert paired_rate_difference(a, a, n=50)["ci"] == [0.0, 0.0]
