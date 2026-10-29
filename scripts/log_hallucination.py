"""Add a row to the hallucination log (task 2.18) from a scored run's claims file.

score_numbering.py writes score_claims.csv (score_rules_claims.csv with --rules) and
score_findings.py writes score_findings_claims.csv. This reads one of them and adds one row
to experiments/hallucination_log.csv: claims made, unsupported, rate with a 95% interval
(images resampled), a count per error type, omissions, and Balel 2026's four groups. What a
claim is and what each error means: src/opg/claims.py.

Every new model version gets a row per test set (one PR = its rows). A row with the same
run, test set, layer and setting replaces the old one.

Usage:
    python scripts/log_hallucination.py runs/scores/v1-test/score_claims.csv --run yash-teeth-v1 --test-set "DENTEX teeth test" --setting "rules off"
    python scripts/log_hallucination.py runs/scores/v1-test/score_rules_claims.csv --run yash-teeth-v1 --test-set "DENTEX teeth test" --setting "rules on"
    python scripts/log_hallucination.py runs/scores/findings-winner-test/score_findings_claims.csv --run shubh-findings-plain-v1 --test-set "DENTEX findings test"
    python scripts/log_hallucination.py <claims.csv> --run x --test-set y --dry-run     # print the row only
    python scripts/log_hallucination.py runs/scores/v1-test/score_rules_claims.csv --run yash-teeth-v1 --test-set "DENTEX teeth test" --setting "rules on" --vs runs/scores/v1-test/score_claims.csv

--vs OTHER_claims.csv also prints this run's rate minus the other's on the same images, with its
95% interval ("no clear difference" if it contains 0), and puts it in the note.

Tufts rows: Krisha runs this on her own scores and sends the printed row (numbers only).
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

from opg.claims import ERRORS, bootstrap_rate_ci, paired_rate_difference, summarize_claims

LOG_FIELDS = [
    "date", "layer", "component", "run", "test_set", "setting", "images", "claims", "unsupported",
    "rate", "rate_ci_low", "rate_ci_high", *ERRORS, "omissions", "omission_rate",
    "balel_true", "balel_false", "balel_missing", "balel_hallucination", "note",
]
KEY = ["layer", "run", "test_set", "setting"]


def read_claims(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "supported": int(r["supported"])} for r in csv.DictReader(f)]


def make_row(rows: list[dict], n: int = 1000) -> dict:
    s = summarize_claims(rows)
    ci = bootstrap_rate_ci(rows, n=n) or [None, None]

    def r4(v):
        return "" if v is None else round(v, 4)

    return {
        "component": "findings" if any(r["claim"] in ("finding", "omitted") for r in rows) else "numbering",
        "images": len({r["image"] for r in rows}),
        "claims": s["claims"],
        "unsupported": s["unsupported"],
        "rate": r4(s["rate"]),
        "rate_ci_low": r4(ci[0]),
        "rate_ci_high": r4(ci[1]),
        **s["errors"],
        "omissions": s["omissions"],
        "omission_rate": r4(s["omission_rate"]),
        **{f"balel_{k}": v for k, v in s["balel"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("claims", type=Path, help="a *_claims.csv written by score_numbering.py or score_findings.py")
    parser.add_argument("--run", required=True, help="model version, e.g. yash-teeth-v1")
    parser.add_argument("--test-set", required=True, help='e.g. "DENTEX teeth test"')
    parser.add_argument("--setting", default="", help='what else changed, e.g. "rules on", "CLAHE", "merge caries"')
    parser.add_argument("--layer", default="detector", choices=["detector", "text-vs-json", "text-vs-labels"])
    parser.add_argument("--note", default="")
    parser.add_argument("--log", type=Path, default=Path("experiments/hallucination_log.csv"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="print the row, don't write it")
    parser.add_argument("--vs", type=Path, default=None, help="another claims file to compare with (paired)")
    args = parser.parse_args()

    row = {
        "date": datetime.date.today().isoformat(), "layer": args.layer, "run": args.run,
        "test_set": args.test_set, "setting": args.setting, "note": args.note,
        **make_row(read_claims(args.claims), args.bootstrap),
    }
    vs_line = ""
    if args.vs:
        d = paired_rate_difference(read_claims(args.vs), read_claims(args.claims), n=args.bootstrap)
        lo, hi = d["ci"]
        verdict = "no clear difference" if lo <= 0 <= hi else "different"
        vs_line = (f"{100 * d['diff']:+.1f} points vs {args.vs.parent.name}/{args.vs.name} "
                   f"(paired, {d['images']} images, 95% CI {100 * lo:+.1f} to {100 * hi:+.1f}, {verdict})")
        row["note"] = "; ".join(x for x in (row["note"], vs_line) if x)
    row = {k: row[k] for k in LOG_FIELDS}

    def pct(v):
        return "n/a" if v == "" else f"{100 * v:.1f}%"

    print(f"{row['run']} on {row['test_set']} ({row['setting'] or 'default'}), {row['component']}, {row['layer']} layer:")
    print(f"  {row['unsupported']} of {row['claims']} claims unsupported = {pct(row['rate'])}"
          f"   95% CI {pct(row['rate_ci_low'])} to {pct(row['rate_ci_high'])}")
    print("  " + ", ".join(f"{e} {row[e]}" for e in ERRORS if row[e]))
    if row["component"] == "findings":
        print(f"  omissions {row['omissions']} ({pct(row['omission_rate'])} of real findings)")
    print(f"  Balel: true {row['balel_true']}, false {row['balel_false']}, "
          f"missing {row['balel_missing']}, hallucination {row['balel_hallucination']}")
    if vs_line:
        print(f"  rate {vs_line}")
    if args.dry_run:
        return

    old = []
    if args.log.exists():
        with open(args.log, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if [r[k] for k in KEY] != [str(row[k]) for k in KEY]]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(old + [row])
    print(f"Logged in {args.log} ({len(old) + 1} rows)")


if __name__ == "__main__":
    main()
