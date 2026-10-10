"""Compare runs scored by score_numbering.py: intervals, seeds, A vs B, subsets (task 2.13).

Each argument is a run folder holding score.json + score_images.csv (or the score.json
itself). Everything is recomputed from the per-image file, so runs scored before the
bootstrap was added must be scored again first.

Usage:
    python scripts/compare_scores.py runs/yash-teeth-v1 runs/yash-teeth-v1-seed1 runs/yash-teeth-v1-seed2
    python scripts/compare_scores.py runs/shubh-plain-v1 runs/shubh-clahe-v1
    python scripts/compare_scores.py runs/tufts-v1 --skip data/yolo/tufts_test/mixed_dentition.txt

Prints, for each run, the four numbers with a 95% bootstrap interval. With 2+ runs it adds
the mean, SD and range across them (for seeds) and, for each run after the first, the
paired difference from the first with its interval: if the interval contains 0, the two
runs are not shown to differ on these images.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from opg.scoring import CI_METRICS, IMAGE_FIELDS, bootstrap_ci, paired_difference_ci, summarize

LABELS = {
    "numbering_accuracy": "numbering accuracy",
    "accuracy_if_found": "accuracy if found",
    "detection_rate": "detection rate",
    "missing_tooth_f1": "missing-tooth F1",
}


def read_images(arg: Path, rules: bool) -> tuple[str, list[dict]]:
    stem = "score_rules" if rules else "score"
    folder = arg.parent if arg.suffix == ".json" else arg
    if arg.suffix == ".json":
        stem = arg.stem
    path = folder / f"{stem}_images.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Score the run again with the current score_numbering.py.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k: (v if k == "image" else int(v)) for k, v in r.items() if k in IMAGE_FIELDS} for r in csv.DictReader(f)]
    return folder.name, rows


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders or score.json files")
    parser.add_argument("--rules", action="store_true", help="read score_rules_images.csv instead")
    parser.add_argument("--skip", type=Path, default=None, help="text file of image names to leave out")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    runs = [read_images(r, args.rules) for r in args.runs]
    if args.skip:
        skip = {Path(line.strip()).stem.lower() for line in args.skip.read_text(encoding="utf-8").splitlines() if line.strip()}
        runs = [(name, [r for r in rows if Path(r["image"]).stem.lower() not in skip]) for name, rows in runs]
        print(f"Leaving out the {len(skip)} images listed in {args.skip}")

    summaries = []
    for name, rows in runs:
        s, ci = summarize(rows), bootstrap_ci(rows, n=args.bootstrap)
        summaries.append(s)
        print(f"\n{name}: {s['images']} images, {s['teeth']} teeth")
        for m in CI_METRICS:
            bounds = ci[m]
            interval = f"95% CI {pct(bounds[0])} to {pct(bounds[1])}" if bounds else ""
            print(f"  {LABELS[m]:<20}{pct(s[m]):>7}   {interval}")

    if len(runs) < 2:
        return

    print(f"\nAcross the {len(runs)} runs (mean, SD, range):")
    for m in CI_METRICS:
        values = [s[m] for s in summaries if s[m] is not None]
        if len(values) < 2:
            continue
        sd = statistics.stdev(values)
        print(f"  {LABELS[m]:<20}{pct(statistics.mean(values)):>7}   SD {100 * sd:.1f} points   range {pct(min(values))} to {pct(max(values))}")

    first_name, first_rows = runs[0]
    for name, rows in runs[1:]:
        d = paired_difference_ci(first_rows, rows, n=args.bootstrap)
        print(f"\n{name} minus {first_name} (paired, same {d['images']} images):")
        for m in CI_METRICS:
            diff, bounds = d[m]["diff"], d[m]["ci"]
            if diff is None or bounds is None:
                continue
            verdict = "no clear difference" if bounds[0] <= 0 <= bounds[1] else "different"
            print(
                f"  {LABELS[m]:<20}{100 * diff:+6.1f} points   95% CI {100 * bounds[0]:+.1f} to {100 * bounds[1]:+.1f}   {verdict}"
            )


if __name__ == "__main__":
    main()
