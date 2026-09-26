"""Run the full pipeline on one OPG and write the Findings JSON, annotated image and reports.

Usage:
    python scripts/analyze.py path/to/opg.jpg --teeth runs/teeth-v1/weights/best.pt \
        --findings runs/findings-v1/weights/best.pt --numbering universal --out outputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opg.pipeline import analyze, draw
from opg.report import dentist_report, patient_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--teeth", required=True)
    parser.add_argument("--findings", default=None)
    parser.add_argument("--numbering", default="fdi", choices=["fdi", "universal", "palmer"])
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    report = analyze(args.image, args.teeth, args.findings)
    report.save(args.out / f"{stem}.findings.json")
    draw(args.image, report, args.out / f"{stem}.annotated.jpg")
    (args.out / f"{stem}.report.md").write_text(dentist_report(report, args.numbering), encoding="utf-8")
    (args.out / f"{stem}.patient.txt").write_text(patient_summary(report), encoding="utf-8")
    print(dentist_report(report, args.numbering))


if __name__ == "__main__":
    main()
