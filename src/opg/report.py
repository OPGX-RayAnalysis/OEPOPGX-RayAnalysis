"""Deterministic template reports from a FindingsReport.

This is the grounded baseline for stage 5. The LLM report generator (task 3.3)
should say no more than this template does. It is also the reference text for the
report evaluation (task 4.2): any tooth or finding the LLM mentions that is not
here counts as a hallucination.
"""

from __future__ import annotations

from . import fdi as F
from .schema import FindingsReport

UNCERTAIN_BELOW = 0.5

DENTIST_TEXT = {
    "caries": "caries",
    "deep_caries": "deep caries (approaching pulp)",
    "periapical_lesion": "periapical radiolucency",
    "impacted": "impacted",
    "crown": "crown",
    "filling": "restoration / filling",
    "root_canal_filling": "root canal treated",
    "implant": "implant",
    "residual_root": "residual root",
    "bridge": "bridge",
    "fracture": "suspected fracture",
    "bone_loss": "alveolar bone loss",
}

PATIENT_TEXT = {
    "caries": "a cavity (tooth decay)",
    "deep_caries": "a deep cavity that may be close to the nerve",
    "periapical_lesion": "a possible infection or inflammation at the tip of the root",
    "impacted": "a tooth that has not come in fully (impacted)",
    "crown": "a crown (cap)",
    "filling": "a filling",
    "root_canal_filling": "a root canal treatment",
    "implant": "a dental implant",
    "residual_root": "a leftover piece of root",
    "bridge": "a bridge",
    "fracture": "a possible crack or fracture",
    "bone_loss": "some loss of the bone that supports the teeth",
}

DISCLAIMER = (
    "AI-generated draft for decision support only. It must be reviewed and "
    "confirmed by a qualified dentist before any clinical use."
)


def _tooth_label(fdi: int, numbering: str) -> str:
    if numbering == "fdi":
        return str(fdi)
    return f"{F.convert(fdi, numbering)} (FDI {fdi})"


def dentist_report(r: FindingsReport, numbering: str = "fdi") -> str:
    lines = [
        "# OPG findings report (AI draft)",
        "",
        f"Image: {r.image.file}",
        f"Numbering: {numbering.upper()}",
        "",
        "## Summary",
        f"- Permanent teeth detected: {r.summary.tooth_count} of 32",
    ]
    missing = [m for m in r.summary.missing_permanent if m not in F.WISDOM_TEETH]
    if missing:
        lines.append(
            "- Not detected (excluding third molars): "
            + ", ".join(_tooth_label(m, numbering) for m in missing)
        )
    wisdom = ", ".join(f"{_tooth_label(int(k), numbering)}: {v}" for k, v in r.summary.wisdom_teeth.items())
    lines.append(f"- Third molars: {wisdom}")

    lines += ["", "## Findings by tooth"]
    any_finding = False
    for quadrant, name in ((1, "Upper right"), (2, "Upper left"), (3, "Lower left"), (4, "Lower right")):
        rows = [
            (t, f)
            for t in r.teeth
            if t.fdi // 10 == quadrant
            for f in t.findings
        ]
        if not rows:
            continue
        any_finding = True
        lines.append(f"### {name}")
        for t, f in rows:
            flag = " (uncertain)" if f.confidence < UNCERTAIN_BELOW else ""
            lines.append(
                f"- Tooth {_tooth_label(t.fdi, numbering)}: {DENTIST_TEXT[f.label]}{flag}, "
                f"confidence {f.confidence:.2f}"
            )
    if not any_finding:
        lines.append("- No findings detected on identified teeth.")
    if r.unassigned_findings:
        lines += ["", "## Findings not linked to a tooth"]
        for f in r.unassigned_findings:
            lines.append(f"- {DENTIST_TEXT[f.label]}, confidence {f.confidence:.2f}, box {[round(v) for v in f.bbox]}")

    lines += ["", "---", DISCLAIMER]
    return "\n".join(lines)


def patient_summary(r: FindingsReport) -> str:
    sentences = [
        f"Your X-ray shows {r.summary.tooth_count} adult teeth.",
    ]
    for t in r.teeth:
        for f in t.findings:
            if f.label == "impacted" and t.fdi in F.WISDOM_TEETH:
                continue  # covered in the wisdom-teeth sentence below
            sentences.append(f"Your {F.describe(t.fdi)} shows {PATIENT_TEXT[f.label]}.")
    impacted = [k for k, v in r.summary.wisdom_teeth.items() if v == "impacted"]
    if impacted:
        sentences.append(
            f"{len(impacted)} wisdom tooth/teeth appear impacted, which means they have not fully come in."
        )
    sentences.append("Your dentist will go through these results with you and confirm them.")
    return " ".join(sentences)
