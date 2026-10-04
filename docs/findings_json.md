# Findings JSON v1.0

The one file every stage after detection reads (report, patient history, agent, demo app).
Defined in `src/opg/schema.py`; this page is the readable version. Plan task 1.5.

## Rules

- Tooth numbers are always **FDI** (11–48, primary 51–85). Convert only for display (`opg.fdi`).
- Boxes are `[x1, y1, x2, y2]` in pixels of the **original** image, even if the model ran on a
  cropped or resized copy.
- Confidences are 0–1. Invalid tooth numbers, boxes or confidences raise an error on creation.
- Version `"<major>.<minor>"`: a new optional field bumps the minor version and old files still
  load; a rename, removal or change of meaning bumps the major version. Any change needs
  Kavya's sign-off.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `image.file`, `width`, `height` | str, int, int | The original image |
| `image.patient_id` | str or null | Anonymised id linking one patient's X-rays (history, 3.5) |
| `image.study_date` | str or null | ISO date the X-ray was taken |
| `teeth[]` | list | One entry per detected tooth, sorted by FDI |
| `teeth[].fdi` | int | Final FDI number, after the anatomy rules |
| `teeth[].renumbered_from` | int or null | The model's original number if the rules changed it |
| `teeth[].findings[]` | list | Findings linked to this tooth |
| `finding.label` | str | One of the labels below |
| `finding.tooth_fdi` | int or null | The tooth it belongs to; null = not linked |
| `finding.source` | str | Which model or rule produced it |
| `summary.tooth_count` | int | Permanent teeth detected |
| `summary.missing_permanent` | list[int] | Permanent teeth not detected (wisdom teeth included) |
| `summary.wisdom_teeth` | dict | `"18"` → `present`, `missing` or `impacted` |
| `summary.dentition` | str | `permanent` (mixed/primary is stretch goal 3.11) |
| `summary.finding_counts` | dict | Label → count, linked and unlinked |
| `unassigned_findings[]` | list | Findings not linked to any tooth (e.g. a lesion in the bone) |
| `models` | dict | Which weights produced this, e.g. `{"teeth": "yash-teeth-v1"}` |
| `schema_version` | str | `"1.0"` |

## Finding labels (final for v1)

| Group | Labels | Trained from |
|---|---|---|
| Diagnoses | `caries`, `deep_caries`, `periapical_lesion`, `impacted` | DENTEX (Phase 2, 2.6) |
| Diagnoses | `fracture`, `residual_root`, `bone_loss` | Extra datasets (2.2) |
| Treatments | `crown`, `filling`, `root_canal_filling`, `implant`, `bridge` | Extra datasets (2.2) |

Converters map other datasets' class names onto these (e.g. an "infection" class →
`periapical_lesion`, "broken root" → `residual_root`). A class with no match is dropped,
not added, unless Kavya agrees to a new label. Every label has dentist and patient wording in
`src/opg/report.py`.

## Example

```json
{
  "image": {"file": "opg_0001.png", "width": 2900, "height": 1300,
            "patient_id": "P-0042", "study_date": "2026-10-04"},
  "teeth": [
    {"fdi": 11, "bbox": [1380.0, 430.0, 1450.0, 700.0], "confidence": 0.97,
     "findings": [], "renumbered_from": null},
    {"fdi": 36, "bbox": [1650.0, 720.0, 1790.0, 1010.0], "confidence": 0.94,
     "findings": [
       {"label": "deep_caries", "bbox": [1690.0, 740.0, 1760.0, 820.0], "confidence": 0.81,
        "tooth_fdi": 36, "source": "findings_model"}
     ],
     "renumbered_from": 37}
  ],
  "summary": {"tooth_count": 2, "missing_permanent": [12, 13, "...", 48],
              "wisdom_teeth": {"18": "missing", "28": "missing", "38": "missing", "48": "missing"},
              "dentition": "permanent",
              "finding_counts": {"deep_caries": 1, "periapical_lesion": 1}},
  "unassigned_findings": [
    {"label": "periapical_lesion", "bbox": [2100.0, 1050.0, 2160.0, 1110.0], "confidence": 0.44,
     "tooth_fdi": null, "source": "findings_model"}
  ],
  "models": {"teeth": "yash-teeth-v1", "findings": "shubh-findings-v1"},
  "schema_version": "1.0"
}
```

(`missing_permanent` is shortened here; a real file lists every number.)
