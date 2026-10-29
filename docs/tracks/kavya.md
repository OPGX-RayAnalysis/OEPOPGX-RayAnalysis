# Track: Kavya (lead)

**Phase 2 tasks (25 Oct to Thu 5 Nov 2026, hard deadline 5 Nov):** 2.15, 2.16, 2.17, 2.18, plus reviews.

## Goal
Build the shared pieces the team waits on first, then start the hallucination log.

## Depends on nothing (start now, in this order)
- **2.15 Scorer additions.** `score_numbering.py`: `whole_scan_accuracy` (share of images where every
  real tooth is found with the right number and no extra tooth is claimed) in `score.json` and the printout;
  `score_preds.csv` (one row per predicted tooth: image, fdi, confidence, matched, correct) for calibration.
- **2.16 DENTEX findings data + scorer.** `dentex_findings_v1.zip` on Drive `data/` (disease subset split
  70/15/15 train/val/test, 4 classes, plus `test_official`: the released official test images, 3 classes), `docs/datasets/dentex_findings.md` (with class
  counts), `scripts/score_findings.py` (per-class precision, recall, F1 at IoU 0.5 with 95% CI;
  `score_findings.json`, `score_findings_images.csv`; `--merge-caries` counts deep caries as caries; classes matched by name, so any findings data.yaml works),
  and `compare_scores.py --findings` for paired comparisons.
- **2.17 OralXrays-9.** Authors' release from their GitHub. Converter mapping its 9 classes to ours, official
  split kept, no DENTEX images, card, `oralxrays_yolo_v1.zip` on Drive. Diagnoses = outside test for 2.6;
  implants (if present) go to 2.7.

## Depends on 2.15 (Balel 2026 read 27 Oct: wording settled)
- **2.18 Claim definition + hallucination log.** One row per model version per test set; first rows for
  existing runs (v1 rules off / all / molars-only, plain vs CLAHE, v1 on Tufts via Krisha).
  - Claim = one tooth + one statement (present, missing, or one finding). Error types: invented tooth,
    invented finding, wrong FDI number, wrong side, negation flip, severity inflation, omission.
  - Two layers: vs Findings JSON (faithfulness, the text's fault) and vs labels (correctness). Until the
    report generator exists only the detector layer has rows.
  - Columns: date, component, run, test set, claims, unsupported, rate per error type, 95% CI
    (patient-level bootstrap), plus Balel's four counts (true, false, missing, hallucination).

## Later (January 2027, after the report generator 3.3; not Phase 2)
- Report test on the DENTEX test images (then Tufts via Krisha, numbers only), five conditions:
  template-only; Findings JSON -> LLM; JSON -> LLM + checker; same with the true labels as JSON
  (splits detector errors from LLM errors); image-only VLM (OralGPT).
- LLMs, all local in 4-bit, matching Balel 2026: Qwen3 8B, DeepSeek R1 8B, Mistral 7B. Fixed recorded
  prompt, temperature 0.6, 3 runs each (Balel's settings). Strip DeepSeek's `<think>` text before scoring.
- Inject known errors into reports to measure the checker's catch and false-reject rates.

## Files you may touch
Anything. A change to `schema.py` bumps `SCHEMA_VERSION` and gets announced to the team.

## Reviewing a PR
```
git fetch origin
git checkout <name>
git pull
pytest
git checkout kavya
```
Check: only that task's files, no data or weights, flips still off, tests for any logic.
