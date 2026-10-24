# Track: Faizan (checker)

**Phase 2 tasks (25 Oct to Thu 5 Nov 2026, hard deadline 5 Nov):** 2.9 baby-teeth chart test, 3.1 report examples, 2.20 findings spot-check.
**Phase 1 (done):** 1.15 FDI chart test (32 adult teeth), 1.16 label spot-check (30 images).

## Goal
Check the numbering code for baby teeth, collect real report examples (they define what our
reports may claim), and find wrong labels in the new findings data.

## Depends on nothing (start now)
- **2.9 Baby-teeth chart test.** Same method as 1.15, for the 20 primary teeth (FDI 51–85, Universal
  A–T, Palmer letters A–E). New file `tests/test_fdi_chart_primary.py`, chart source at the top.
  Don't change `fdi.py`; report failures in the team WhatsApp group.
- **3.1 Report examples (no coding).** 5–10 real OPG reports from public sources, copied word for word
  with source links, plus a table of the kinds of statements ("claim types") they make. Google Doc
  `results/faizan/report_examples` on Drive. Questions for Dr. Pinak go to Kavya as a list.

## Depends on Kavya's and Krisha's work
- **2.20 Findings spot-check (no coding).** Same method as 1.16: 30 images from Kavya's DENTEX findings
  data (`dentex_findings_v1.zip`, task 2.16) and 20 from Krisha's Zhou data (`zhou_yolo_v1.zip`, task 2.2).
  Log problems in the Drive sheet `results/faizan/findings_spotcheck`.

## Files you may touch
- `tests/test_fdi_chart_primary.py` (create). Nothing else in the code.
- Drive: `results/faizan/` (the two sheets/docs and flagged images).

## Definition of done
- 2.9 PR merged, chart source cited.
- 3.1 doc filled: 5–10 reports with links, claim-type table.
- 2.20 sheet filled (or "no problems found"), flagged images on Drive.
