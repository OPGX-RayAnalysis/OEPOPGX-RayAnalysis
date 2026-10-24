# Track: Krisha (outside test data and calibration)

**Phase 2 tasks (25 Oct to Thu 5 Nov 2026, hard deadline 5 Nov):** 2.19 Tufts scores, 2.2 Zhou converter, 2.14 calibration.
**Phase 1 (done):** 1.14 Tufts converter (1,000 images, 25,259 adult teeth, 82 with baby teeth).

## Goal
Show how the numbering model behaves on data it never saw (Tufts), onboard the Zhou dataset for
treatments, and check whether the model's confidence can be trusted (calibration).

## Depends on nothing (start now)
- **2.19 (part 1) Tufts score of v1.** Re-run `tufts_to_yolo.py` (the 10 Oct version also writes
  `mixed_dentition.txt`), download v1's `best.pt` from Drive, score all 1,000 and the 918 without baby
  teeth, rules off and on. Share the score files on Drive `results/krisha/tufts/`. Numbers only, never images.
- **2.2 Zhou dual-labelled dataset.** Download from Kaggle, write `src/opg/data/zhou.py`,
  `scripts/zhou_to_yolo.py` (two outputs: `data/yolo/zhou/teeth/` with the 32 FDI classes, and
  `data/yolo/zhou/findings/` with tooth states as finding labels), `tests/test_zhou.py`, `docs/datasets/zhou.md`.
  Split 70/15/15 by image, seed 42. Zip both to Drive `data/zhou_yolo_v1.zip`.

## Depends on Kavya's work
- **2.14 Calibration** needs `score_preds.csv` from Kavya's task 2.15 (one row per predicted tooth:
  confidence, and whether it was right). Write `src/opg/calibration.py` (ECE with 10 bins, reliability
  table), `scripts/calibration.py` (prints ECE, saves a reliability diagram PNG), `tests/test_calibration.py`.
  Run it on v1 for the DENTEX test split and for Tufts.
- **2.19 (part 1 again)** after 2.15: re-score v1 on Tufts so the files include whole-scan accuracy.

## Depends on Yash's work
- **2.19 (part 2)** Score Yash's tuned model (2.4) on Tufts the same way, and compare it with v1.

## Files you may touch
- `src/opg/data/zhou.py`, `scripts/zhou_to_yolo.py`, `tests/test_zhou.py`, `docs/datasets/zhou.md`
- `src/opg/calibration.py`, `scripts/calibration.py`, `tests/test_calibration.py`
- `experiments/krisha/TUFTS.md`, `experiments/krisha/CALIBRATION.md`

## Data rules
- Tufts: never share images or files with anyone, never in any AI tool; share numbers only.
- Zhou: non-commercial research only, team Drive only, never on GitHub; no real images in tests.

## Definition of done
- Tufts scores for v1 and the tuned model on Drive and in `TUFTS.md`.
- Zhou converter merged, zip on Drive, 20 images checked in the viewer.
- Calibration merged, ECE + diagram for DENTEX test and Tufts in `CALIBRATION.md`.
