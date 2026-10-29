# Track: Shubh (findings detector)

**Phase 2 tasks (25 Oct to Thu 5 Nov 2026, hard deadline 5 Nov):** 2.6 diagnosis detector, 2.7 treatment detector.
**Phase 1 (done):** 1.8 crop + CLAHE gave no numbering gain (90.9% both), so numbering trains on plain images.

## Goal
Bring findings to the level numbering reached in Phase 1: a trained model, scored on held-out test
images with a 95% CI, plain vs CLAHE compared, a seed check, and a score on outside data.

## Depends on Kavya's work
- **2.6 Diagnosis detector** (caries, deep caries, periapical lesion, impacted tooth) needs Kavya's
  `dentex_findings_v1.zip` and `scripts/score_findings.py` (task 2.16). Train plain and CLAHE, pick on
  **val**, score the winner on **test** (106 held-out DENTEX images) and on `test_official` with `--merge-caries`
  (the official DENTEX images; different label scheme, an extra number only), then 2 more seeds of the winner.
- **Outside test** of the 2.6 winner on OralXrays-9 needs Kavya's `oralxrays_yolo_v1.zip` (task 2.17).
  Score with `--merge-caries`, once with the default IoU rule and once with `--match cover`, and report
  both (it boxes the cavity, our model boxes the tooth: `docs/datasets/oralxrays.md`).

## Depends on Krisha's work
- **2.7 Treatment detector** (filling, root canal filling, crown, plus caries and residual root) needs
  Krisha's `zhou_yolo_v1.zip` (task 2.2). OralXrays-9 has implants (1,913 train boxes), but it boxes the restoration itself, not the tooth:
  pick one box convention before mixing it with Zhou (decide with Kavya).

## Files you may touch
- `experiments/shubh/FINDINGS.md` (create)
- `scripts/preprocess_dataset.py` (yours) if a dataset needs a small change.

## Data rules
- DENTEX test images: scoring only, never training.
- Zhou and OralXrays-9: non-commercial research only, team Drive only, never on GitHub.

## Definition of done
- 2.6: plain vs CLAHE on val, winner on test with CI, 2 extra seeds, OralXrays-9 outside score, weights on Drive, PR merged.
- 2.7: trained, scored on the Zhou test split, weights on Drive, PR merged.

## Starter commands
```
python scripts/train_yolo.py --data data/yolo/findings/data.yaml --name shubh-findings-plain-v1 --batch 8
python scripts/preprocess_dataset.py --data data/yolo/findings/data.yaml --out data/yolo/findings_clahe
python scripts/score_findings.py --weights runs/<run>/weights/best.pt --data data/yolo/findings/data.yaml --split val
```
Flips stay off in every run (`train_yolo.py` already does this).
