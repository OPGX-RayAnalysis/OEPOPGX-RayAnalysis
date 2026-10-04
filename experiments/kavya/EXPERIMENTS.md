# Kavya: experiment log

One row per run. Dataset: `dentex_yolo_v1` (434/91/93, see `docs/datasets/dentex.md`).
Scores are on the **test** split from `scripts/score_numbering.py` (conf 0.25, IoU 0.5).

| Date | Run | Model | Settings | Numbering acc. (rules off / on) | Acc. if found | Detection rate | Missing-tooth F1 (off / on) | mAP50 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-04 | kavya-smoke | yolo11n | 5 epochs, 1024 px, batch 4, no flips, RTX 4050 (1.6 GB, ~35 s/epoch) | 15.4% / 17.0% | 45.0% / 49.6% | 34.3% | 20.6% / 21.5% | 32.2% (Ultralytics: 30.0%) | Smoke test of the shared tools (1.11, 1.12 `--pred`), not a baseline. Anatomy rules already add +1.6 points |

## Notes

- **kavya-smoke:** proves train → score → view works end to end on the real data. The low
  numbers are expected after 5 epochs. Weakest teeth were the upper premolars and molars
  (15, 16, 18, 24, 25 at 0%). Ultralytics val mAP50 on the same test split was 30.0%; our
  scorer averages precision differently, so compare mAP50 only between runs scored by our script.
  Found and fixed: Ultralytics 8.4 saved runs to `runs/detect/runs/<name>`; `train_yolo.py`
  now passes an absolute project path so runs land in `runs/<name>/`.
- Weights are not on Drive (test run only). Local: `runs/kavya-smoke/`.
