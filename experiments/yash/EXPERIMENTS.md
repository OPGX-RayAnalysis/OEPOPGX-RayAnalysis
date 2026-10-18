# Yash Experiments

## Task 1.7 — v1 numbering baseline

| Run | Dataset | Model | Epochs | Actual Epochs | Image Size | Batch | Rules | Numbering Accuracy | Accuracy if Found | Detection Rate | Missing-Tooth F1 | mAP50 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| yash-teeth-v1-3 | DENTEX standard | YOLO11s | 100 | 69 | 1024 | 8 | Off | 94.4% | 95.5% | 98.9% | 82.2% | 94.9% |
| yash-teeth-v1-3 | DENTEX standard | YOLO11s | 100 | 69 | 1024 | 8 | On | 94.3% | 95.5% | 98.7% | 81.1% | 94.9% |
| yash-teeth-v1-seed1 | DENTEX standard | YOLO11s | 100 | 57 | 1024 | 8 | Off | 93.9% | 94.6% | 99.2% | 79.4% | 95.2% |
| yash-teeth-v1-seed2 | DENTEX standard | YOLO11s | 100 | 57 | 1024 | 8 | Off | 94.3% | 95.1% | 99.1% | 82.1% | 94.8% |

### Notes

- Best model was observed at epoch 44.
- Training stopped at epoch 69 due to early stopping.
- Rules did not improve the baseline; rules-off produced the better numbering accuracy, detection rate, and missing-tooth F1.
- Test split: 93 images, 2,682 teeth.
- Compared with Shubh's plain run (YOLO11n, 50 epochs), which achieved 90.9% numbering accuracy.
- Earlier runs: v1 stopped at epoch 32 and v1-2 stopped at epoch 4; neither was scored on the test set.

### Seed check (3 seeds)

- Numbering accuracy over seeds 42, 1, 2: 94.2% (SD 0.3 points, range 93.9% to 94.4%).
- v1 (seed 42): 94.4%, 95% CI 92.5% to 96.0% (bootstrap over the 93 test images).
- Full output: seed_check.txt
- All three runs used the same 93-image test split (2,682 teeth), with anatomy rules turned off for the seed comparison.

Paired comparisons against v1 (seed 42) showed no clear difference for either seed 1 or seed 2 across numbering accuracy, accuracy if found, detection rate, or missing-tooth F1.
