# Yash Experiments

## Task 1.7 — v1 numbering baseline

| Run | Dataset | Model | Epochs | Actual Epochs | Image Size | Batch | Rules | Numbering Accuracy | Accuracy if Found | Detection Rate | Missing-Tooth F1 | mAP50 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| yash-teeth-v1-3 | DENTEX standard | YOLO11s | 100 | 69 | 1024 | 8 | Off | 94.4% | 95.5% | 98.9% | 82.2% | 94.9% |
| yash-teeth-v1-3 | DENTEX standard | YOLO11s | 100 | 69 | 1024 | 8 | On | 94.3% | 95.5% | 98.7% | 81.1% | 94.9% |

### Notes

- Best model was observed at epoch 44.
- Training stopped at epoch 69 due to early stopping.
- Rules did not improve the baseline; rules-off produced the better numbering accuracy, detection rate, and missing-tooth F1.
- Test split: 93 images, 2,682 teeth.
- Compared with Shubh's plain run (YOLO11n, 50 epochs), which achieved 90.9% numbering accuracy.
- Earlier runs: v1 stopped at epoch 32 and v1-2 stopped at epoch 4; neither was scored on the test set.