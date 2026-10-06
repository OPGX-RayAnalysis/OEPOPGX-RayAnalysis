# Shubh: does cleaning the image help numbering? (task 1.8)

Both runs: yolo11n, 50 epochs, 1024 px, batch 8, seed 42, mosaic 0.5, horizontal flip off (`fliplr=0.0`).
Scored on the TEST split with `score_numbering.py` (93 images, 2682 teeth, rules off).

| Run | Images | Numbering acc. | Acc. if found | Detection rate | Missing-tooth F1 | mAP50 |
|---|---|---|---|---|---|---|
| shubh-plain-v1 | plain | 90.9% | 91.9% | 98.9% | 79.6% | 95.1% |
| shubh-clahe-v1 | crop + CLAHE | 90.9% | 92.1% | 98.7% | 77.0% | 94.9% |

Conclusion: Crop + CLAHE did not improve numbering accuracy (both 90.9%, the same count of correctly numbered teeth). The other metrics did not improve either. Keep the simpler plain-image pipeline for numbering.
