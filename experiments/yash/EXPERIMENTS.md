# Yash Experiments

## Task 1.7 — v1 numbering baseline

| Run                 | Dataset         | Model   | Epochs | Actual Epochs | Image Size | Batch | Rules | Numbering Accuracy | Accuracy if Found | Detection Rate | Missing-Tooth F1 | mAP50 |
| ------------------- | --------------- | ------- | -----: | ------------: | ---------: | ----: | ----- | -----------------: | ----------------: | -------------: | ---------------: | ----: |
| yash-teeth-v1-3     | DENTEX standard | YOLO11s |    100 |            69 |       1024 |     8 | Off   |              94.4% |             95.5% |          98.9% |            82.2% | 94.9% |
| yash-teeth-v1-3     | DENTEX standard | YOLO11s |    100 |            69 |       1024 |     8 | On    |              94.3% |             95.5% |          98.7% |            81.1% | 94.9% |
| yash-teeth-v1-seed1 | DENTEX standard | YOLO11s |    100 |            57 |       1024 |     8 | Off   |              93.9% |             94.6% |          99.2% |            79.4% | 95.2% |
| yash-teeth-v1-seed2 | DENTEX standard | YOLO11s |    100 |            57 |       1024 |     8 | Off   |              94.3% |             95.1% |          99.1% |            82.1% | 94.8% |

### Notes

* Best model was observed at epoch 44.
* Training stopped at epoch 69 due to early stopping.
* Rules did not improve the baseline; rules-off produced the better numbering accuracy, detection rate, and missing-tooth F1.
* Test split: 93 images, 2,682 teeth.
* Compared with Shubh's plain run (YOLO11n, 50 epochs), which achieved 90.9% numbering accuracy.
* Earlier runs: v1 stopped at epoch 32 and v1-2 stopped at epoch 4; neither was scored on the test set.

### Seed check (3 seeds)

* Numbering accuracy over seeds 42, 1, 2: 94.2% (SD 0.3 points, range 93.9% to 94.4%).
* v1 (seed 42): 94.4%, 95% CI 92.5% to 96.0% (bootstrap over the 93 test images).
* Full output: seed_check.txt
* All three runs used the same 93-image test split (2,682 teeth), with anatomy rules turned off for the seed comparison.

Paired comparisons against v1 (seed 42) showed no clear difference for either seed 1 or seed 2 across numbering accuracy, accuracy if found, detection rate, or missing-tooth F1.

---

## Task 2.4 — Numbering model tuning

### Objective

Evaluate whether increasing model size or input image resolution improves tooth detection and FDI numbering accuracy compared with the original v1 baseline.

All experiments used the standard DENTEX dataset and the same validation split.

### Experiment configurations

| Run                        | Dataset         | Model   | Epochs | Actual Epochs | Best Epoch | Image Size | Batch | Workers | Rules |
| -------------------------- | --------------- | ------- | -----: | ------------: | ---------: | ---------: | ----: | ------: | ----- |
| yash-teeth-v1-3 (baseline) | DENTEX standard | YOLO11s |    100 |            69 |         44 |       1024 |     8 |       8 | Off   |
| yash-teeth-v2-m            | DENTEX standard | YOLO11m |    100 |            36 |         11 |       1024 |     4 |       8 | Off   |
| yash-teeth-v2-s1280-2      | DENTEX standard | YOLO11s |    100 |            44 |         19 |       1280 |     4 |       2 | Off   |

Note: The first YOLO11s 1280px attempt encountered an OpenCV memory allocation error during epoch 3. The successful rerun used 2 DataLoader workers and was saved as `yash-teeth-v2-s1280-2`.

### Validation results

All three models were evaluated on the same validation split: 91 images and 2,607 teeth.

| Run       | Model           | Numbering Accuracy | Accuracy if Found | Detection Rate | Missing-Tooth F1 | Whole-Scan Accuracy | Extra Teeth Claimed | mAP50 |
| --------- | --------------- | -----------------: | ----------------: | -------------: | ---------------: | ------------------: | ------------------: | ----: |
| v1-val    | YOLO11s, 1024px |              92.9% |             94.6% |          98.2% |            83.2% |               35.2% |                 150 | 93.3% |
| m-val     | YOLO11m, 1024px |              91.0% |             92.7% |          98.2% |            80.2% |               20.9% |                 239 | 92.9% |
| s1280-val | YOLO11s, 1280px |              93.4% |             95.0% |          98.3% |            81.8% |               25.3% |                 189 | 93.8% |

### Paired statistical comparisons against v1

Comparisons used the same 91 validation images.

**Run A: YOLO11m, 1024px minus v1**

| Metric              |   Difference | 95% CI        | Result              |
| ------------------- | -----------: | ------------- | ------------------- |
| Numbering accuracy  |  -1.9 points | -3.2 to -0.5  | Different           |
| Accuracy if found   |  -2.0 points | -3.3 to -0.6  | Different           |
| Detection rate      |  +0.0 points | -0.3 to +0.4  | No clear difference |
| Missing-tooth F1    |  -3.0 points | -7.6 to +1.4  | No clear difference |
| Whole-scan accuracy | -14.3 points | -24.2 to -3.3 | Different           |

**Conclusion:** YOLO11m performed significantly worse than v1 in numbering accuracy and whole-scan accuracy. It was not selected.

**Run B: YOLO11s, 1280px minus v1**

| Metric              |  Difference | 95% CI        | Result              |
| ------------------- | ----------: | ------------- | ------------------- |
| Numbering accuracy  | +0.5 points | -0.4 to +1.5  | No clear difference |
| Accuracy if found   | +0.4 points | -0.5 to +1.4  | No clear difference |
| Detection rate      | +0.2 points | -0.1 to +0.5  | No clear difference |
| Missing-tooth F1    | -1.3 points | -5.7 to +2.7  | No clear difference |
| Whole-scan accuracy | -9.9 points | -19.8 to -1.1 | Different           |

**Conclusion:** Although YOLO11s at 1280px achieved the highest raw validation numbering accuracy (93.4%), the improvement over v1 was not statistically clear. Whole-scan accuracy was significantly worse.

### Final decision

* Neither experiment demonstrated a statistically clear improvement in numbering accuracy over the original v1 baseline.
* Increasing model size to YOLO11m reduced numbering performance.
* Increasing input resolution to 1280px produced a small numerical improvement, but no statistically clear gain.
* The original YOLO11s model at 1024px remains the selected baseline.
* Original v1 DENTEX test numbering accuracy remains 94.4% (2,533 correctly detected-and-numbered teeth out of 2,682).

**Test-set policy:** The new experiments were evaluated on the validation set only. Their test-set performance has not been evaluated. The original v1 test result remains unchanged.

### Output locations

* Original baseline: `runs/yash-teeth-v1-3/`
* Run A: `runs/yash-teeth-v2-m/`
* Run B: `runs/yash-teeth-v2-s1280-2/`
* Baseline validation scores: `runs/scores/v1-val/`
* Run A validation scores: `runs/scores/m-val/`
* Run B validation scores: `runs/scores/s1280-val/`

---

## Task 2.5 — Molar-only anatomy rules

### Objective

Evaluate whether applying anatomy-based numbering corrections only to molars (FDI positions 6, 7 and 8) can preserve the improvements from anatomy rules while avoiding incorrect renumbering of premolars.

The original v1 model (`yash-teeth-v1-3`, YOLO11s, 1024px) was used for all experiments.

Three configurations were compared:

1. Rules OFF — original model predictions.
2. All rules ON — complete anatomy postprocessing.
3. Molar-only — anatomy corrections are retained only for molar positions; other teeth preserve the model's original numbering.

### Implementation

* Added `molars_only: bool = False` to `apply_anatomy_rules()` in `src/opg/postprocess.py`.
* Added `--molars-only` to `scripts/score_numbering.py`.
* The option requires `--rules`.
* Default behaviour remains unchanged.
* Molar-only mode retains rule-adjusted numbering for molars (positions 6, 7 and 8) and restores original model numbering for other teeth.
* Dropped non-molar detections are restored, while dropped molar detections remain dropped.
* Added tests covering duplicate premolar predictions, duplicate molar predictions and dropped detections.

### Validation results

All configurations were evaluated on the same DENTEX validation split: 91 images and 2,607 real teeth.

| Metric              | Rules OFF | All rules ON | Molar-only |
| ------------------- | --------: | -----------: | ---------: |
| Numbering accuracy  |     92.9% |    **93.7%** |      93.5% |
| Accuracy if found   |     94.6% |    **95.5%** |      95.3% |
| Detection rate      |     98.2% |        98.1% |      98.2% |
| Missing-tooth F1    |     83.2% |        82.5% |  **83.9%** |
| Whole-scan accuracy |     35.2% |    **60.4%** |      46.2% |
| Extra teeth claimed |       150 |       **55** |        105 |
| mAP50 (raw model)   |     93.3% |        93.3% |      93.3% |

### Paired statistical comparisons against rules OFF

All comparisons used the same 91 validation images.

**All rules ON minus rules OFF**

| Metric              |   Difference | 95% CI         | Result              |
| ------------------- | -----------: | -------------- | ------------------- |
| Numbering accuracy  |  +0.8 points | +0.0 to +1.6   | Different           |
| Accuracy if found   |  +0.9 points | +0.1 to +1.7   | Different           |
| Detection rate      |  -0.0 points | -0.1 to +0.0   | No clear difference |
| Missing-tooth F1    |  -0.7 points | -2.7 to +1.6   | No clear difference |
| Whole-scan accuracy | +25.3 points | +16.5 to +34.1 | Different           |

**Molar-only minus rules OFF**

| Metric              |   Difference | 95% CI        | Result              |
| ------------------- | -----------: | ------------- | ------------------- |
| Numbering accuracy  |  +0.6 points | +0.1 to +1.1  | Different           |
| Accuracy if found   |  +0.6 points | +0.1 to +1.1  | Different           |
| Detection rate      |  +0.0 points | +0.0 to +0.0  | No clear difference |
| Missing-tooth F1    |  +0.7 points | -1.1 to +2.7  | No clear difference |
| Whole-scan accuracy | +11.0 points | +4.4 to +17.6 | Different           |

### Validation hallucination comparison

Hallucination rates were compared using `log_hallucination.py --vs`, with paired comparisons across the same 91 images.

| Configuration | Unsupported claims | Total claims | Hallucination rate | Difference vs OFF |
| ------------- | -----------------: | -----------: | -----------------: | ----------------: |
| Rules OFF     |                  — |            — |           Baseline |                 — |
| All rules ON  |                217 |        2,912 |               7.4% |       +0.5 points |
| Molar-only    |                203 |        2,912 |               7.0% |       +0.0 points |

* All rules ON: paired 95% CI for difference = -0.3 to +1.4 points.
* Molar-only: paired 95% CI for difference = -0.5 to +0.5 points.
* Neither configuration showed a statistically clear hallucination-rate difference from rules OFF on validation.

### Configuration selection

All rules ON was selected based on validation performance.

Reasons:

* Highest validation numbering accuracy (93.7%).
* Highest whole-scan accuracy (60.4%).
* Fewest extra teeth claimed (55).
* No statistically clear increase in hallucination rate on validation.

Molar-only achieved the highest missing-tooth F1 (83.9%), but its overall validation performance was weaker than all rules ON.

### Final test evaluation

The selected configuration (v1 with all anatomy rules ON) was evaluated on the untouched DENTEX test split: 93 images and 2,682 real teeth.

| Metric              |  Rules OFF | All rules ON |
| ------------------- | ---------: | -----------: |
| Numbering accuracy  |  **94.4%** |        94.3% |
| 95% CI              | 92.5–96.0% |   92.2–96.0% |
| Accuracy if found   |      95.5% |        95.5% |
| Detection rate      |      98.9% |        98.7% |
| Missing-tooth F1    |  **82.2%** |        81.1% |
| Whole-scan accuracy |      26.9% |    **43.0%** |
| Extra teeth claimed |        153 |       **50** |
| mAP50 (raw model)   |      94.9% |        94.9% |

### Test hallucination comparison

The selected configuration produced:

* 210 unsupported claims out of 2,976 claims.
* Hallucination rate: 7.1%.
* 95% CI: 5.1% to 9.2%.
* Paired difference versus rules OFF: +1.0 percentage point.
* Paired 95% CI: +0.2 to +2.0 points.
* Result: statistically different, indicating a clear increase in hallucination rate.

### Final conclusion

All rules ON was selected using validation results, primarily due to its substantially higher whole-scan accuracy and fewer extra tooth claims.

However, test evaluation showed trade-offs:

* Numbering accuracy remained comparable to the original rules-OFF baseline (94.3% vs 94.4%).
* Whole-scan accuracy improved from 26.9% to 43.0%.
* Extra teeth claims decreased from 153 to 50.
* Missing-tooth F1 decreased from 82.2% to 81.1%.
* Hallucination rate increased significantly by 1.0 percentage point.

Therefore, anatomy rules improve whole-scan consistency but introduce a measurable hallucination trade-off. They should not be described as improving every evaluation metric.

### Output loc
