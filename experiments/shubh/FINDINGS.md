# Shubh: findings detector (tasks 2.6, 2.7)

## 2.6 Diagnoses (DENTEX: impacted, caries, deep_caries, periapical_lesion)

All runs: yolo11s, 100 epochs, 1024 px, batch 8, flips off, mosaic 0.5, workers 2.
Chosen on VAL, winner reported on TEST (106 held-out DENTEX images). Scores from `score_findings.py`.

The findings set was rebuilt with `download_dentex.py --subsets disease test` and `dentex_to_yolo.py`.
Counts match `docs/datasets/dentex_findings.md`: train 493, val 106, test 106, test_official 249.

| Run | Images | Split | Mean F1 (95% CI) | impacted F1 | caries F1 | deep_caries F1 | periapical F1 |
|---|---|---|---|---|---|---|---|
| shubh-findings-plain-v1 | plain | val | 48.9% (40.9% to 57.0%) | 84.3% | 38.3% | 45.8% | 27.3% |
| shubh-findings-clahe-v1 | crop + CLAHE | val | 53.7% (46.8% to 60.3%) | 83.1% | 35.0% | 46.8% | 50.0% |
| shubh-findings-plain-v1 | plain | test | 43.7% (38.7% to 48.8%) | 85.1% | 39.2% | 38.9% | 11.4% |

Plain vs CLAHE (`compare_scores.py --findings`): mean F1 +4.8 points, 95% CI -3.0 to +11.9, no clear difference. Plain wins.

Seeds 42, 1, 2 of the winner on test: mean F1 46.1% (SD 4.6 points). Seed 42 was 43.7%, seed 1 was 51.4%, seed 2 was 43.3%.

Official DENTEX images (test_official, --merge-caries, different label protocol): mean F1 45.4% (95% CI 41.9% to 49.2%). Per class: caries 37.1%, periapical_lesion 12.9%, impacted 86.2%. No deep_caries in this split.

Outside test, OralXrays-9 test split (--merge-caries; caries + periapical_lesion):
- pending: `oralxrays_yolo_v1.zip` is not in the public Drive `data/` folder (that folder lists only `dentex_findings_v1.zip` and `dentex_yolo_v1.zip`) and is not on this machine.

## 2.7 Treatments (Zhou dual-labelled set, 500 images)

| Run | Split | Mean F1 (95% CI) | filling | root_canal_filling | crown | caries | residual_root |
|---|---|---|---|---|---|---|---|
| shubh-treatments-v1 | val | pending | | | | | |
| shubh-treatments-v1 | test | pending | | | | | |

Pending: `zhou_yolo_v1.zip` is not in the public Drive `data/` folder and is not on this machine. No combined implant YAML was provided, so `shubh-treatments-v2` was not trained.
