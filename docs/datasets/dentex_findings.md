# Dataset card: DENTEX findings (dentex_findings_v1)

The diagnosis dataset for the findings detector (tasks 2.6, 2.16).
Get it from the team Drive: `data/dentex_findings_v1.zip` (about 3.5 GB). Extract it inside `opg-ai`;
it creates `data/yolo/findings/`.

## Source

- DENTEX, "quadrant-enumeration-disease" subset: 705 panoramic X-rays. Only teeth with a
  diagnosis have a box; healthy teeth have none. 4 classes: `caries`, `deep_caries`,
  `periapical_lesion`, `impacted`.
- DENTEX released test set (`test_data.zip`, 250 images): used only as the extra split
  `test_official`, see below.
- Licence: CC BY-NC-SA 4.0. Non-commercial use only; cite both DENTEX papers.
- Rebuild from scratch: `python scripts/download_dentex.py --subsets disease test`, then
  `python scripts/dentex_to_yolo.py`.

## Split

| Split | Images | Findings | caries | deep_caries | periapical_lesion | impacted | Use |
|---|---|---|---|---|---|---|---|
| train | 493 | 2,555 | 1,611 | 403 | 117 | 424 | Training |
| val | 106 | 459 | 283 | 69 | 16 | 91 | Choosing settings (plain vs CLAHE, epochs) |
| test | 106 | 515 | 295 | 106 | 25 | 89 | The headline score. Never train or tune on it |
| test_official | 249 | 1,039 | 745 | 0 | 73 | 221 | Extra score, with the caveats below |

- 70/15/15 random split by image, seed 42 (same rule as the teeth set). DENTEX has no patient ids.
- 27 images have no finding at all (21 train, 2 val, 4 test). They stay in: a model that
  invents findings on them is penalised.
- Periapical lesion is rare (25 in test), so its F1 interval will be wide.

## Why the official 250 images are not the main test set

The DENTEX challenge scored entries against hidden labels in the same 4-class format as
training. Those were never released. The released `test_data.zip` labels use a different
scheme: one whole-tooth polygon per tooth, with a treatment-need category in Turkish:

| Label | Meaning | Count | Kept as |
|---|---|---|---|
| çürük | caries | 747 | caries |
| gömülü | impacted | 221 | impacted |
| lezyon | lesion | 75 | periapical_lesion |
| küretaj | curettage (periodontal) | 265 | dropped |
| kanal | root canal | 161 | dropped |
| saglam | healthy | 91 | dropped |
| çekim | extraction | 29 | dropped |
| kırık | fracture | 11 | dropped |

So `test_official`:
- has no `deep_caries`, and its caries labels don't separate deep from shallow: **always score
  it with `--merge-caries`**;
- drops teeth labelled root canal, curettage, extraction or fracture. Some of those have deep
  caries or a lesion, so a correct prediction there counts as an invented finding. Expect lower
  precision here than on `test`, and report it as "official images, different label protocol";
- leaves out `test_181` (the same X-ray as `train_332`; listed in `excluded_findings.txt`).

## Scoring

```
python scripts/score_findings.py --weights runs/<run>/weights/best.pt --data data/yolo/findings/data.yaml --split val
python scripts/score_findings.py --weights runs/<run>/weights/best.pt --data data/yolo/findings/data.yaml --split test
python scripts/score_findings.py --weights runs/<run>/weights/best.pt --data data/yolo/findings/data.yaml --split test_official --merge-caries --out runs/<run>/official
python scripts/compare_scores.py --findings runs/<run-a> runs/<run-b>
```

Headline: mean F1 over the 4 classes with its 95% CI. Also report per-class F1 and the
pooled numbers.

## Facts worth knowing

- Boxes are whole teeth (median side about 210 px), not the lesion itself: the model finds
  "a tooth with caries", and the tooth number comes from the numbering model.
- Same image sizes as the teeth set (mostly about 2,900 × 1,316 px). Train at 1024 px; flips stay off.
