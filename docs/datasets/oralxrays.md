# Dataset card: OralXrays-9 (oralxrays_yolo_v1)

Outside test for the findings detector (task 2.6) and training data for treatments and
implants (task 2.7). Built in task 2.17.
Get it from the team Drive: `data/oralxrays_yolo_v1.zip`. Extract it inside `opg-ai`;
it creates `data/yolo/oralxrays/`.

## Source and terms

- OralXrays-9, Chen et al., CVPR 2025. The authors' own release: 12,688 panoramic X-rays
  (all resized to 1991 × 1127 JPG), 9 classes, COCO boxes. Download link and unzip password
  on their GitHub page (github.com/Binz-Chan/CVPR2025_OralXrays-9).
- **Written permission from the authors (24 Oct 2026):** non-commercial research including
  training, evaluation and publication; cite their CVPR 2025 paper; use the official split
  when comparing with their results; an outside test stays independent of training and
  validation; **no redistribution** of the images or annotations (team Drive only, never on
  GitHub, never uploaded to an AI tool).
- Rebuild: download their four files into `data/raw/oralxrays/`, unzip with their password,
  then `python scripts/oralxrays_to_yolo.py`.

## Classes

| Their class | Ours | Kept? |
|---|---|---|
| Decay | caries | yes (no deep vs shallow: score with `--merge-caries`) |
| Apical Periodontitis | periapical_lesion | yes |
| Dental Filling | filling | yes |
| Root Canal Filling | root_canal_filling | yes |
| Porcelain Crown | crown | yes |
| Ceramic Bridge | bridge | yes |
| Implant | implant | yes |
| Wisdom Tooth | — | no: boxes every third molar, not only impacted ones (27,325 boxes) |
| Missing Tooth | — | no: a numbering fact, not a finding (4,093 boxes) |

31 boxes with category id 0 (not one of their classes) are also left out.

## Split

| Split | Images | caries | periapical_lesion | filling | root_canal_filling | crown | bridge | implant | No box |
|---|---|---|---|---|---|---|---|---|---|
| train | 9,000 | 8,700 | 4,106 | 8,462 | 5,472 | 3,408 | 3,873 | 1,913 | 1,483 |
| val | 1,000 | 919 | 408 | 965 | 622 | 392 | 448 | 289 | 171 |
| test | 2,688 | 1,609 | 1,070 | 2,212 | 2,340 | 1,652 | 1,497 | 2,327 | 429 |

- test = the authors' `val2017` split, unchanged (their benchmark split).
- train / val = their `train2017` split, 90/10 by image, seed 42.
- No image is shared with DENTEX (all 1,589 DENTEX images checked by fingerprint).
- 8 test images repeat a train image inside the authors' own split. They are kept (official
  split) and listed in `test_seen_in_train.txt`: add `--skip data/yolo/oralxrays/test_seen_in_train.txt`
  when scoring a model trained on OralXrays-9.
- "No box" counts images with none of the 7 kept classes (many only had wisdom or missing teeth).

## Boxes are a different size from DENTEX: score with both rules

OralXrays-9 boxes the lesion or the restoration itself; DENTEX (and so our model) boxes the
whole tooth. Median box, as a share of image width × height:

| | OralXrays-9 | DENTEX |
|---|---|---|
| caries | 0.035 × 0.071 | 0.058 × 0.215 |
| periapical lesion | 0.038 × 0.059 | 0.067 × 0.224 |

A correct whole-tooth prediction overlaps a cavity box by an IoU of about 0.2, so the normal
rule counts it as a miss. Score the outside test twice and report both:

```
python scripts/score_findings.py --weights runs/<winner>/weights/best.pt --data data/yolo/oralxrays/data.yaml --split test --merge-caries --out runs/scores/findings-oralxrays
python scripts/score_findings.py --weights runs/<winner>/weights/best.pt --data data/yolo/oralxrays/data.yaml --split test --merge-caries --match cover --out runs/scores/findings-oralxrays
```

`--match cover`: a prediction matches when its box covers at least half of the real box
("did the model flag the tooth that has it?"), still one real finding per prediction. The IoU
number shows the protocol gap; the cover number is the meaningful one. Only caries and
periapical_lesion are scored for a DENTEX model (the other classes are treatments it never learnt).

## For task 2.7 (treatments, implants)

- Treatment boxes are also tight (filling 0.028 × 0.040 of the image), so don't mix these
  with a dataset that boxes whole teeth in one training set without deciding on one convention.
- Implants: 1,913 train, 289 val, 2,327 test.
