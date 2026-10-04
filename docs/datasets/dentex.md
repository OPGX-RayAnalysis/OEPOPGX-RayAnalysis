# Dataset card: DENTEX teeth (dentex_yolo_v1)

The tooth-numbering dataset everyone trains and scores on in Phase 1 (task 1.6).
Get it from the team Drive: `data/dentex_yolo_v1.zip` (1.7 GB). Extract it inside `opg-ai`;
it creates `data/yolo/teeth/`.

## Source

- DENTEX, "quadrant-enumeration" subset: 634 panoramic X-rays where every tooth has a box and
  an FDI number (32 adult teeth, classes '11'…'48').
- Licence: CC BY-NC-SA 4.0. Non-commercial use only; cite both DENTEX papers.
- Rebuild from scratch: `python scripts/download_dentex.py --subsets enumeration`, then
  `python scripts/dentex_to_yolo.py`.

## Split

| Split | Images | Teeth | Use |
|---|---|---|---|
| train | 434 | 12,336 | Training |
| val | 91 | 2,607 | Choosing settings, early stopping |
| test | 93 | 2,682 | Final scores only. Never train or tune on it |

- 70/15/15 random split by image, seed 42. DENTEX has no patient ids, so it can't be split by patient.
- The official 250-image DENTEX test set isn't used here: it only labels diseased teeth, so it
  can't score numbering of every tooth.

## Left out: 16 images with labelling mistakes

These images label the same tooth number twice. A person has each tooth once, so the labels are
wrong: for example, `train_507` labels the lower-right teeth 11–17 instead of 41–47, and
`train_506` has the lower-left quadrant labelled twice. They are removed after the split, so every
other image keeps its split. The list is also in `data/yolo/teeth/excluded_teeth.txt`.

| Split | Image | Repeated numbers |
|---|---|---|
| train | train_150 | 44 45 46 47 |
| train | train_222 | 26 |
| train | train_255 | 26 |
| train | train_258 | 14 15 |
| train | train_504 | 46 |
| train | train_506 | 31 32 33 34 35 36 37 41 |
| train | train_507 | 11 12 13 14 15 16 17 |
| train | train_535 | 23 |
| train | train_584 | 43 |
| train | train_628 | 22 23 24 25 |
| val | train_24 | 36 |
| val | train_247 | 41 |
| val | train_466 | 16 17 |
| val | train_486 | 36 |
| test | train_70 | 44 |
| test | train_415 | 31 |

Other mistakes (a wrong number that isn't repeated) can't be found automatically. Faizan's
spot-check (task 1.16) looks for those; report any you find to Kavya.

## Facts worth knowing

- 6 to 32 teeth per image after cleaning; about 28.5 on average. Missing teeth are common.
- Most images are about 2,900 × 1,316 px; a few are smaller (e.g. 1,976 × 976). `train_yolo.py` trains at 1024 px.
- The patient's right is on the image's left: tooth 18 is top far left.
