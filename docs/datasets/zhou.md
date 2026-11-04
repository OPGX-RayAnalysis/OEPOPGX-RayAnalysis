# Dataset card: Zhou et al. dual-labelled set (zhou_yolo_v1)

Panoramic radiographs where every tooth has two labels: its FDI number and its state (healthy,
filling, root canal filling, crown, caries, residual root). We use it for treatment findings
(DENTEX has no fillings, root canals or crowns) and later as a second outside numbering test (task 2.2).
**Non-commercial research only. Team Drive only, never on GitHub, never in an AI tool.**

Rebuild: put the Kaggle download in `data/raw/zhou/`, then
`python scripts/zhou_to_yolo.py --raw data/raw/zhou --out data/yolo/zhou`.
It creates two YOLO datasets from **one** split:
`data/yolo/zhou/teeth/` (32 FDI classes, same order as DENTEX, `opg.data.dentex.TEETH_CLASSES`) and
`data/yolo/zhou/findings/` (5 classes, `opg.data.zhou.FINDING_CLASSES`).
Shared on Drive as `data/zhou_yolo_v1.zip`.

## Source and licence

- Kaggle: https://www.kaggle.com/datasets/zwbzwb12341234/a-dual-labeled-dataset
- Paper: Zhou et al., BMC Oral Health, 2024, doi: 10.1186/s12903-024-04984-2.
- Licence: Kaggle lists no licence; used for non-commercial research.

## How we got it (date downloaded, how many images)

- Downloaded from Kaggle on 27 Oct 2026 by Krisha (about 780 MB).
- `images1/`: **500 PNG images** (`1000.png` ... `4431.png`).
- `labels/`: 2,066 LabelMe JSON files:
  - 1,295 `NNNN.json`: one for each of the 500 images, plus 795 whose image is not in the Kaggle copy.
  - 771 `cwNNNN.json`: for a second image set (`cwNNNN.png`) that is **not** in the Kaggle copy.
    100 share a number with one of our images but describe a different image (different size and
    teeth), so they are not a second opinion on it. The reader ignores every `cw` file.
- So the Kaggle copy is a subset: 500 of the annotated images. Team decision (4 Nov 2026): use the
  Kaggle copy as it is; we are not waiting for the full set from the authors.

## What's in it (images, teeth, tooth states, image size)

- LabelMe JSON, one per image: `{"shapes": [...], "imagePath": "1000.png", "imageWidth": 2440, "imageHeight": 1280}`.
- Per tooth, one shape: `{"label": "36", "group_id": 1, "points": [[x, y], ...], "shape_type": "polygon"}`.
  - **Tooth number:** `label`, already **FDI** (11–48). No conversion needed.
  - **State:** `group_id` (`null` = 0, no anomaly; 1–6 below).
  - **Shape:** `points`, one polygon outline in pixels (one shape in the whole set is a `linestrip`;
    it is not in our 500 images).
- In the 500 images: **13,168 teeth** (11–48), plus 24 shapes labelled `91` (skipped).
- `imageWidth`/`imageHeight` match the real PNG for all 500 images, and every outline lies inside its image.
- Image sizes vary: 205 are 2800 × 1316, 53 are 2940 × 1435, and most of the rest are small,
  about 900–990 × 430–600 px.

| State (`group_id`) | Meaning | Teeth (500 images) |
|---|---|---|
| 0 / null | Tooth without anomalies | 10,583 |
| 1 | Tooth with fillings | 628 |
| 2 | Tooth with RCT | 246 |
| 3 | Tooth with crown | 444 |
| 4 | Tooth with caries | 596 |
| 5 | Residual root | 88 |
| 6 | Tooth with RCT and crown | 583 |

## Conversion (numbering system, outlines to boxes, state -> finding label)

- Numbering: labels are FDI already, used as is. Labels outside 11–48 are skipped.
- Outline → box: smallest x and y, largest x and y of the polygon's points (as in `tufts.py`).
- No flips. Checked on the data: the centres of 3,222 of 3,271 quadrant-1 teeth and 3,219 of 3,351
  quadrant-4 teeth are in the image's left half, and quadrants 2 and 3 are on the right
  (the few exceptions are front teeth near the midline). That is the standard OPG view.
  Checked by eye with `view_labels.py` on 10 random train images per set (4 Nov 2026): 18 top far
  left, 48 bottom far left, finding boxes on the treated teeth.
- State → finding label (`opg.data.zhou.STATE_LABELS`); the findings set has one box per label:

| State | Finding label(s) |
|---|---|
| 0 | none (no finding box) |
| 1 | `filling` |
| 2 | `root_canal_filling` |
| 3 | `crown` |
| 4 | `caries` |
| 5 | `residual_root` |
| 6 | `root_canal_filling` and `crown` (two boxes, same tooth) |

## Split (70/15/15 by image, seed 42) and counts per class

`opg.data.dentex.split_records(records, {"train": 0.70, "val": 0.15, "test": 0.15}, 42)`, done once;
the teeth and findings sets use the same images in each split.

| Split | Images | Tooth boxes | filling | root_canal_filling | crown | caries | residual_root |
|---|---|---|---|---|---|---|---|
| train | 350 | 9,224 | 409 | 559 | 657 | 381 | 54 |
| val | 75 | 1,981 | 133 | 123 | 162 | 91 | 20 |
| test | 75 | 1,963 | 86 | 147 | 208 | 124 | 14 |
| total | 500 | 13,168 | 628 | 829 | 1,027 | 596 | 88 |

(`root_canal_filling` = state 2 + state 6; `crown` = state 3 + state 6.)

The split is by image. The dataset has no patient ids, so two images of one patient could end up in
different splits.

## Skipped or odd labels

- 24 shapes labelled `91` in the 500 images (all state 0): not an FDI tooth number, skipped.
- All 771 `cw*.json` files and the 795 `NNNN.json` files with no image: not used.
- 10 images repeat a tooth number (labelling mistakes; kept as labelled, so the teeth set has two
  boxes with the same class there): 1001.png (42 43 44 45), 1030.png (12), 1081.png (26),
  4300.png (11 13), 4332.png (26), 4340.png (26), 4364.png (17), 4373.png (11), 4384.png (31),
  4413.png (14 17).
- Outside our 500 images (not used): one `linestrip` shape (5180.json) and one shape with
  description "7" (5041.json).
