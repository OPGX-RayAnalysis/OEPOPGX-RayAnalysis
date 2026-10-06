# Track: Shubh (does cleaning the image help numbering?)

**Task:** 1.8 (done, merged 6 Oct 2026: CLAHE gives no gain). After 17 Oct: 2.1 (resize, masking burned-in text), then the findings detector (2.6).

## Goal
Make the X-rays clearer (crop, CLAHE), then prove whether that helps the model number teeth.

## Files you may touch
- `scripts/preprocess_dataset.py` (create)
- `tests/test_preprocess.py` (create)
- `experiments/shubh/RESULTS.md` (create)
- `src/opg/preprocess.py` (crop, CLAHE and resize already exist; extend, don't rewrite)

## Inputs
- `dentex_yolo_v1.zip` from Drive `data/` → `data/yolo/teeth/`
- `src/opg/preprocess.py`, `src/opg/data/yolo.py`, `scripts/train_yolo.py`, `scripts/score_numbering.py`

## Outputs
- `python scripts/preprocess_dataset.py --data data/yolo/teeth/data.yaml --out data/yolo/teeth_clahe`
  makes a cleaned copy with **label boxes shifted to match the crop**, and a `data.yaml` with
  the same class order.
- `--preview N` writes N before/after images for the slides.
- Drive: `weights/shubh-plain-v1/`, `weights/shubh-clahe-v1/`, `results/shubh/before_after/*.png`.
  Don't upload the cleaned dataset; the script recreates it.
- One line for Yash: e.g. "CLAHE: +X% numbering accuracy, use it" or "no gain".

## Definition of done
- `tests/test_preprocess.py` proves the boxes still line up after cropping (fake image + box).
- Two runs with identical settings (yolo11n or s, ~50 epochs), plain vs cleaned, both scored
  with `score_numbering.py` on the test split.
- `RESULTS.md`: the two-row score table and the conclusion. `pytest` passes.

## Starter commands
```
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name shubh-plain-v1 --model yolo11n.pt --epochs 50 --batch 8
python scripts/train_yolo.py --data data/yolo/teeth_clahe/data.yaml --name shubh-clahe-v1 --model yolo11n.pt --epochs 50 --batch 8
python scripts/score_numbering.py --weights runs/shubh-plain-v1/weights/best.pt --data data/yolo/teeth/data.yaml --split test
python scripts/score_numbering.py --weights runs/shubh-clahe-v1/weights/best.pt --data data/yolo/teeth_clahe/data.yaml --split test
python scripts/view_labels.py --data data/yolo/teeth_clahe/data.yaml --n 5 --random --out check_clahe/
```
Check the cleaned labels with `view_labels.py` before training: the boxes must sit on the teeth.
