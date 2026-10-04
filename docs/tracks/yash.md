# Track: Yash (tooth-numbering model)

**Tasks:** 1.7 (v1 baseline), then 2.4 (tuning). After that: 2.5 anatomy-rules evaluation, 3.7 robustness.

## Goal
Teach the AI to find every tooth and give it the correct FDI number (11–48). Your work is
mostly experiments, not new code.

## Files you may touch
- `experiments/yash/EXPERIMENTS.md` (create it)
- `scripts/train_yolo.py` only if needed. It is shared: flips must stay off (`fliplr=0.0`).

## Inputs
- `dentex_yolo_v1.zip` from Drive `data/`, extracted inside `opg-ai` → `data/yolo/teeth/`
- `scripts/train_yolo.py`, `scripts/score_numbering.py`, `scripts/view_labels.py`
- Shubh's one-line answer on whether CLAHE preprocessing helps (you don't need it to start)

## Outputs
- Drive `weights/yash-teeth-v1/`: `best.pt`, `results.csv`, `args.yaml`, `score.json`
- Drive `results/yash/`: 4 result images for the slides (from `view_labels.py --pred`)
- PR: `experiments/yash/EXPERIMENTS.md` (+ any script change). Weights never go in git.

## Definition of done (1.7)
- v1 trained and scored on the **test** split, with and without `--rules`.
- `EXPERIMENTS.md` has one row per run: settings, numbering accuracy, one line on what changed.
- `pytest` passes.

## Starter commands
```
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name yash-teeth-v1 --batch 8
python scripts/score_numbering.py --weights runs/yash-teeth-v1/weights/best.pt --data data/yolo/teeth/data.yaml --split test
python scripts/score_numbering.py --weights runs/yash-teeth-v1/weights/best.pt --data data/yolo/teeth/data.yaml --split test --rules
python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split test --n 4 --random --pred runs/yash-teeth-v1/weights/best.pt --out results_yash/
```
v1 = yolo11s, 1024 px, 100 epochs, batch 8 (RTX 4060, 8 GB). Tuning ideas for 2.4: yolo11m,
image size 1280, more epochs. Choose settings on the **val** split; use test only to report.
