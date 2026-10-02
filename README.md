# opg-ai

Agentic AI for OPG (orthopantomogram) X-ray analysis. Given a panoramic dental X-ray, it:
numbers every tooth (FDI), finds missing and wisdom teeth, detects findings (caries,
lesions, impactions, restorations), writes a dentist report and a patient summary, and
answers questions through an agent.

Team plan and task tracker: see the shared doc "OPG X-Ray AI — Project Pipeline & Team Plan".
Task numbers in comments (e.g. "task 2.5") refer to that tracker.

## Pipeline

```
OPG ─► 1 preprocess ─┬─► 2 teeth + FDI numbers ─┐
                     └─► 3 findings detector ───┴─► 4 Findings JSON ─► 5 report
                                                         │              ─► 6 patient history
                                                         └────────────► 7 agent Q&A ─► 8 demo app
```

Everything after stage 4 reads the Findings JSON (`src/opg/schema.py`), never the raw image.

## Setup (task 1.1)

Use Python 3.11–3.13. PyTorch doesn't support 3.14 yet.

```bash
# Windows
py -3.13 -m venv .venv
.venv\Scripts\activate
# macOS / Linux
python3.13 -m venv .venv && source .venv/bin/activate

pip install numpy opencv-python pytest
pip install -e .
pytest            # smoke test, should print "39 passed"
```

For training, install PyTorch with CUDA from https://pytorch.org/get-started/locally/, then run
`pip install ultralytics`. No GPU? Use Google Colab or Kaggle (free T4): clone the repo,
`pip install -e . ultralytics`, and run the same scripts.

## Data (task 1.6)

```bash
python scripts/download_dentex.py          # ~4.7 GB: enumeration + disease + official test set
python scripts/dentex_to_yolo.py --list-test-labels   # check the Turkish test-label map
python scripts/dentex_to_yolo.py           # -> data/yolo/teeth and data/yolo/findings
```

| Dataset made | Source | Classes | Splits |
|---|---|---|---|
| `data/yolo/teeth` | DENTEX quadrant-enumeration (634 imgs, all teeth labelled) | 32 FDI numbers | 85/15 train/val |
| `data/yolo/findings` | DENTEX quadrant-enumeration-disease (705 imgs, abnormal teeth only) | caries, deep caries, periapical lesion, impacted | 90/10 train/val + official 250-image test |

DENTEX is CC BY-NC-SA 4.0: non-commercial use only, and cite both DENTEX papers.
`data/` is git-ignored. Share it through the team Drive folder.

## Train and run

```bash
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name teeth-v1        # task 1.7 / 2.4
python scripts/train_yolo.py --data data/yolo/findings/data.yaml --name findings-v1  # task 2.6
python scripts/analyze.py some_opg.png --teeth runs/teeth-v1/weights/best.pt \
    --findings runs/findings-v1/weights/best.pt --numbering universal
```

Never train with horizontal flips. Mirroring an OPG swaps left and right, so tooth 18
would really be 28. `train_yolo.py` turns flips off.

## Code map

| Path | What | Plan task |
|---|---|---|
| `src/opg/fdi.py` | FDI ↔ Universal ↔ Palmer, tooth names | 2.9 |
| `src/opg/schema.py` | Findings JSON (the shared interface) | 1.5 |
| `src/opg/preprocess.py` | Crop, CLAHE, resize | 1.8, 2.1 |
| `src/opg/postprocess.py` | Anatomy rules: unique numbers, left/right, missing, wisdom teeth | 2.5 |
| `src/opg/associate.py` | Link findings to teeth | 2.8 |
| `src/opg/pipeline.py` | Image → Findings JSON + annotated image | 2.8 |
| `src/opg/report.py` | Template dentist report + patient summary (the grounded baseline) | 3.3 |
| `src/opg/data/dentex.py` | DENTEX readers, YOLO writer, splits | 1.6, 2.2 |
| `scripts/` | Download, convert, train, analyze | — |
| `tests/` | Pytest suite, run before every merge | — |

The initial scaffold (every file above) was written by Kavya Desai. Later contributions are
credited through the git history. Who works on each task is tracked in the team plan, not here.

Still to come: `src/opg/history/` (task 3.5), `src/opg/agent/` (task 3.8) and `app.py` (Gradio, task 3.9).

## Rules

- One branch per person and task (`yash/1.7-tooth-numbering`). Push small pieces often and open a
  PR when one works; the PR is how you tell Kavya it's ready. Kavya reviews before merging to `main`.
- `pytest` must pass before a merge. Add a test for anything with logic in it.
- Changing `schema.py` needs sign-off, because every stage reads it.
- Never commit patient images, datasets, weights or anything under `data/`.
