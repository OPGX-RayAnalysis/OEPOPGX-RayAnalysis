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

On Windows, one command does everything below (needs Python 3.13 installed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Or step by step:

```bash
# Windows
py -3.13 -m venv .venv
.venv\Scripts\activate
# macOS / Linux
python3.13 -m venv .venv && source .venv/bin/activate

pip install numpy opencv-python pytest
pip install -e .
pytest            # smoke test: last line says "passed", no "failed"
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
| `data/yolo/teeth` | DENTEX quadrant-enumeration (634 imgs, all teeth labelled) | 32 FDI numbers | 70/15/15 train/val/test, minus 16 mislabelled images (`docs/datasets/dentex.md`) |
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
| `src/opg/schema.py`, `docs/findings_json.md` | Findings JSON (the shared interface) | 1.5 |
| `src/opg/preprocess.py`, `scripts/preprocess_dataset.py` | Crop, CLAHE, resize; crop + CLAHE dataset copy with shifted boxes | 1.8, 2.1 |
| `src/opg/postprocess.py` | Anatomy rules: unique numbers, left/right, missing, wisdom teeth | 2.5 |
| `src/opg/associate.py` | Link findings to teeth | 2.8 |
| `src/opg/pipeline.py` | Image → Findings JSON + annotated image | 2.8 |
| `src/opg/report.py` | Template dentist report + patient summary (the grounded baseline) | 3.3 |
| `src/opg/data/dentex.py` | DENTEX readers, YOLO writer, splits | 1.6, 2.2 |
| `src/opg/data/yolo.py` | Read any YOLO dataset (data.yaml, images, labels) | 1.11, 1.12 |
| `scripts/view_labels.py` | Draw labels (and predictions) with tooth numbers, for spot-checks | 1.12 |
| `src/opg/scoring.py`, `scripts/score_numbering.py` | Numbering accuracy, per-tooth table, missing-tooth F1, mAP50, 95% bootstrap CI | 1.11, 2.13 |
| `scripts/compare_scores.py` | Mean and spread over seeds, paired A-vs-B difference, subsets | 2.13 |
| `docs/PROJECT_CONTEXT.md`, `docs/tracks/`, `scripts/make_context.py` | Shared rules, per-person briefs, AI context bundle | 1.13 |
| `scripts/` | Download, convert, train, analyze | — |
| `tests/` | Pytest suite, run before every merge | — |

The initial scaffold (every file above) was written by Kavya Desai. Later contributions are
credited through the git history. Who works on each task is tracked in the team plan, not here.

Still to come: `src/opg/history/` (task 3.5), `src/opg/agent/` (task 3.8) and `app.py` (Gradio, task 3.9).

## Rules

- Everyone works on their own branch, named after them (`kavya`, `yash`, `shubh`, `krisha`,
  `faizan`). Changes reach `main` only through a pull request that Kavya reviews and merges.
- Before starting a task, run `git pull --no-edit origin main` on your branch. Open one PR per
  task; the PR is how you tell Kavya it's ready. Wait for it to be merged before starting the
  next task, so each PR holds one task.
- `pytest` must pass before a merge. Add a test for anything with logic in it.
- Changing `schema.py` needs sign-off, because every stage reads it.
- Never commit patient images, datasets, weights or anything under `data/`.
