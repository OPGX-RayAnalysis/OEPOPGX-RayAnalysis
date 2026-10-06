# Project context: OPG X-Ray AI

Version 1.1 · 6 Oct 2026 · Owner: Kavya (team lead)

This file is the shared ground for everyone on the team and for the AI tools we use. Upload it
(as part of your `context_<name>.md` bundle) before asking an AI for code, so its answers fit
this project.

## 1. What we are building

A college OEP project: "Agentic AI Solution for Intelligent OPG X-ray Analysis".
An OPG (orthopantomogram) is a panoramic dental X-ray showing all teeth in one image.
Given an OPG, the system:

1. numbers every tooth using the FDI system (11–48),
2. finds missing and wisdom teeth,
3. detects findings (caries, deep caries, periapical lesions, impactions, restorations),
4. writes a dentist report and a plain-language patient summary,
5. answers questions about the X-ray through an agent.

Pipeline:

```
OPG ─► 1 preprocess ─┬─► 2 teeth + FDI numbers ─┐
                     └─► 3 findings detector ───┴─► 4 Findings JSON ─► 5 report
                                                         │              ─► 6 patient history
                                                         └────────────► 7 agent Q&A ─► 8 demo app
```

Everything after stage 4 reads the Findings JSON (`src/opg/schema.py`), never the raw image.

**Phase 1 goal (now):** a model that numbers teeth accurately, measured on a held-out test set.
The first presentation is on 17 Oct 2026 and needs a first baseline, not a finished model.

**Results so far:** cropping plus CLAHE does not help numbering (task 1.8: 90.9% numbering
accuracy with and without, yolo11n, 50 epochs, test split). **Train numbering models on the
plain images** (`data/yolo/teeth/`). Details: `experiments/shubh/RESULTS.md`.

## 2. Team

| Person | Phase 1 job |
|---|---|
| Kavya | Lead. Data zip, scoring script, label viewer, these files; reviews and merges every PR |
| Yash | Trains the tooth-numbering model (1.7, then 2.4) |
| Shubh | Image preprocessing, and tests whether it helps numbering (1.8) |
| Krisha | Tufts external test set (1.14) |
| Faizan | Tooth-number converter tests (1.15) and label spot-check (1.16) |

Task numbers (like 1.11) refer to the team plan and tracker.

## 3. Data

- **DENTEX** (public, CC BY-NC-SA 4.0: non-commercial, cite the DENTEX papers). We use the
  634 fully labelled "quadrant-enumeration" images, split 70/15/15 into train/val/test
  (seed 42), then leave out 16 images whose labels repeat a tooth number: 434/91/93 images.
  The **test split is never trained on**; it is how we score numbering. Details: `docs/datasets/dentex.md`.
- The team gets the data as `dentex_yolo_v1.zip` from the team Drive (`data/` folder).
  Extract it inside the `opg-ai` folder; it creates `data/yolo/teeth/`.
- YOLO layout: `data/yolo/teeth/data.yaml`, `images/<split>/*.png`, `labels/<split>/*.txt`.
  Each label line is `class cx cy w h` (normalised 0–1). Class index → FDI number via the
  `names` list in `data.yaml` (index 0 = tooth 11, index 31 = tooth 48).
- `data.yaml` has no `path:` line on purpose, so it works wherever the folder is.
- **Tufts Dental Database** (Krisha) is a second, external test set. It uses Universal numbers
  (1–32), which must be converted to FDI with `opg.fdi.from_universal`.

## 4. Tooth numbering (read this before touching any numbering code)

- FDI: two digits, quadrant then position. Quadrant 1 = upper right, 2 = upper left,
  3 = lower left, 4 = lower right (the **patient's** sides). Position 1 = central incisor
  … 8 = wisdom tooth. Primary (baby) teeth use quadrants 5–8.
- On an OPG the **patient's right is on the image's left**. So tooth 18 is at the top far left,
  28 at the top far right, 48 bottom left, 38 bottom right.
- **Never train with horizontal flips.** Mirroring swaps left and right, so 18 would become 28
  and every label would be silently wrong. `scripts/train_yolo.py` sets `fliplr=0.0`.
- Conversions live in `src/opg/fdi.py`: `to_universal`, `from_universal`, `to_palmer`,
  `from_palmer`, `convert(fdi, system)`, `describe(fdi)`. Use them; don't write new ones.

## 5. Folder layout

```
opg-ai/
  src/opg/            the package (import as `opg`)
    fdi.py            FDI ↔ Universal ↔ Palmer, tooth names
    schema.py         Findings JSON (the shared interface)
    preprocess.py     crop, CLAHE, resize, shift YOLO boxes after a crop
    postprocess.py    anatomy rules: unique numbers, left/right, missing, wisdom teeth
    scoring.py        numbering metrics (used by score_numbering.py)
    associate.py      link findings to teeth
    pipeline.py       image → Findings JSON + annotated image
    report.py         dentist report + patient summary
    data/dentex.py    DENTEX readers, YOLO writer, splits
    data/yolo.py      read any YOLO dataset (data.yaml, images, labels)
  scripts/            command-line tools (download, convert, train, score, view, analyze)
  tests/              pytest suite
  docs/               this file, per-person tracks, dataset cards
  experiments/<name>/ experiment logs and result tables (markdown only)
  data/               datasets, git-ignored, comes from Drive
  runs/               training output, git-ignored
```

Team Drive (`OPGX-RayAnalysis`): `data/` zips, `weights/<person>-<model>-v<N>/`
(best.pt, results.csv, args.yaml, score.json), `results/<person>/` (slide images, sheets),
`deck/` (presentation).

## 6. Shared tools

```
python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split train --n 30 --random --out spotcheck/
python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split test --pred best.pt --out mistakes/
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name <person>-teeth-v1
python scripts/score_numbering.py --weights runs/<name>/weights/best.pt --data data/yolo/teeth/data.yaml --split test [--rules]
python scripts/preprocess_dataset.py --data data/yolo/teeth/data.yaml --out data/yolo/teeth_clahe [--preview 5]
```

- `view_labels.py` draws boxes with tooth numbers, coloured by quadrant (`--system universal|palmer`
  for other numbering systems, `--pred` adds a model's predictions as a second panel).
- `score_numbering.py` is the one fair measure for everyone. Headline number: **numbering
  accuracy** = teeth found with the right number ÷ all real teeth (a match needs box overlap
  IoU ≥ 0.5). It also gives a 32-row per-tooth table, missing-tooth F1 and mAP50, and writes
  `score.json` and `score.csv` into the run folder. `--rules` applies the anatomy rules first.
- `preprocess_dataset.py` writes a crop + CLAHE copy of a YOLO dataset with the boxes shifted to
  match the crop (`--preview N` saves before/after images). Not needed for numbering (see above).

## 7. Findings JSON contract

`src/opg/schema.py` defines it (the full file is appended to your context bundle). In short:

- `FindingsReport` = `image` (file, width, height, optional `patient_id` and `study_date`)
  + `teeth` + `summary` + `unassigned_findings` + `models` + `schema_version`.
- Each `Tooth` has `fdi`, `bbox`, `confidence`, its `findings`, and `renumbered_from` if the
  anatomy rules changed the model's number.
- Each `Finding` has a `label` from `FINDING_LABELS`, `bbox`, `confidence`, `tooth_fdi`
  (None if not linked) and `source`. Labels (final for v1): diagnoses `caries`, `deep_caries`,
  `periapical_lesion`, `impacted`, `fracture`, `residual_root`, `bone_loss`; treatments `crown`,
  `filling`, `root_canal_filling`, `implant`, `bridge`. Map other datasets' classes onto these.
- Invalid tooth numbers, boxes or confidences raise an error when created.
- Boxes are `[x1, y1, x2, y2]` in pixels of the **original** image. Tooth numbers are always FDI.
- **Changing `schema.py` needs Kavya's sign-off.** A new optional field bumps the minor version
  (1.0 → 1.1, old files still load); a breaking change bumps the major version.
- Full field table and an example: `docs/findings_json.md`.

## 8. Code style

- Python 3.13 (3.11–3.13 work; not 3.14). Type hints on function signatures.
- Use `pathlib.Path`, not string paths. Use `argparse` for scripts, with the usage examples in
  the module docstring.
- Small functions with clear names. Comments explain *why*, not *what*.
- Put logic in `src/opg/` and keep scripts thin, so the logic can be tested.
- Reuse what exists (`opg.fdi`, `opg.data.yolo`, `opg.geometry`, `opg.preprocess`) instead of
  rewriting it.
- No new dependencies without asking Kavya. Core: numpy, opencv-python, pytest. Training:
  torch, ultralytics.

## 9. Naming

- Branches: your first name in lowercase (`kavya`, `yash`, `shubh`, `krisha`, `faizan`).
- Training runs and Drive weight folders: `<person>-<model>-v<N>`, e.g. `yash-teeth-v1`,
  `shubh-clahe-v1`.
- Tests: `tests/test_<module>.py`, functions `test_<what_it_checks>`.
- Datasets: `data/yolo/<name>/`, zips `<name>_yolo_v<N>.zip`.

## 10. Tests

```
.venv\Scripts\activate
pytest
```

Done means the last line says `passed` and no `failed`. Add a test for anything with logic in
it. Tests must not need the real dataset, a GPU or the internet: build tiny fake inputs in
`tmp_path`.

## 11. Git flow

1. Work only on your own branch: `git checkout <name>`.
2. Before each task: `git pull --no-edit origin main`.
3. Commit as you go: `git add <your files>` (by name, never `git add .` or `-A`), `git status`
   (check no data, images, zips or weights),
   `git commit -m "<task number>: <what changed>"`.
4. `git push origin <name>`, then open a pull request on GitHub (base `main`, compare `<name>`)
   and fill in the template. The PR is how you tell Kavya it's ready.
5. One PR per task. Wait for it to be merged before starting the next task.
6. Kavya reviews and merges, or comments with changes to make.

## 12. Don'ts

- Don't train with horizontal flips.
- Don't commit datasets, images, zips, model weights or anything under `data/` or `runs/`.
  They go on the team Drive.
- Don't use real patient X-rays from anywhere outside the public datasets.
- Don't train on, tune on or look closely at the DENTEX **test** split; it is for scoring only.
- Don't change `schema.py`, `fdi.py` or another person's files without asking Kavya.
  If a shared file seems wrong, report it.
- Don't push to `main`. Kavya reviews and merges every PR.
- Don't paste AI-written code you can't explain. Run it and test it first.

## 13. Starter prompt for your AI tool

Upload your `context_<name>.md`, then paste:

> You are helping me on a team Python project. The attached file describes the project, its
> rules and my task. Follow the code style, naming and don'ts in it, reuse the existing
> functions it lists, and never add horizontal flips. Before writing code, tell me in a few
> lines which files you will change and why. Then give complete files, not fragments, plus a
> pytest test for any logic. My task: <paste the task from your track>.
