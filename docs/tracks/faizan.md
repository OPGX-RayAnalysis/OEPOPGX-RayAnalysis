# Track: Faizan (checker)

**Tasks:** 1.15 (converter tests), 1.16 (label spot-check). Also: Introduction and Motivation slides.

## Goal
Check that our tooth-number conversions are right, and find wrong labels in the data.

## Files you may touch
- `tests/test_fdi_chart.py` (create). Nothing else in the code.
- Drive: the `label_spotcheck` sheet and images in `results/faizan/`.

## 1.15 Converter check
- Type out all 32 adult teeth from a trusted dental chart: FDI, Universal, Palmer, tooth name.
  Put the chart's source (link or book) in a comment at the top.
- A loop checks that `opg.fdi` gives the same answers both ways: `to_universal`,
  `from_universal`, `to_palmer`, `from_palmer`, and `describe` for the name.
- If a test fails, post it in the team WhatsApp group. Don't change `fdi.py` yourself.
- Done: `pytest` passes (or the failures are reported), PR opened.

## 1.16 Label spot-check (no coding)
- Run the view command below, open each image, and check every tooth number against the chart.
- Log each problem as a row in the sheet: image, tooth, problem
  (e.g. `train_187.png | 36 | labelled 37`).
- Copy flagged images into Drive `results/faizan/`.
- Done: 30 images checked, sheet filled (or "no problems" written).

## Starter commands
```
pytest tests/test_fdi_chart.py
python scripts/view_labels.py --data data/yolo/teeth/data.yaml --split train --n 30 --random --seed 1 --out spotcheck/
```
The patient's right is on the image's left, so 18 is top far left and 48 bottom far left.
Quadrant colours: 1 red, 2 green, 3 blue, 4 orange.

## Git in 5 steps
1. `git checkout faizan`, then `git pull --no-edit origin main`
2. Make your change, then run `pytest`
3. `git add tests/test_fdi_chart.py`, then `git status` (only your file listed)
4. `git commit -m "1.15: FDI chart test"`, then `git push origin faizan`
5. On GitHub: Pull requests → New pull request → base `main`, compare `faizan` → fill in the template → Create
