# Track: Yash (tooth-numbering model)

**Phase 2 tasks (25 Oct to Thu 5 Nov 2026, hard deadline 5 Nov):** 2.4 tune numbering, 2.5 molar-only anatomy rules.
**Phase 1 (done):** 1.7 v1 = yolo11s, 94.4% numbering accuracy on the test split (95% CI 92.5% to 96.0%);
2.13 seeds 42, 1, 2: mean 94.2% (SD 0.3 points).

## Goal
Freeze the numbering model for the rest of the system: the best setting we can find, measured fairly.

## Depends on nothing (start now)
- **2.4 Tune numbering.** Train yolo11m (1024 px) and yolo11s at 1280 px. Pick the winner on the **val**
  split against v1 (`compare_scores.py`), then score only the winner on **test**, with and without `--rules`.
  Upload the winner to Drive `weights/<run>/` and tell Krisha (she scores it on Tufts).
- **2.5 Molar-only anatomy rules.** Phase 1 showed the rules fix wisdom teeth but hurt premolars.
  New option: only molars (positions 6, 7, 8) take the rules' number; every other tooth keeps the
  model's own number, exactly as with rules off. Add it to `src/opg/postprocess.py` as an argument of
  `apply_anatomy_rules` (default unchanged) and to `score_numbering.py` as `--molars-only` (used with
  `--rules`, writes `score_rules_molars.*`). Add tests. Compare off / all / molars-only on **val**, report on test.

## Depends on Kavya's work
- **Whole-scan accuracy (2.15).** When Kavya posts that 2.15 is merged, `git pull` and re-score your
  final runs: `score.json` then also has `whole_scan_accuracy`. Put it in `EXPERIMENTS.md`.

## Files you may touch
- `experiments/yash/EXPERIMENTS.md`
- `src/opg/postprocess.py` and `scripts/score_numbering.py`: only the molar-only option (2.5), with tests in
  `tests/test_postprocess.py`. Don't change the default behaviour.
- `scripts/train_yolo.py` only if needed: flips must stay off (`fliplr=0.0`).

## Definition of done
- 2.4: winner chosen on val, scored once on test (rules off and on, with CI), best.pt on Drive, Krisha told, PR merged.
- 2.5: option + tests merged, three-way comparison in `EXPERIMENTS.md`.
- `pytest` passes.

## Starter commands
```
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name yash-teeth-v2-m --model yolo11m.pt --batch 4
python scripts/train_yolo.py --data data/yolo/teeth/data.yaml --name yash-teeth-v2-s1280 --imgsz 1280 --batch 4
python scripts/score_numbering.py --weights runs/<run>/weights/best.pt --data data/yolo/teeth/data.yaml --split val --out runs/<run>/val
python scripts/compare_scores.py runs/yash-teeth-v1-3/val runs/<run>/val
```
Choose settings on **val**; use **test** only to report the winner.
