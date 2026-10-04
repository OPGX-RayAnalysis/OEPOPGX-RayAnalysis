# Track: Krisha (Tufts external test set, no GPU needed)

**Task:** 1.14. Also: Related Literature slides (1.4, with Yash). After 17 Oct: 2.2.

## Goal
Check that the model works on OPGs that aren't from DENTEX, using the Tufts Dental Database
(~1,000 OPGs with tooth numbers).

## Files you may touch
- `src/opg/data/tufts.py` (create)
- `scripts/tufts_to_yolo.py` (create)
- `tests/test_tufts.py` (create)
- `docs/datasets/tufts.md` (create)

## Inputs
- Tufts access (request form for academic use; use the college email).
- `opg.fdi.from_universal` (Tufts uses Universal 1–32; convert to FDI).
- `opg.data.dentex.write_yolo` to write the dataset, and `scripts/view_labels.py` to check it.

## Outputs
- `src/opg/data/tufts.py`: reads Tufts annotations, turns tooth outlines into boxes, converts
  Universal → FDI.
- `scripts/tufts_to_yolo.py` writes `data/yolo/tufts_test/` (images, labels, `data.yaml` with a
  `test` split). Use `opg.data.dentex.TEETH_CLASSES` for the class list so the order matches
  DENTEX exactly.
- `docs/datasets/tufts.md`: image count, tooth count, missing teeth, licence terms, anything odd.
- Drive `data/tufts_yolo_v1.zip`, only if the Tufts terms allow sharing inside the team.

## Definition of done
- `tests/test_tufts.py` tests the reader on a tiny fake sample (no real data in tests).
- 20 converted images viewed with `view_labels.py`, and left/right is correct: the patient's
  right is on the image's left (18 top far left).
- `pytest` passes. Yash or Kavya score the model on your set; you don't need a GPU.

## Starter commands
```
python scripts/tufts_to_yolo.py --raw data/raw/tufts --out data/yolo/tufts_test
python scripts/view_labels.py --data data/yolo/tufts_test/data.yaml --split test --n 20 --random --out tufts_check/
python scripts/view_labels.py --data data/yolo/tufts_test/data.yaml --split test --n 3 --system universal --out tufts_check_universal/
```
While waiting for access: write the dataset card outline and the reader skeleton.
