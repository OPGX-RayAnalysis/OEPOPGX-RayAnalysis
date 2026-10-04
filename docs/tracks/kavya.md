# Track: Kavya (lead)

**Tasks:** 1.6, 1.12, 1.11, 1.13 (shared pieces), then 1.5, the Research Gap and Proposed
Approach slides, and reviewing every PR.

## Goal
Get the shared data and tools ready so nobody waits, keep the code consistent, and merge
everyone's work.

## Files you may touch
Anything. A change to `schema.py` bumps `SCHEMA_VERSION` and gets announced to the team.

## Outputs
- 1.6: `dentex_yolo_v1.zip` on Drive (70/15/15 split, portable `data.yaml`)
- 1.12: `scripts/view_labels.py`, `src/opg/data/yolo.py`
- 1.11: `scripts/score_numbering.py`, `src/opg/scoring.py`
- 1.13: `docs/PROJECT_CONTEXT.md`, `docs/tracks/`, `scripts/make_context.py`
- 1.5: Findings JSON v1 and the final findings class list

## Definition of done
Each piece merged through its own PR with `pytest` passing, and the team told it's ready.

## Reviewing a PR
```
git fetch origin
git checkout <name>
git pull
pytest
git checkout kavya
```
Check: only that task's files, no data or weights, flips still off, tests for any logic.
