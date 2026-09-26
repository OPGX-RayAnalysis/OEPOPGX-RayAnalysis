from opg.associate import assign_findings
from opg.postprocess import RawTooth, apply_anatomy_rules, summarize
from opg.schema import Finding

# Synthetic upper jaw: patient right (quadrant 1) on the image LEFT.
# Tooth centres step 60 px away from a midline at x=1000.


def upper_tooth(fdi, conf=0.9, y=300):
    q, n = divmod(fdi, 10)
    offset = (n - 0.5) * 60
    cx = 1000 - offset if q == 1 else 1000 + offset
    return RawTooth(fdi, [cx - 25, y, cx + 25, y + 120], conf)


def full_upper():
    return [upper_tooth(q * 10 + n) for q in (1, 2) for n in range(1, 9)]


def fdis(teeth):
    return [t.fdi for t in teeth]


def test_correct_input_is_unchanged():
    teeth = apply_anatomy_rules(full_upper())
    assert fdis(teeth) == [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
    assert all(t.renumbered_from is None for t in teeth)


def test_duplicate_number_is_resolved_by_order():
    raw = full_upper()
    raw[4] = RawTooth(14, raw[4].bbox, 0.4)  # tooth in position 15 wrongly called 14, low confidence
    teeth = apply_anatomy_rules(raw)
    assert fdis(teeth).count(14) == 1
    fixed = next(t for t in teeth if t.renumbered_from == 14)
    assert fixed.fdi == 15


def test_wrong_side_is_flipped():
    raw = full_upper()
    raw[2] = RawTooth(23, raw[2].bbox, 0.9)  # tooth 13 (image left) labelled as 23
    teeth = apply_anatomy_rules(raw)
    assert 13 in fdis(teeth) and fdis(teeth).count(23) == 1


def test_gap_becomes_missing_tooth():
    raw = [t for t in full_upper() if t.fdi != 16]
    teeth = apply_anatomy_rules(raw)
    summary = summarize(teeth)
    assert 16 not in fdis(teeth)
    assert 16 in summary.missing_permanent
    assert summary.wisdom_teeth["18"] == "present"
    assert summary.wisdom_teeth["38"] == "missing"  # no lower jaw in this synthetic image


def test_near_duplicate_boxes_are_merged():
    raw = full_upper() + [RawTooth(12, [box + 2 for box in upper_tooth(12).bbox], 0.3)]
    teeth = apply_anatomy_rules(raw)
    assert len(teeth) == 16


def test_findings_attach_to_overlapping_tooth_and_impacted_wisdom():
    teeth = apply_anatomy_rules(full_upper())
    t18 = next(t for t in teeth if t.fdi == 18)
    x1, y1, x2, y2 = t18.bbox
    findings = [
        Finding("impacted", [x1 + 5, y1 + 5, x2 - 5, y2 - 5], 0.8),
        Finding("periapical_lesion", [0, 900, 40, 940], 0.6),  # far from any tooth
    ]
    unassigned = assign_findings(teeth, findings)
    assert findings[0].tooth_fdi == 18
    assert unassigned == [findings[1]]
    summary = summarize(teeth, unassigned)
    assert summary.wisdom_teeth["18"] == "impacted"
    assert summary.finding_counts == {"impacted": 1, "periapical_lesion": 1}
