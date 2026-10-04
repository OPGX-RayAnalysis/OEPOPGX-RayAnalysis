import pytest

from opg.associate import assign_findings
from opg.postprocess import apply_anatomy_rules, summarize
from opg.report import dentist_report, patient_summary
from opg.schema import Finding, FindingsReport, ImageInfo
from test_postprocess import full_upper


def make_report():
    teeth = apply_anatomy_rules(full_upper())
    t26 = next(t for t in teeth if t.fdi == 26)
    unassigned = assign_findings(teeth, [Finding("deep_caries", list(t26.bbox), 0.42)])
    return FindingsReport(
        image=ImageInfo("sample.png", 2000, 1000),
        teeth=teeth,
        summary=summarize(teeth, unassigned),
        unassigned_findings=unassigned,
    )


def test_json_round_trip(tmp_path):
    report = make_report()
    report.save(tmp_path / "r.json")
    loaded = FindingsReport.load(tmp_path / "r.json")
    assert loaded == report
    assert loaded.tooth(26).findings[0].label == "deep_caries"


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        Finding("cavity", [0, 0, 1, 1], 0.5)


def test_dentist_report_numbering_and_uncertainty():
    report = make_report()
    fdi_text = dentist_report(report)
    assert "Tooth 26: deep caries" in fdi_text and "(uncertain)" in fdi_text
    universal = dentist_report(report, "universal")
    assert "Tooth 14 (FDI 26)" in universal
    assert "must be reviewed" in universal


def test_patient_summary_is_plain_language():
    text = patient_summary(make_report())
    assert "upper left first molar" in text
    assert "deep cavity" in text


def test_invalid_tooth_box_and_confidence_rejected():
    from opg.schema import Tooth

    with pytest.raises(ValueError):
        Tooth(19, [0, 0, 1, 1], 0.9)  # no tooth 19
    with pytest.raises(ValueError):
        Tooth(11, [5, 0, 1, 1], 0.9)  # x1 > x2
    with pytest.raises(ValueError):
        Finding("caries", [0, 0, 1, 1], 1.5)
    with pytest.raises(ValueError):
        Finding("caries", [0, 0, 1, 1], 0.5, tooth_fdi=99)


def test_minor_versions_load_major_versions_rejected():
    d = make_report().to_dict()
    del d["image"]["patient_id"], d["image"]["study_date"]  # a file written before these existed
    d["schema_version"] = "1.7"
    assert FindingsReport.from_dict(d).image.patient_id is None
    d["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        FindingsReport.from_dict(d)


def test_label_groups_cover_every_label_once():
    from opg.data.dentex import FINDING_CLASSES
    from opg.report import DENTIST_TEXT, PATIENT_TEXT
    from opg.schema import DENTEX_LABELS, DIAGNOSIS_LABELS, FINDING_LABELS, TREATMENT_LABELS

    assert len(set(FINDING_LABELS)) == len(DIAGNOSIS_LABELS) + len(TREATMENT_LABELS)
    assert set(DENTEX_LABELS) <= set(FINDING_LABELS) and FINDING_CLASSES == list(DENTEX_LABELS)
    assert set(DENTIST_TEXT) == set(PATIENT_TEXT) == set(FINDING_LABELS)  # reports can say every label
