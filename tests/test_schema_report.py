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
