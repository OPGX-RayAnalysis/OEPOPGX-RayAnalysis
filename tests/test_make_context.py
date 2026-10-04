import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_context", ROOT / "scripts" / "make_context.py")
make_context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_context)

PEOPLE = ["kavya", "yash", "shubh", "krisha", "faizan"]


@pytest.mark.parametrize("name", PEOPLE)
def test_bundle_has_all_three_parts(name):
    text = make_context.build(name)
    assert "# Project context" in text
    assert f"# Track: {name.capitalize()}" in text
    assert (ROOT / "src" / "opg" / "schema.py").read_text(encoding="utf-8").strip() in text


def test_project_context_stays_short():
    words = len((ROOT / "docs" / "PROJECT_CONTEXT.md").read_text(encoding="utf-8").split())
    assert words < 3000


def test_unknown_name_is_a_clear_error():
    with pytest.raises(SystemExit, match="Choose one of"):
        make_context.build("nobody")
