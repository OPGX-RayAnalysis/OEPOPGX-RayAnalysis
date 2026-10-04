"""Bundle the shared context for one person's AI tool (plan task 1.13).

Joins docs/PROJECT_CONTEXT.md, docs/tracks/<name>.md and the current src/opg/schema.py into
context_<name>.md. Upload that one file to ChatGPT, Gemini or Claude before asking for code.
It is built from the repo each time, so re-run it after pulling main.

Usage:
    python scripts/make_context.py faizan
    python scripts/make_context.py yash --out somewhere/
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(name: str, root: Path = ROOT) -> str:
    track = root / "docs" / "tracks" / f"{name}.md"
    if not track.exists():
        people = sorted(p.stem for p in (root / "docs" / "tracks").glob("*.md"))
        raise SystemExit(f"No track for '{name}'. Choose one of: {', '.join(people)}")
    parts = [
        (root / "docs" / "PROJECT_CONTEXT.md").read_text(encoding="utf-8").strip(),
        track.read_text(encoding="utf-8").strip(),
        "# Appendix: src/opg/schema.py (current)\n\n```python\n"
        + (root / "src" / "opg" / "schema.py").read_text(encoding="utf-8").strip()
        + "\n```",
    ]
    return "\n\n---\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="lowercase first name, e.g. faizan")
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"context_{args.name.lower()}.md"
    text = build(args.name.lower())
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path} ({len(text.split())} words). Upload it to your AI tool.")


if __name__ == "__main__":
    main()
