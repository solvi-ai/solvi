"""Copy gallery entries (task.py + state.json) into the browser playground as presets and list them in its index.html.

Run from the repo root:  uv run python tools/sync_gallery.py
                         uv run python tools/sync_gallery.py --check   # change nothing; exit 1 when out of sync
(tests/test_audit_1_1.py runs the check, so CI fails when a gallery entry changed and the playground was not synced.)"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PG = ROOT / "spaces" / "playground"


def title(readme: Path, slug: str) -> str:
    head = readme.read_text().splitlines()[0].lstrip("# ").strip() if readme.exists() else slug
    return "Gallery · " + head


def planned():
    """What the playground should hold → {path: bytes} (the presets, app.py, index.html)."""
    out, entries = {}, []
    for d in sorted((ROOT / "gallery").glob("[0-9][0-9]_*")):
        stem = "g" + d.name
        out[PG / "presets" / f"{stem}.py"] = (d / "task.py").read_bytes()
        out[PG / "presets" / f"{stem}.json"] = (d / "state.json").read_bytes()
        entries.append((stem, title(d / "README.md", d.name)))
    app = (PG / "app.py").read_text()
    block = "".join(f'    "{s}": {t!r},\n' for s, t in entries)
    app = re.sub(r"(PRESET_TITLES = \{\n(?:    \"\d\d_[^\n]*\n)+)(?:    \"g\d\d_[^\n]*\n)*", lambda m: m.group(1) + block, app)
    out[PG / "app.py"] = app.encode()
    html = (PG / "index.html").read_text()
    html = re.sub(r'\n\s*<gradio-file name="presets/g\d\d_[^\n]*', "", html)
    files = "".join(f'\n    <gradio-file name="presets/{s}.{ext}" url="presets/{s}.{ext}"></gradio-file>' for s, _ in entries for ext in ("py", "json"))
    html = html.replace('url="presets/09_blank_template.json"></gradio-file>', 'url="presets/09_blank_template.json"></gradio-file>' + files, 1)
    out[PG / "index.html"] = html.encode()
    return out, len(entries)


def stale():
    """The playground files that differ from what the gallery gives → [path]."""
    want, _ = planned()
    return [p for p, b in want.items() if not p.exists() or p.read_bytes() != b]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        bad = stale()
        for p in bad:
            print(f"out of sync: {p.relative_to(ROOT)}")
        return 1 if bad else 0
    want, n = planned()
    for p, b in want.items():
        p.write_bytes(b)
    print(f"{n} gallery entries synced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
