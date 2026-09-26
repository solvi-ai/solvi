"""Copy gallery entries (task.py + state.json) into the browser playground as presets and list them in its index.html.

Run from the repo root:  uv run python tools/sync_gallery.py"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PG = ROOT / "spaces" / "playground"


def title(readme: Path, slug: str) -> str:
    head = readme.read_text().splitlines()[0].lstrip("# ").strip() if readme.exists() else slug
    return "Gallery · " + head


def main():
    entries = []
    for d in sorted((ROOT / "gallery").glob("[0-9][0-9]_*")):
        stem = "g" + d.name
        shutil.copy(d / "task.py", PG / "presets" / f"{stem}.py")
        shutil.copy(d / "state.json", PG / "presets" / f"{stem}.json")
        entries.append((stem, title(d / "README.md", d.name)))
    app = (PG / "app.py").read_text()
    block = "".join(f'    "{s}": {t!r},\n' for s, t in entries)
    app = re.sub(r"(PRESET_TITLES = \{\n(?:    \"\d\d_[^\n]*\n)+)(?:    \"g\d\d_[^\n]*\n)*", lambda m: m.group(1) + block, app)
    (PG / "app.py").write_text(app)
    html = (PG / "index.html").read_text()
    html = re.sub(r'\n\s*<gradio-file name="presets/g\d\d_[^\n]*', "", html)
    files = "".join(f'\n    <gradio-file name="presets/{s}.{ext}" url="presets/{s}.{ext}"></gradio-file>' for s, _ in entries for ext in ("py", "json"))
    html = html.replace('url="presets/09_blank_template.json"></gradio-file>', 'url="presets/09_blank_template.json"></gradio-file>' + files, 1)
    (PG / "index.html").write_text(html)
    print(f"{len(entries)} gallery entries synced")


if __name__ == "__main__":
    main()
