"""The 1.0 migration table of CHANGELOG.md (old import path → new one), generated from solvi._deprecate.

Run from the repo root:  uv run python tools/migration_table.py           (rewrites the table in CHANGELOG.md)
                         uv run python tools/migration_table.py --check   (exit 1 when the table is out of date)"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
START, END = "<!-- migration table: tools/migration_table.py -->", "<!-- end of migration table -->"


def level(new):
    if any(part.startswith("_") for part in new.split(".")):
        return "internal"
    if new.startswith("solvi.experimental"):
        return "experimental: may change; removed in 1.2 unless it graduates"
    if new.startswith("solvi.core"):
        return "low level: building blocks"
    return "high level: ready to use"


def table():
    sys.path.insert(0, str(ROOT / "src"))
    from solvi import _deprecate
    rows = ["| you imported (0.9) | import now (1.0) | level |", "|---|---|---|"]
    for old, new in sorted(_deprecate.old_paths().items()):
        rows.append(f"| `{old}` | `{new}` | {level(new)} |")
    for name, new in sorted(_deprecate.TOP_LEVEL_MOVED.items()):
        rows.append(f"| `from solvi import {name}` | `from {new} import {name}` | {level(new)} |")
    return "\n".join(rows)


def current():
    m = re.search(re.escape(START) + r"\n(.*?)" + re.escape(END), CHANGELOG.read_text(), re.S)
    return m.group(1).rstrip("\n") if m else None


def main():
    if "--check" in sys.argv:
        ok = current() == table()
        print("the migration table is up to date" if ok else "the migration table is out of date: "
              "uv run python tools/migration_table.py")
        return 0 if ok else 1
    text = CHANGELOG.read_text()
    if current() is None:
        sys.exit(f"CHANGELOG.md has no {START} ... {END} block")
    text = re.sub(re.escape(START) + r"\n.*?" + re.escape(END), lambda _: f"{START}\n{table()}\n{END}", text,
                  flags=re.S)
    CHANGELOG.write_text(text)
    print("rewrote the migration table in CHANGELOG.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
