"""`solvi migrate PATH [--check]`: rewrite the 0.9 import paths in your code to the 1.0 ones.

    solvi migrate src/            # rewrites .py and .md files in place, prints each file it changed
    solvi migrate src/ --check    # changes nothing; lists what would change and exits 1 when anything would

1.0 moved most modules (`solvi.core.types` → `solvi.core.types`, `solvi.storage` → `solvi.core.store`, ...). The old paths
still import for one release with a SolviDeprecationWarning; this rewrites them, from the same table the warnings come
from (solvi._deprecate: MOVED and the names that left `solvi` itself):

- dotted paths anywhere in the file — imports, `import solvi.x as y`, attribute chains, strings such as
  `monkeypatch.setattr("solvi.llm.urlopen", ...)` or `"solvi.hooks:rules_system"`, comments;
- `from solvi import storage` (a module imported from the package) → `from solvi.core import store as storage`;
- `from solvi import JSONLStorage` (a name `solvi` no longer exports) → `from solvi.core.store import JSONLStorage`.

Code that reaches a moved module some other way (`getattr(solvi, name)`, `importlib.import_module(f"solvi.{x}")`) is not
seen: run your tests with `-W error::solvi.SolviDeprecationWarning` after migrating."""
from __future__ import annotations

import ast
import os
import re
import sys

from . import _deprecate

SUFFIXES = (".py", ".md")
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".mypy_cache", ".ruff_cache", "site"}


def _dotted_pattern(paths):
    names = sorted(paths, key=len, reverse=True)
    return re.compile(r"(?<![\w.])(" + "|".join(re.escape(n) for n in names) + r")(?![\w])")


def rewrite_dotted(text, paths=None):
    """Every 0.9 dotted module path in `text` → its 1.0 path (the longest old path that matches at a position)."""
    paths = _deprecate.old_paths() if paths is None else paths
    if not paths:
        return text
    return _dotted_pattern(paths).sub(lambda m: paths[m.group(1)], text)


def _split_import(names, indent, paths, top_names):
    """The names of one `from solvi import ...` → the statements that import them in 1.0 (None: nothing to change)."""
    kept, moved = [], []
    for name, asname in names:
        if f"solvi.{name}" in paths:                                  # a module: from solvi import storage
            parent, _, leaf = paths[f"solvi.{name}"].rpartition(".")
            moved.append((parent, leaf, asname or (name if leaf != name else None)))
        elif name in top_names:                                       # a name that left the package's surface
            moved.append((top_names[name], name, asname))
        else:
            kept.append((name, asname))
    if not moved:
        return None
    groups: dict[str, list] = {}
    for parent, leaf, asname in moved:
        groups.setdefault(parent, []).append(f"{leaf} as {asname}" if asname and asname != leaf else leaf)
    out = []
    if kept:
        out.append("from solvi import " + ", ".join(f"{n} as {a}" if a else n for n, a in kept))
    out += [f"from {parent} import {', '.join(items)}" for parent, items in groups.items()]
    return ("\n" + indent).join(out)


_FROM_SOLVI = re.compile(r"^([ \t]*)from solvi import (\([^)]*\)|[^\n#]*?)([ \t]*(?:#[^\n]*)?)$", re.M)


def _names(spec):
    out = []
    for part in re.sub(r"#[^\n]*", "", spec).strip().strip("()").replace("\n", " ").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, asname = part.partition(" as ")
        out.append((name.strip(), asname.strip() or None))
    return out


def rewrite(text, paths=None, top_names=None):
    """A file's text → the text with the 1.0 paths (see the module docs)."""
    paths = _deprecate.old_paths() if paths is None else paths
    top_names = _deprecate.TOP_LEVEL_MOVED if top_names is None else top_names

    def from_solvi(m):
        indent, spec, tail = m.group(1), m.group(2), m.group(3)
        if not re.fullmatch(r"\(?[\w\s,]*\)?", re.sub(r"#[^\n]*", "", spec)):
            return m.group(0)                                         # not a plain list of names: leave it
        new = _split_import(_names(spec), indent, paths, top_names)
        return m.group(0) if new is None else indent + new + tail

    return rewrite_dotted(_FROM_SOLVI.sub(from_solvi, text), paths)


def files(path):
    """The .py and .md files under `path` (or `path` itself)."""
    if os.path.isfile(path):
        yield path
        return
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for n in sorted(names):
            if n.endswith(SUFFIXES):
                yield os.path.join(root, n)


def migrate(path, check=False, out=None):
    """Rewrite the files under `path` (check=True: only report) → the files that changed (or would)."""
    out = out or sys.stdout
    changed = []
    for f in files(path):
        try:
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        new = rewrite(text)
        if new == text:
            continue
        if f.endswith(".py"):
            try:
                ast.parse(text)
            except SyntaxError as e:
                print(f"{f}: not rewritten (it does not parse: {e.msg}, line {e.lineno})", file=out)
                continue
            try:
                ast.parse(new)
            except SyntaxError as e:                                  # never leave a file that does not parse
                print(f"{f}: not rewritten (the result would not parse: {e.msg}, line {e.lineno})", file=out)
                continue
        changed.append(f)
        old_lines, new_lines = text.splitlines(), new.splitlines()
        n = sum(1 for a, b in zip(old_lines, new_lines) if a != b) + abs(len(new_lines) - len(old_lines))
        print(f"{f}: {n} line(s) {'would change' if check else 'rewritten'}", file=out)
        if not check:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
    return changed


def add_parser(sub):
    p = sub.add_parser("migrate", help="rewrite 0.9 import paths in your code to the 1.0 ones (--check: only report)")
    p.add_argument("path", help="a file or a folder (its .py and .md files)")
    p.add_argument("--check", action="store_true", help="change nothing; exit 1 when a file would change")
    return p


def cmd_migrate(args):
    changed = migrate(args.path, check=args.check)
    if not changed:
        print("nothing to migrate")
    return 1 if (args.check and changed) else 0


__all__ = ["migrate", "rewrite", "rewrite_dotted"]
