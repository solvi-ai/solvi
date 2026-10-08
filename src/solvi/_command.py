"""What every `solvi` command shares — usage errors, JSON output, a System named on the command line — in one module that
the command line (solvi.cli) and the library modules that carry a command (solvi.serve, solvi.check, solvi.models,
solvi.core.calibfile) import, so no library module imports from solvi.cli. A usage error ends the command with status 2;
library code that is not a command uses solvi._loader, which raises LoadError."""
from __future__ import annotations

import os
import sys

from ._loader import LoadError, load_module as _load
from .core.schema import dumps


def fail(msg):
    """A usage error: the message on stderr, exit status 2."""
    print(f"solvi: {msg}", file=sys.stderr)
    raise SystemExit(2)


def dump(obj):
    """Print JSON (solvi's own encoder: dates, enums, non-finite floats)."""
    print(dumps(obj, ensure_ascii=False, indent=2, default=repr))


def load_object(spec):
    """"module:attr" or "file.py:attr" → the attribute (called when it is a function: a System factory)."""
    return load_module(spec)[1]


def load_module(spec):
    """"module:attr" or "file.py:attr" → (the module, the attribute — called when it is a function: a System factory).
    For the command: a name that cannot be read ends it with status 2 (library code uses solvi._loader, which raises
    LoadError)."""
    try:
        return _load(spec)
    except LoadError as e:
        fail(str(e))


def load_system(spec):
    """"module:attr" or "file.py:attr" → the System (calling attr when it is a function)."""
    obj = load_object(spec)
    if not hasattr(obj, "ask") or not hasattr(obj, "catalog"):
        fail(f"{spec}: not a solvi System (module:attr or file.py:attr — a System or a function returning one)")
    return obj


__all__ = ["dump", "fail", "load_module", "load_object", "load_system"]


def open_store(path, system=None, must_exist=False):
    """A store named on the command line → open_storage(path, system); a path that cannot be a store ends the command
    with status 2 and one line: a directory, a store that must exist and does not (verify, query, replay), a file in a
    directory that does not exist, or one the database cannot open (sqlite3 / OS errors)."""
    from .core.store import open_storage
    p = os.fspath(path)
    if not p.startswith(("postgresql://", "postgres://")):
        if os.path.isdir(p):
            fail(f"{p} is a directory, not a store (a .db / .sqlite / .duckdb file, a JSON-lines file or a "
                 "postgresql:// URL)")
        if must_exist and not os.path.exists(p):
            fail(f"no such store: {p}")
        parent = os.path.dirname(os.path.abspath(p))
        if not os.path.isdir(parent):
            fail(f"cannot create the store {p}: the directory {os.path.dirname(p) or parent} does not exist")
    try:
        return open_storage(p, system)
    except OSError as e:
        fail(f"cannot open the store {p}: {e.strerror or e}")
    except Exception as e:                                # noqa: BLE001 — sqlite3 / duckdb errors: one line, no traceback
        if type(e).__module__.split(".")[0] in ("sqlite3", "duckdb", "_duckdb"):
            fail(f"cannot open the store {p}: {e}")
        raise
