"""What every `solvi` command shares — usage errors, JSON output, a System named on the command line — in one module that
the command line (solvi.cli) and the library modules that carry a command (solvi.serve, solvi.check, solvi.models,
solvi.calibfile) import, so no library module imports from solvi.cli. A usage error ends the command with status 2;
library code that is not a command uses solvi.loader, which raises LoadError."""
from __future__ import annotations

import sys

from .loader import LoadError, load_module as _load
from .schema import dumps


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
    For the command: a name that cannot be read ends it with status 2 (library code uses solvi.loader, which raises
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
