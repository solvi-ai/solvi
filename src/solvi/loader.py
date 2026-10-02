"""Loading an object named on a command line or in a config: "module:attr" or "file.py:attr".

    from solvi.loader import load_object
    system = load_object("catalog.py:system")        # a function is called: a System factory

A name that cannot be read — no ":", a missing file, a missing attribute — raises LoadError (never SystemExit: this is
library code; the `solvi` command turns it into its own error message). Whatever the module itself raises while it is
imported passes through."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys


class LoadError(Exception):
    """A "module:attr" / "file.py:attr" name that cannot be read."""


def load_object(spec):
    """"module:attr" or "file.py:attr" → the attribute (called when it is a function: a System factory)."""
    return load_module(spec)[1]


def load_module(spec):
    """"module:attr" or "file.py:attr" → (the module, the attribute — called when it is a function: a System factory)."""
    mod_name, _, attr = spec.rpartition(":")
    if not mod_name or not attr:
        raise LoadError(f"expected module:attribute or file.py:attribute, got {spec!r}")
    if mod_name.endswith(".py") or os.sep in mod_name:
        path = os.path.abspath(mod_name)
        if not os.path.isfile(path):
            raise LoadError(f"no such file: {mod_name}")
        sys.path.insert(0, os.path.dirname(path))
        s = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(s)
        sys.modules[s.name] = mod
        s.loader.exec_module(mod)
    else:
        sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(mod_name)
    if not hasattr(mod, attr):
        raise LoadError(f"{mod_name} has no attribute {attr!r}")
    obj = getattr(mod, attr)
    if callable(obj) and not hasattr(obj, "ask") and not callable(getattr(obj, "decision", None)):
        obj = obj()
    return mod, obj
