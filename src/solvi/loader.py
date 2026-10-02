"""Loading an object named on a command line or in a config: "module:attr" or "file.py:attr".

    from solvi.loader import load_object
    system = load_object("catalog.py:system")        # a function is called: a System factory

A name that cannot be read — no ":", a missing file, a missing attribute — raises LoadError (never SystemExit: this is
library code; the `solvi` command turns it into its own error message). Whatever the module itself raises while it is
imported passes through.

optional(module, extra, what) imports an optional dependency, or raises ImportError naming the extra that installs it
(`pip install 'solvi[model]'`) — what the model loaders use instead of a bare ModuleNotFoundError."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys


class LoadError(Exception):
    """A "module:attr" / "file.py:attr" name that cannot be read — or a factory it names that says it cannot build
    its object (solvi.hooks:rules_system without SOLVI_HOOK_MODEL)."""


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
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:               # the named module itself (an import inside it passes through)
            if e.name and (mod_name == e.name or mod_name.startswith(e.name + ".")):
                raise LoadError(f"no module named {mod_name!r} (looked in the current folder and on the Python "
                                "path)") from None
            raise
    if not hasattr(mod, attr):
        raise LoadError(f"{mod_name} has no attribute {attr!r}")
    obj = getattr(mod, attr)
    if callable(obj) and not hasattr(obj, "ask") and not callable(getattr(obj, "decision", None)):
        obj = obj()
    return mod, obj


def optional(module, extra, what):
    """Import an optional dependency → the module; when it (or a module it needs) is not installed: ImportError
    "<what> needs <module>: pip install 'solvi[<extra>]'". extra: "model" (torch, transformers), "onnx" (onnxruntime,
    tokenizers, huggingface_hub), ..."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ImportError(f"{what} needs {e.name or module}: pip install 'solvi[{extra}]'") from e
