"""The in-process check and load of a module of pure functions over facts (a catalog version's module; moved here from
solvi.experimental.compile.sandbox in 1.1, which re-exports it and keeps the subprocess runner).

    check(source)            # [] when allowed, else the reasons it is refused
    ns = load(source)        # the module's namespace, in this process (refuses what check refuses)

`check` reads the module with `ast` and refuses: an import outside `ALLOWED` (pure standard-library modules: re, math,
datetime, itertools, collections, functools, string, unicodedata, json, fractions, decimal, heapq, bisect, statistics,
typing) or a relative one; a name or a definition from `FORBIDDEN` (open, eval, exec, compile, __import__, globals,
locals, vars, getattr, setattr, delattr, input, breakpoint, exit, quit, help, memoryview, os, sys, subprocess, socket —
getattr & co. would reach a dunder by a string); any dunder name or attribute (`__class__`, `__builtins__`, `__name__`).

`load` executes an allowed module in this process with restricted builtins (the forbidden names removed, `__import__`
admitting `ALLOWED` only, plus `INTERNAL` — `_strptime`, which `datetime.strptime` imports at call time), its source
registered with `linecache` under a name made of its hash, so `inspect.getsource` works and a part's fingerprint is
taken from its syntax tree, the same across processes.

What it does not do: it is not a security boundary against a determined attacker once code is loaded into your process —
the checks close the usual ways out (imports, files, dunders, reflection). A pure function can still loop for ever or
allocate without bound in this process. Load only what passed your checks.
"""
from __future__ import annotations

import ast
import hashlib
import linecache

ALLOWED = frozenset({"re", "math", "datetime", "itertools", "collections", "functools", "string", "unicodedata", "json",
                     "fractions", "decimal", "heapq", "bisect", "statistics", "typing"})
FORBIDDEN = frozenset({"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
                       "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys", "subprocess",
                       "socket"})


class Refused(ValueError):
    """The module breaks the sandbox's rules; `.reasons` lists each one with its line."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("the module is refused by the sandbox: " + "; ".join(self.reasons[:5])
                         + (f" (and {len(self.reasons) - 5} more)" if len(self.reasons) > 5 else ""))


def _dunder(s):
    return s.startswith("__") and s.endswith("__")


def check(source: str) -> list[str]:
    """The reasons the sandbox refuses this module, each with its line ([] — allowed). See the module docs."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    why = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED:
                    why.append(f"line {line}: import {a.name} is not allowed (allowed: {', '.join(sorted(ALLOWED))})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED:
                why.append(f"line {line}: from {'.' * node.level}{node.module or ''} import ... is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN:
                why.append(f"line {line}: the name {node.id!r} is not allowed")
            elif _dunder(node.id):
                why.append(f"line {line}: the dunder name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if _dunder(node.attr):
                why.append(f"line {line}: the dunder attribute .{node.attr} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in FORBIDDEN or _dunder(node.name):
                why.append(f"line {line}: defining {node.name!r} is not allowed")
        elif isinstance(node, ast.arg):
            if node.arg in FORBIDDEN or _dunder(node.arg):
                why.append(f"line {line}: the argument name {node.arg!r} is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            why.append(f"line {line}: global / nonlocal state is not allowed (parts are pure functions)")
    return list(dict.fromkeys(why))


# what an allowed module imports from C at call time through the caller's builtins (`datetime.strptime` imports
# `_strptime`): admitted by the import hook, never by `check` — a module may not import these itself
INTERNAL = frozenset({"_strptime"})


def _guarded_import(real):
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level or (name.split(".")[0] not in ALLOWED and name not in INTERNAL):
            raise ImportError(f"import {name} is not allowed in the sandbox")
        return real(name, globals, locals, fromlist, level)
    return guarded


def safe_builtins():
    """The builtins a sandboxed module sees: the forbidden names removed, `__import__` admitting ALLOWED only."""
    import builtins
    safe = {k: v for k, v in vars(builtins).items() if k not in FORBIDDEN}
    safe["__import__"] = _guarded_import(builtins.__import__)
    safe["__build_class__"] = builtins.__build_class__
    return safe


def source_name(source: str) -> str:
    """The file name a module is compiled under: `<solvi-sandbox-<sha256 prefix>>`, the same for the same text."""
    return f"<solvi-sandbox-{hashlib.sha256(source.encode()).hexdigest()[:16]}>"


def load(source: str, extra: dict | None = None) -> dict:
    """Execute an allowed module in this process → its namespace. Raises `Refused` with the reasons when `check` refuses
    it. `extra`: names put into the namespace before it runs (trusted helpers such as `Fail`)."""
    why = check(source)
    if why:
        raise Refused(why)
    name = source_name(source)
    lines = source.splitlines(keepends=True)
    linecache.cache[name] = (len(source), None, lines, name)    # mtime None: never evicted by checkcache
    ns = {"__builtins__": safe_builtins(), "__name__": name}
    ns.update(extra or {})
    exec(compile(source, name, "exec"), ns)                      # noqa: S102 — checked above, restricted builtins
    return ns


__all__ = ["ALLOWED", "FORBIDDEN", "INTERNAL", "Refused", "check", "load", "safe_builtins", "source_name"]
