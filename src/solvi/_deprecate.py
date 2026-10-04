"""Old names: the warning a name gets for the one release it is kept, and the clear error it gives once it is gone.

    renamed(old, new)                  # warn that `old` is now `new` (once per old name): a name kept for one release
    removed_kwargs(old="new", ...)     # a function (or, above @dataclass, a class) that refuses the old keyword names
                                       # with a TypeError naming the new one
    removed_attr("old", "new", owner)  # an attribute or method that is gone: an AttributeError naming the new one

The names 0.8 renamed were kept with a warning through 0.8 and removed in 0.9 (CHANGELOG, 0.9 "Breaking changes").

The 1.0 layout moved most modules (`solvi.typed` → `solvi.core.types`). One table, MOVED, says where each went: it keeps
fingerprints (below) and gives the old paths for 1.0.x — `import solvi.typed` imports solvi.core.types itself with a
SolviDeprecationWarning (the old-path finder, first on sys.meta_path once solvi is imported), and `solvi migrate`
rewrites code to the new paths. The old paths are removed in 1.1.
"""
from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.machinery
import inspect
import sys
import warnings

SINCE = "0.8"
REMOVED = "0.9"
_seen: set = set()

# Moved code keeps its fingerprints. A fingerprint records the module of a declared type, of a callable object's type and
# of an object stored by its class (solvi.provenance.fp_module, solvi.runtime): moving a class or function to another
# module would change every catalog, decision and store fingerprint that names it, and the decisions stored before the
# move would no longer replay. Each move adds a line here — new location → the 0.9 module — and fingerprints record the
# 0.9 name:
#     "solvi.core.types": "solvi.typed",                  # a whole module moved: everything defined in it
#     "solvi.core.response:Response": "solvi.system",     # one name moved out of a module that stays
# A "module:Name" line wins over a module line. Entries are never removed: stored records name the 0.9 modules forever.
MOVED: dict[str, str] = {
    # 1.0 layout, the kernel (solvi.core): whole modules
    "solvi.core.catalog": "solvi.core",
    "solvi.core.types": "solvi.typed",
    "solvi.core.provenance": "solvi.provenance",
    "solvi.core.primitives": "solvi.primitives",
    "solvi.core.runtime": "solvi.runtime",
    "solvi.core.schema": "solvi.schema",
    "solvi.core.textin": "solvi.textin",
    "solvi.core.costs": "solvi.costs",
    "solvi.core._i18n": "solvi.i18n",
    "solvi.core.calibration": "solvi.calibration",
    "solvi.core.calibfile": "solvi.calibfile",
    "solvi._loader": "solvi.loader",
    # 1.0 layout, the low-level areas (solvi.core.<area>)
    "solvi.core.slow.agree": "solvi.agree",
    "solvi.core.store.audit": "solvi.audit",
    "solvi.core.deciders": "solvi.decide",
    "solvi.core.deciders.adapt": "solvi.decide.adapt",
    "solvi.core.deciders.backends": "solvi.decide.backends",
    "solvi.core.deciders.capabilities": "solvi.decide.capabilities",
    "solvi.core.deciders.gate": "solvi.decide.gate",
    "solvi.core.deciders.kinds": "solvi.decide.kinds",
    "solvi.core.deciders.model": "solvi.decide.model",
    "solvi.core.deciders.part": "solvi.decide.part",
    "solvi.core.deciders.state": "solvi.decide.state",
    "solvi.core.deciders.wire": "solvi.decide.wire",
    "solvi.core.store.diff": "solvi.diff",
    "solvi.core.dispatch": "solvi.dispatch",
    "solvi.core.guarantees.drift": "solvi.drift",
    "solvi.core.knowledge.episodes": "solvi.episode",
    "solvi.core.extract": "solvi.extract_long",
    "solvi.core.slow.generate": "solvi.generate",
    "solvi.core.guarantees.guarantee": "solvi.guarantee",
    "solvi.core.deciders.heads": "solvi.heads",
    "solvi.core._inputs": "solvi.inputs",
    "solvi.core.deciders.llm": "solvi.llm",
    "solvi.core.deciders.longdoc": "solvi.longdoc",
    "solvi.core.knowledge.memory": "solvi.memory",
    "solvi.core.deciders.combine": "solvi.multi",
    "solvi.core.guarantees.openset": "solvi.openset",
    "solvi.core.deciders.perturb": "solvi.perturb",
    "solvi.core.slow.refine": "solvi.refine",
    "solvi.core.deciders._remote": "solvi.remote",
    "solvi.core.store.report": "solvi.report",
    "solvi.core.deciders.rulelist": "solvi.rulelist",
    "solvi.core.slow.search": "solvi.search",
    "solvi.core.sets": "solvi.sets",
    "solvi.core.store.signature": "solvi.signature",
    "solvi.core.store": "solvi.storage",
    "solvi.core.plan.strategist": "solvi.strategist",
    "solvi.core.plan.cost": "solvi.strategy",
    "solvi.core.store.sysreport": "solvi.sysreport",
    "solvi.core.system": "solvi.system",
    "solvi.core.deciders.systemone": "solvi.systemone",
    "solvi.core.knowledge.worldmap": "solvi.worldmap",
    # 1.0 layout, experimental (solvi.experimental.*)
    "solvi.experimental.mcp": "solvi.agents.mcp",
    "solvi.experimental.charts": "solvi.charts",
    "solvi.experimental.charts.check": "solvi.charts.check",
    "solvi.experimental.charts.propose": "solvi.charts.propose",
    "solvi.experimental.charts.render": "solvi.charts.render",
    "solvi.experimental.charts.spec": "solvi.charts.spec",
    "solvi.experimental.compile": "solvi.compile",
    "solvi.experimental.hooks": "solvi.hooks",
    "solvi.experimental.learning": "solvi.learning",
    "solvi.experimental.lora": "solvi.lora",
    "solvi.experimental.compile.sandbox": "solvi.sandbox",
    "solvi.experimental.specialist": "solvi.specialist",
    "solvi.experimental.counterfactual": "solvi.counterfactual",     # removed for 1.0, restored as experimental
    # 1.0 layout, the high level (solvi.solutions.*, the cli and testing packages)
    "solvi.solutions.decisions": "solvi.auto",
    "solvi.solutions.guard": "solvi.agents.guard",
    "solvi.solutions.guard.confirm": "solvi.agents.confirm",
    "solvi.solutions.guard.intents": "solvi.agents.intents",
    "solvi._command": "solvi.command",
    "solvi.cli._scaffold": "solvi.scaffold",
    "solvi.testing.honesty": "solvi.honesty",
    "solvi.cli._models:_question": "solvi.models",
    "solvi.cli._models:_input": "solvi.models",
    "solvi.cli._models:measure": "solvi.models",
    "solvi.cli._models:_read_examples": "solvi.models",
    "solvi.cli._models:_mb": "solvi.models",
    "solvi.cli._models:cmd_list": "solvi.models",
    "solvi.cli._models:cmd_pull": "solvi.models",
    "solvi.cli._models:cmd_check": "solvi.models",
    "solvi.cli._models:_report": "solvi.models",
    "solvi.cli._models:add_parser": "solvi.models",
    "solvi.cli._models:cmd_models": "solvi.models",
    # the import-cycle lane's moves (names that left a 0.9 module), at their 1.0 paths
    "solvi.core.response:Response": "solvi.system",
    "solvi.core.chain:append": "solvi.system",
    "solvi.core.sources:TRUSTED_SOURCES": "solvi.storage",
    "solvi.core.sources:VERIFIED": "solvi.storage",
    "solvi.core.sources:VERIFIED_REFUSED": "solvi.storage",
    "solvi.core.sources:UntrustedLabel": "solvi.storage",
    "solvi.core.sources:check_source": "solvi.storage",
    "solvi.core.calibration:GroupBy": "solvi.decide.gate",
    "solvi.core.calibration:group_name": "solvi.decide.gate",
    "solvi.core.runtime:plan_batches": "solvi.decide.part",
    "solvi.core.costs:_usages": "solvi.dispatch",
    "solvi.core.costs:price_of": "solvi.dispatch",
    "solvi.core.costs:Budget": "solvi.dispatch",
    "solvi.core.costs:Cost": "solvi.dispatch",
    "solvi.core.costs:BudgetStop": "solvi.dispatch",
    "solvi.core.costs:recorded_calls": "solvi.dispatch",
    "solvi.core.costs:cost_of": "solvi.dispatch",
    "solvi._rpc:Limits": "solvi.serve",
    "solvi._rpc:RequestError": "solvi.serve",
    "solvi._rpc:NotFound": "solvi.serve",
    "solvi._rpc:BadRequest": "solvi.serve",
    "solvi._rpc:Busy": "solvi.serve",
    "solvi._rpc:too_deep": "solvi.serve",
    "solvi._rpc:parse_json": "solvi.serve",
    "solvi._rpc:internal_error": "solvi.serve",
    "solvi._rpc:_readline": "solvi.serve",
    "solvi.cli._calibrate:add_parser": "solvi.calibfile",
    "solvi.cli._calibrate:cmd_calibrate": "solvi.calibfile",
    "solvi.cli._calibrate:examples_of": "solvi.calibfile",
    "solvi.cli._calibrate:find_part": "solvi.calibfile",
    "solvi.cli._calibrate:label_of": "solvi.calibfile",
    "solvi.cli._calibrate:read_rows": "solvi.calibfile",
}


# --- old module paths (the 1.0 layout): a 0.9 path still imports for one release, with a warning, as the same module
# The table is MOVED itself: a module line maps a 1.0 path to the 0.9 path it replaces. What it does not say:
OLD_PATHS_REMOVED_IN = "1.1"
EXTRA_PATHS: dict[str, str] = {"solvi.agents": "solvi.solutions.guard"}   # a 0.9 package that only re-exported → its 1.0 module
KEPT = {"solvi.core"}                  # a 0.9 path that is still a module in 1.0 (solvi.core: the package, re-exporting)
# a name `solvi` exported in 0.9 and no longer does → the 1.0 module to import it from
TOP_LEVEL_MOVED: dict[str, str] = {
    "AnswerType": "solvi.core", "NotStated": "solvi.core", "FactTypeError": "solvi.core.types",
    "MISSING": "solvi.core.runtime", "Record": "solvi.core.runtime", "Result": "solvi.core.runtime",
    "Trace": "solvi.core.runtime", "Shadow": "solvi.core.store.diff", "TraceStorage": "solvi.core.store",
    "JSONLStorage": "solvi.core.store", "SQLiteStorage": "solvi.core.store", "PostgresStorage": "solvi.core.store",
    "DuckDBStorage": "solvi.core.store",
}


# a module 1.0 removed from inside a 0.9 package that moved (solvi.agents → solvi.solutions.guard) → what to use instead.
# Importing one raises ModuleNotFoundError, always: without this the old package would be imported first (with its
# warning, an error under -W error), and only then would the submodule be missing.
REMOVED_UNDER_OLD_PATHS: dict[str, str] = {
    f"solvi.agents.{name}": "call guard.check (or guard.call) from your framework's tool-execution step; for MCP "
                            "servers, the proxy (solvi serve --guard --upstream)"
    for name in ("pydantic_ai", "langgraph", "openai_agents")
}


def _removed_error(fullname):
    return ModuleNotFoundError(f"No module named {fullname!r}: removed in 1.0; {REMOVED_UNDER_OLD_PATHS[fullname]} "
                               f"(CHANGELOG 1.0, Removed)", name=fullname)


def _removed_import_in_progress():
    """The removed module (REMOVED_UNDER_OLD_PATHS) whose import is importing its old parent package right now, or None:
    the import machinery imports a parent before it looks for the child, so the parent's finder call is the first
    place the child can be refused."""
    frame = sys._getframe(2)
    while frame is not None and "importlib" in frame.f_code.co_filename:
        if frame.f_code.co_name in ("_find_and_load", "_find_and_load_unlocked"):
            name = frame.f_locals.get("name")
            if name in REMOVED_UNDER_OLD_PATHS:
                return name
        frame = frame.f_back
    return None


def old_paths():
    """{0.9 module path: its 1.0 path} — every module that moved (MOVED's module lines, inverted) and EXTRA_PATHS."""
    out = {old: new for new, old in MOVED.items() if ":" not in new and old not in KEPT}
    out.update(EXTRA_PATHS)
    return out


def new_path(old):
    """A 0.9 dotted path (a module, or a name in one: "solvi.typed.Maybe") → its 1.0 path; None when it did not move."""
    paths = old_paths()
    head, rest = old, ""
    while head:
        if head in paths:
            return paths[head] + rest
        head, _, last = head.rpartition(".")
        rest = f".{last}{rest}"
    return None


def _warn_from_caller(message, category=None, skip=0):
    """warnings.warn, located at the first frame outside the import machinery and this module (the user's import);
    skip: frames of the caller's own to skip too (a module __getattr__)."""
    frame, level = sys._getframe(1 + skip), 2 + skip
    while frame is not None:
        name = frame.f_code.co_filename
        if "importlib" in name and "_bootstrap" in name:      # warnings.warn skips these frames itself: not counted
            frame = frame.f_back
        elif name == __file__ or "importlib" in name:
            frame, level = frame.f_back, level + 1
        else:
            break
    warnings.warn(message, category or SolviDeprecationWarning, stacklevel=level)


def moved_message(old, new):
    return f"{old} moved in 1.0: use {new}; the old path is removed in {OLD_PATHS_REMOVED_IN}"


class _OldPathFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports a 0.9 module path as the module at its 1.0 path (the same module object: classes and functions are the
    same objects, isinstance holds across both names), with a SolviDeprecationWarning naming the new path. First on
    sys.meta_path, so that a submodule of an old package path is not found again (and loaded twice) under the old name."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("solvi.") or fullname in sys.modules:
            return None
        if fullname in REMOVED_UNDER_OLD_PATHS:              # its old parent is already imported
            raise _removed_error(fullname)
        new = old_paths().get(fullname)
        if new is None:
            return None
        removed = _removed_import_in_progress()
        if removed is not None and removed.startswith(fullname + "."):   # refused before the parent warns
            raise _removed_error(removed)
        return importlib.machinery.ModuleSpec(fullname, self, loader_state=new)

    def create_module(self, spec):
        module = importlib.import_module(spec.loader_state)
        spec.loader_state = (spec.loader_state, module.__spec__)       # restored after the import system overwrites it
        _warn_from_caller(moved_message(spec.name, module.__name__))
        return module

    def exec_module(self, module):
        spec = module.__spec__
        if spec is not None and isinstance(spec.loader_state, tuple):
            module.__spec__ = spec.loader_state[1]


def install_old_paths():
    """Put the old-path finder first on sys.meta_path (once)."""
    if not any(isinstance(f, _OldPathFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _OldPathFinder())


def import_quietly(path):
    """A dotted module path, old or new, → the module, without the warning: what stored records name ("module:qualname"
    of a 0.9 store) is data, not a user's import."""
    new = new_path(path)
    return importlib.import_module(new if new is not None else path)


def old_attribute(package, name):
    """`package.name` for a name the package no longer has: a module that moved in 1.0 (imported through its old path,
    which warns) → that module; a name `solvi` no longer exports (TOP_LEVEL_MOVED) → it, with the warning; anything
    else → AttributeError."""
    old = f"{package}.{name}"
    if old in old_paths():
        return importlib.import_module(old)
    if package == "solvi" and name in TOP_LEVEL_MOVED:
        _warn_from_caller(moved_message(old, f"{TOP_LEVEL_MOVED[name]}.{name}"), skip=2)
        return getattr(importlib.import_module(TOP_LEVEL_MOVED[name]), name)
    raise AttributeError(f"module {package!r} has no attribute {name!r}")


class SolviDeprecationWarning(FutureWarning):
    """An old solvi name, kept for one release. A FutureWarning, not a DeprecationWarning: Python hides a
    DeprecationWarning outside `__main__` and tests, and these are meant for the people who call the old name. Silence
    them with `warnings.filterwarnings("ignore", category=solvi.SolviDeprecationWarning)`."""


def renamed(old, new, since=SINCE, removal=None, stacklevel=3):
    """Warn once per process that `old` is deprecated in favour of `new`."""
    if old in _seen:
        return
    _seen.add(old)
    gone = f" and will be removed in {removal}" if removal else ""
    warnings.warn(f"{old} is deprecated since {since}{gone}: use {new}", SolviDeprecationWarning, stacklevel=stacklevel)


def _gone(old, new):
    return f"{old} was renamed in {SINCE} and removed in {REMOVED}: use {new}"


def removed_kwargs(fn=None, /, **mapping):
    """A function that refuses the old keyword names in `mapping` (old → "new", or a sentence saying what to do) with a
    TypeError that names the new one. The old names are not in the signature: without this the call would fail too,
    only with Python's "unexpected keyword argument". On a class (a dataclass: above @dataclass) it wraps __init__."""
    if fn is None:
        return functools.partial(removed_kwargs, **mapping)
    if inspect.isclass(fn):
        setattr(fn, "__init__", removed_kwargs(fn.__init__, **mapping))      # noqa: B010 — the type checkers allow it
        return fn
    where = getattr(fn, "__qualname__", fn.__name__).replace(".__init__", "")

    def check(kw):
        for old, new in mapping.items():
            if old in kw:
                raise TypeError(_gone(f"{where}({old}=)", new if not new.isidentifier() else f"{new}="))

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*args, **kw):
            if kw:
                check(kw)
            return await fn(*args, **kw)
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if kw:
            check(kw)
        return fn(*args, **kw)
    return wrapper


def removed_attr(old, new, owner=""):
    """A class attribute `old` that is gone: reading or writing it raises an AttributeError that names `new` (an
    attribute, `name()` for a method, or a sentence)."""
    label = f"{owner}.{old}" if owner else old

    def gone(self, *_):
        raise AttributeError(_gone(label, new))
    return property(gone, gone, doc=f"Removed in {REMOVED}: {new}.")
