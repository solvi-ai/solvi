"""Old names: the warning a name gets for the one release it is kept, and the clear error it gives once it is gone.

    renamed(old, new)                  # warn that `old` is now `new` (once per old name): a name kept for one release
    removed_kwargs(old="new", ...)     # a function (or, above @dataclass, a class) that refuses the old keyword names
                                       # with a TypeError naming the new one
    removed_attr("old", "new", owner)  # an attribute or method that is gone: an AttributeError naming the new one

The names 0.8 renamed were kept with a warning through 0.8 and removed in 0.9 (CHANGELOG, 0.9 "Breaking changes").
"""
from __future__ import annotations

import functools
import inspect
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
    "solvi.response:Response": "solvi.system",
    "solvi.chain:append": "solvi.system",
    "solvi.sources:TRUSTED_SOURCES": "solvi.storage",
    "solvi.sources:VERIFIED": "solvi.storage",
    "solvi.sources:VERIFIED_REFUSED": "solvi.storage",
    "solvi.sources:UntrustedLabel": "solvi.storage",
    "solvi.sources:check_source": "solvi.storage",
    "solvi.calibration:GroupBy": "solvi.decide.gate",
    "solvi.calibration:group_name": "solvi.decide.gate",
    "solvi.runtime:plan_batches": "solvi.decide.part",
    "solvi.costs:_usages": "solvi.dispatch",
    "solvi.costs:price_of": "solvi.dispatch",
    "solvi._rpc:Limits": "solvi.serve",
    "solvi._rpc:RequestError": "solvi.serve",
    "solvi._rpc:NotFound": "solvi.serve",
    "solvi._rpc:BadRequest": "solvi.serve",
    "solvi._rpc:Busy": "solvi.serve",
    "solvi._rpc:too_deep": "solvi.serve",
    "solvi._rpc:parse_json": "solvi.serve",
    "solvi._rpc:internal_error": "solvi.serve",
    "solvi._rpc:_readline": "solvi.serve",
    "solvi.calibrate:add_parser": "solvi.calibfile",
    "solvi.calibrate:cmd_calibrate": "solvi.calibfile",
    "solvi.calibrate:examples_of": "solvi.calibfile",
    "solvi.calibrate:find_part": "solvi.calibfile",
    "solvi.calibrate:label_of": "solvi.calibfile",
    "solvi.calibrate:read_rows": "solvi.calibfile",
}


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
