"""Old names kept for one release: each warns once per process (SolviDeprecationWarning, a FutureWarning, so Python shows
it to users and not only in tests) and names its replacement; all go in 0.9.

    renamed(old, new)                  # warn that `old` is now `new` (once per old name)
    kwargs(fn, old="new", ...)         # a function that still takes the old keyword names
    @init_kwargs(old="new", ...)       # a class (a dataclass: above @dataclass) whose __init__ still takes them
    attr("old", "new")                 # a property that reads / writes the new attribute under the old name
    module_getattr(__name__, {...})    # a module's __getattr__ for names that moved or were renamed
"""
from __future__ import annotations

import functools
import importlib
import inspect
import warnings
from typing import TypeVar

T = TypeVar("T")

SINCE = "0.8"
REMOVAL = "0.9"
_seen: set = set()


class SolviDeprecationWarning(FutureWarning):
    """An old solvi name, kept for one release. A FutureWarning, not a SolviDeprecationWarning: Python hides
    SolviDeprecationWarning outside `__main__` and tests, and these are meant for the people who call the old name. Silence
    them with `warnings.filterwarnings("ignore", category=solvi.SolviDeprecationWarning)`."""


def renamed(old, new, stacklevel=3):
    """Warn once per process that `old` is deprecated in favour of `new`."""
    if old in _seen:
        return
    _seen.add(old)
    warnings.warn(f"{old} is deprecated since {SINCE} and will be removed in {REMOVAL}: use {new}",
                  SolviDeprecationWarning, stacklevel=stacklevel)


def kwargs(fn=None, /, **mapping):
    """A function that also accepts the old keyword names in `mapping` (old → "new", or old → ("new", convert, "how")
    when the old value must be converted), warning once per name. The old names stay out of the signature. Giving the
    old and the new name together is a TypeError."""
    if fn is None:
        return functools.partial(kwargs, **mapping)
    where = getattr(fn, "__qualname__", fn.__name__).replace(".__init__", "")

    def translate(kw):
        for old, target in mapping.items():
            if old in kw:
                new, convert, how = (target, None, f"{target}=") if isinstance(target, str) else target
                if new in kw:
                    raise TypeError(f"{where}() got both {old}= and {new}=: {old}= is the old name of {new}=")
                renamed(f"{where}({old}=)", how, stacklevel=4)
                value = kw.pop(old)
                kw[new] = value if convert is None else convert(value)
        return kw

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def awrapper(*args, **kw):
            return await fn(*args, **translate(kw))
        return awrapper

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        return fn(*args, **translate(kw))
    return wrapper


def init_kwargs(**mapping):
    """A class decorator (above @dataclass): the class's __init__ also accepts the old keyword names, as kwargs()."""
    def wrap(cls: type[T]) -> type[T]:
        setattr(cls, "__init__", kwargs(cls.__init__, **mapping))      # noqa: B010 — the type checkers allow it here
        return cls
    return wrap


def attr(old, new, owner=""):
    """A property `old` that reads and writes the attribute `new`, warning once."""
    label = f"{owner}.{old}" if owner else old

    def get(self):
        renamed(label, new, stacklevel=3)
        return getattr(self, new)

    def put(self, value):
        renamed(label, new, stacklevel=3)
        setattr(self, new, value)
    return property(get, put, doc=f"Deprecated: `{new}` (removed in {REMOVAL}).")


def method(old, new, owner=""):
    """A method `old` that calls the method `new`, warning once."""
    label = f"{owner}.{old}" if owner else old

    def call(self, *args, **kw):
        renamed(f"{label}()", f"{new}()", stacklevel=3)
        return getattr(self, new)(*args, **kw)
    call.__name__ = old
    call.__doc__ = f"Deprecated: `{new}()` (removed in {REMOVAL})."
    return call


def module_getattr(module, names):
    """A module-level __getattr__: names = {old: "new"} (the same module) or {old: "package.module:new"}."""
    def __getattr__(name):
        target = names.get(name)
        if target is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        mod, _, new = target.rpartition(":")
        renamed(f"{module}.{name}", f"{mod or module}.{new}", stacklevel=3)
        return getattr(importlib.import_module(mod or module), new)
    return __getattr__


class Result(dict):
    """A result dict whose old keys still read (warning once) but are not in it: `Result(d, old="new")`. Iteration,
    equality and JSON see the new keys only."""

    def __init__(self, data=(), owner="", **old):
        super().__init__(data)
        self._old, self._owner = old, owner

    def __missing__(self, key):
        new = self._old.get(key)
        if new is None:
            raise KeyError(key)
        renamed(f"{self._owner}[{key!r}]", f"[{new!r}]", stacklevel=3)
        return self[new]

    def get(self, key, default=None):
        if key not in self and key in self._old:
            return self[key]
        return super().get(key, default)

    def __reduce__(self):                             # pickles / copies as a plain dict
        return (dict, (dict(self),))
