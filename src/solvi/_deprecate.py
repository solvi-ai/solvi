"""Old names kept for one release: each warns once (DeprecationWarning) and names its replacement; all go in 0.9.

    renamed(old, new)                  # warn that `old` is now `new` (once per old name)
    kwargs(fn, old="new", ...)         # a function that still takes the old keyword names
    attr("old", "new")                 # a property that reads / writes the new attribute under the old name
    module_getattr(__name__, {...})    # a module's __getattr__ for names that moved or were renamed
"""
from __future__ import annotations

import functools
import importlib
import inspect
import warnings

REMOVAL = "0.9"
_seen: set = set()


def renamed(old, new, stacklevel=3):
    """Warn once that `old` is deprecated in favour of `new`."""
    if old in _seen:
        return
    _seen.add(old)
    warnings.warn(f"{old} is deprecated: use {new}; the old name will be removed in solvi {REMOVAL}",
                  DeprecationWarning, stacklevel=stacklevel)


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
