"""A rule that reads a mutable module-level container (a registry dict, a list) keeps one fingerprint within a process
while the container's content changes; a changed constant in a new process is still seen (its first value is hashed)."""
import warnings

from solvi import Catalog
from solvi.core.provenance import catalog_fingerprint

_REG = {}
LIMITS = [1, 2]


def _make():
    cat = Catalog()

    @cat.rule("q")
    def q(x) -> bool:
        return len(_REG) >= 0 and x > LIMITS[0]
    return cat


def test_a_mutated_global_container_does_not_change_the_fingerprint():
    fp1 = catalog_fingerprint(_make())["fp"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _REG["k"] = 1
        fp2 = catalog_fingerprint(_make())["fp"]
        _REG["f"] = _make                             # a function value: not plain data any more
        fp3 = catalog_fingerprint(_make())["fp"]
        catalog_fingerprint(_make())
    assert fp1 == fp2 == fp3
    msgs = [str(x.message) for x in w if "_REG" in str(x.message)]
    assert len(msgs) == 1, msgs                       # said once per global


def test_the_first_value_of_a_container_is_hashed():
    from solvi.core import provenance

    def r1(x):
        return x > LIMITS[0]
    a = provenance.code_fingerprint(r1)
    g = {"LIMITS": [5, 6]}                            # the same code reading another module's LIMITS
    r2 = type(r1)(r1.__code__, {**r1.__globals__, **g}, "r1")
    r2.__qualname__ = r1.__qualname__
    assert provenance.code_fingerprint(r2) != a
