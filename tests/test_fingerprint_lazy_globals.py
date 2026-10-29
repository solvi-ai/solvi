"""A fingerprint does not depend on call order: a module global set lazily on first use (`_X = None`, then filled by
the first call) gives the same code fingerprint before and after that first call."""
import re

import solvi.solutions.guard as guard_mod
from solvi import Catalog
from solvi.core.provenance import catalog_fingerprint, own_code_fingerprint
from solvi.solutions.guard import Guard

_PAT = None


def _pat():
    global _PAT
    if _PAT is None:
        _PAT = re.compile("[0-9]+")
    return _PAT


def _make():
    cat = Catalog()

    @cat.rule("has_digits")
    def has_digits(text) -> bool:
        return bool(_pat().search(text))
    return cat


def test_a_lazily_set_global_does_not_change_the_fingerprint(monkeypatch):
    monkeypatch.setattr(__import__(__name__), "_PAT", None)
    before = catalog_fingerprint(_make())["fp"]
    _pat()
    after = catalog_fingerprint(_make())["fp"]
    assert before == after


def test_own_code_fingerprint_does_not_see_a_lazily_set_global(monkeypatch):
    monkeypatch.setattr(__import__(__name__), "_PAT", None)
    before = own_code_fingerprint(_pat)
    _pat()
    assert own_code_fingerprint(_pat) == before


def _guard():
    g = Guard()

    @g.tool
    def send_email(to: str, body: str) -> str:
        return "sent"
    return g


def _call(g):
    g.call({"name": "send_email", "arguments": {"to": "a", "body": "b"}}, [{"role": "user", "content": "send a b"}])


def test_guard_fingerprint_before_and_after_the_first_call_is_the_same(monkeypatch):
    monkeypatch.setattr(guard_mod, "_CF", None)              # as in a fresh process: the pattern is built on first use
    g = _guard()
    before = g.system("send_email").fingerprint()
    _call(g)
    assert g.system("send_email").fingerprint() == before
    assert _guard().system("send_email").fingerprint() == before


def test_two_guards_called_in_different_orders_have_one_fingerprint(monkeypatch):
    monkeypatch.setattr(guard_mod, "_CF", None)
    a = _guard()
    fa = a.system("send_email").fingerprint()           # fingerprint first, then call
    _call(a)
    monkeypatch.setattr(guard_mod, "_CF", None)
    b = _guard()
    _call(b)                                            # call first, then fingerprint
    fb = b.system("send_email").fingerprint()
    assert fa == fb
    assert fa["parts"]["argument_errors"] == fb["parts"]["argument_errors"]
