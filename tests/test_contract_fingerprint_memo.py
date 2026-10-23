"""Contract.fingerprint() reads `applicable`'s source once per code object (a System's questions fingerprint calls it on
every ask), and the fingerprint still changes when the code changes."""
import hashlib
import inspect

from solvi import Answer, Catalog, Question, System
from solvi.core import contract as contract_mod
from solvi.core.contract import Contract


def one_line(text):
    return "\n" not in text


def two_lines(text):
    return text.count("\n") < 2


def _old_fingerprint(c):
    """The fingerprint as computed before the memo (source read on every call)."""
    from solvi.core.contract import _is_answer_type, _type_name
    r = c.returns
    rt = repr(r) if _is_answer_type(r) else _type_name(r) if r is not None else None
    f = inspect.unwrap(c.applicable)
    code = f"{f.__qualname__}\n{inspect.getsource(f)}"
    echo = list(c.echo) if isinstance(c.echo, tuple) else "auto" if c.echo is None else c.echo
    return hashlib.sha256(repr((rt, echo, code)).encode()).hexdigest()[:16]


def test_the_fingerprint_is_the_one_before_the_memo():
    c = Contract(str, None, one_line)
    assert c.fingerprint() == _old_fingerprint(c) == c.fingerprint()


def test_the_fingerprint_changes_when_the_code_changes():
    def check(text):
        return "\n" not in text
    c = Contract(str, None, check)
    before = c.fingerprint()
    check.__code__ = two_lines.__code__                  # the same function object, new code
    after = c.fingerprint()
    assert after != before
    check.__code__ = one_line.__code__
    assert c.fingerprint() not in (before, after)        # a third body: a third fingerprint
    assert Contract(str, None, two_lines).fingerprint() != Contract(str, None, one_line).fingerprint()


def test_asks_read_the_source_once(monkeypatch):
    def applicable(text):
        return len(text) < 100
    cat = Catalog()

    @cat.rule("short")
    def short(text: str) -> str:
        return "yes" if len(text) < 20 else "no"
    cat.contract("short", applicable=applicable)
    s = System(cat, [Question("short", "short?", Answer.yes_no())])
    calls = []
    real = inspect.getsource

    def counting(obj):
        if inspect.unwrap(obj) is applicable:
            calls.append(obj)
        return real(obj)
    monkeypatch.setattr(inspect, "getsource", counting)
    contract_mod._CODE.pop(applicable, None)
    answers = {s.ask({"text": f"hello world {i}"}, store=False)["short"].answer for i in range(20)}
    assert answers == {"yes"}
    assert len(calls) <= 1
