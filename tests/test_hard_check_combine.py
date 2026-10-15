"""Several false hard checks with different `then` answers: the first declared decides (as before) with a warning that
names them; Catalog(combine="strictest") gives the answer that comes last in the question's options (or an abstention,
when a governing check sets no answer); replay re-derives either."""
import warnings

import pytest

from solvi import Answer, Catalog, Question, System


def _system(combine=None, order=("b", "a")):
    cat = Catalog() if combine is None else Catalog(combine=combine)
    checks = {"b": ({"size": "half"}, "y"), "a": ({"size": "zero"}, "x")}

    def make(name):
        then, arg = checks[name]
        if arg == "y":
            def b_ok(y: bool) -> bool:
                return y
            fn = b_ok
        else:
            def a_ok(x: bool) -> bool:
                return x
            fn = a_ok
        cat.check(hard=True, then=then)(fn)
    for name in order:
        make(name)

    @cat.rule("size")
    def size(z: bool) -> str:
        return "zero" if z else "full"
    return System(cat, [Question("size", "s", Answer.choice(["full", "half", "zero"]), requires=["a_ok", "b_ok"])])


BOTH = {"x": False, "y": False, "z": True}


def test_first_declared_decides_and_warns_on_conflicting_then():
    s = _system()
    with pytest.warns(UserWarning, match="several hard checks are false"):
        r = s.ask(BOTH, store=False)["size"]
    assert (r.answer, r.status) == ("half", "forced")


def test_one_failed_check_does_not_warn():
    s = _system()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = s.ask({"x": True, "y": False, "z": True}, store=False)["size"]
    assert r.answer == "half"


def test_strictest_takes_the_last_option_whatever_the_declaration_order():
    for order in (("b", "a"), ("a", "b")):
        s = _system("strictest", order)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = s.ask(BOTH)
        assert res["size"].answer == "zero" and res["size"].status == "forced"
        assert res.trace.replay(s, res.flow)["ok"]


def test_combine_is_validated_and_enters_the_fingerprint():
    with pytest.raises(ValueError):
        Catalog(combine="loudest")
    assert _system().fingerprint()["catalog"] != _system("strictest").fingerprint()["catalog"]
    assert _system().fingerprint()["catalog"] == _system("first").fingerprint()["catalog"]
