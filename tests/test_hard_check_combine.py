"""Several false hard checks with different `then` answers: by default (since 1.2) the answer that comes last in the
question's options decides (or an abstention, when a governing check sets no answer), with a warning that names them;
Catalog(combine="strictest") does the same without the warning; Catalog(combine="first") keeps the declaration order
(before 1.2 the default); where the options give no order the first declared decides, with the warning. Replay
re-derives each."""
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


def test_default_takes_the_strictest_and_warns_on_conflicting_then():
    for order in (("b", "a"), ("a", "b")):
        s = _system(order=order)
        with pytest.warns(UserWarning, match="several hard checks are false.*strictest.*a_ok.*default since 1.2"):
            res = s.ask(BOTH)
        assert (res["size"].answer, res["size"].status) == ("zero", "forced")
        assert res.trace.replay(s, res.flow)["ok"]


def test_default_without_an_order_of_answers_keeps_the_first_declared_with_the_warning():
    cat = Catalog()

    @cat.check(hard=True, then={"size": lambda: "half"})
    def b_ok(y: bool) -> bool:
        return y

    @cat.check(hard=True, then={"size": "zero"})
    def a_ok(x: bool) -> bool:
        return x

    @cat.rule("size")
    def size(z: bool) -> str:
        return "zero" if z else "full"
    s = System(cat, [Question("size", "s", Answer.choice(["full", "half", "zero"]), requires=["a_ok", "b_ok"])])
    with pytest.warns(UserWarning, match="the first declared, b_ok, decides"):
        r = s.ask(BOTH, store=False)["size"]
    assert r.answer == "half"


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
    assert _system().fingerprint()["catalog"] == _system("strictest").fingerprint()["catalog"]
    assert _system().fingerprint()["catalog"] != _system("first").fingerprint()["catalog"]


def test_the_strictest_default_changes_the_fingerprint_from_1_1():
    # 1.1 decided with "first" and left it out of the fingerprint: an explicit "first" keeps that fingerprint, the 1.2
    # default does not, so a decision stored under 1.1 replays against the default as made by another catalog
    from solvi.core.provenance import catalog_fingerprint, digest
    for combine in (None, "strictest", "first"):
        cat = _system(combine).catalog
        unmarked = digest(sorted(catalog_fingerprint(cat)["parts"].items()))
        assert (catalog_fingerprint(cat)["fp"] == unmarked) == (combine == "first")


def test_combine_first_chosen_explicitly_keeps_the_order_without_a_warning():
    s = _system("first")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = s.ask(BOTH)
    assert (res["size"].answer, res["size"].status) == ("half", "forced")
    assert res.trace.replay(s, res.flow)["ok"]


def test_learned_order_gives_the_same_answer_as_the_default_order():
    # the learned order stops at the failed checks that settle a question: with the strictest deciding, every check
    # governing it must have been evaluated first
    for combine, want in ((None, "zero"), ("strictest", "zero"), ("first", "half")):
        s = _system(combine)
        s.learn_order([{"x": True, "y": False, "z": True}] * 30)     # b_ok tends to fail, a_ok to pass: b_ok goes first
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = s.ask(BOTH, store=False)
        assert res["size"].answer == want, (combine, res["size"].answer)
