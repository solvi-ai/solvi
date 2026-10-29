"""Two silent ways a part never runs: a rule argument named like a question (answers are not facts) warns at System(),
and an argument with a default value (never used: every argument is a fact) warns when the part is registered."""
import warnings

import pytest

from solvi import Answer, Catalog, Question, System


def test_a_rule_reading_a_question_name_warns_at_build():
    cat = Catalog()

    @cat.rule("a")
    def a(x: int) -> str:
        return "p"

    @cat.rule("b")
    def b(a: str, x: int) -> float:
        return 1.0
    with pytest.warns(UserWarning, match="'a' is a question of this system.*answers are not facts"):
        s = System(cat, [Question("a", "a?", Answer.choice(["p", "q"])), Question("b", "b?", Answer.estimate())])
    assert s.ask({"x": 1}, store=False)["b"].why == "cannot compute: a"        # behaviour itself unchanged
    assert s.ask({"x": 1, "a": "p"}, store=False)["b"].answer == 1.0


def test_a_part_computing_the_name_does_not_warn():
    cat = Catalog()

    @cat.fn
    def a(x: int) -> str:
        return "p"

    @cat.rule("b")
    def b(a: str) -> str:
        return "yes"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        System(cat, [Question("b", "b?", Answer.yes_no())])


def test_an_argument_with_a_default_value_warns_at_registration():
    cat = Catalog()
    with pytest.warns(UserWarning, match="check chk: _lim has a default value.*never used"):
        @cat.check(hard=True, then={"ok": "no"})
        def chk(n: int, _lim: int = 5) -> bool:
            return n <= _lim

    @cat.rule("ok")
    def ok(n: int) -> str:
        return "yes"
    s = System(cat, [Question("ok", "ok?", Answer.choice(["yes", "no"]))])
    assert s.ask({"n": 9}, store=False)["ok"].answer is None                   # as before: the fact is required
    assert s.ask({"n": 9, "_lim": 5}, store=False)["ok"].answer == "no"


def test_arguments_without_a_default_do_not_warn():
    cat = Catalog()
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @cat.fn
        def f(n, *args, **kw):
            return n
