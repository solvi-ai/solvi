"""A hard check that no question's flow reaches warns at System() and is a `solvi check` error; a hard check reached
through `requires=` that fails without a `then=` answer keeps its Fail reason in `why`."""
import warnings

import pytest

from solvi import Answer, Catalog, Fail, Question, System
from solvi.check import lint


def _idle_catalog():
    cat = Catalog()

    @cat.check(hard=True)
    def price_positive(price):
        return price > 0

    @cat.rule("ok_to_ship")
    def r(days):
        return "yes" if days < 5 else "no"
    return cat


def test_a_hard_check_no_flow_reaches_warns_at_build_and_is_a_lint_error():
    with pytest.warns(UserWarning, match="price_positive.*never runs"):
        s = System(_idle_catalog(), [Question("ok_to_ship", "ok?", Answer.yes_no())])
    rep = lint(s)
    assert not rep.ok
    assert [(f.code, f.where) for f in rep.findings if f.level == "error"] == [("hard_check_never_runs",
                                                                                    "price_positive")]


def test_a_hard_check_reached_by_requires_or_then_does_not_warn():
    cat = _idle_catalog()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = System(cat, [Question("ok_to_ship", "ok?", Answer.yes_no(), requires=["price_positive"])])
    assert s.ask({"price": 0.0, "days": 3}, store=False)["ok_to_ship"].status == "abstain"
    assert not [f for f in lint(s).findings if f.code == "hard_check_never_runs"]
    cat2 = Catalog()

    @cat2.check(hard=True, then={"ok_to_ship": "no"})
    def price_positive(price):
        return price > 0

    @cat2.rule("ok_to_ship")
    def r(days):
        return "yes"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        System(cat2, [Question("ok_to_ship", "ok?", Answer.yes_no())])


def test_the_fail_reason_of_a_required_hard_check_without_then_is_in_why():
    cat = Catalog()

    @cat.check(hard=True)
    def adult(age):
        return True if age >= 18 else Fail(f"age {age} < 18")

    @cat.rule("may_buy")
    def mb(age):
        return "yes"
    s = System(cat, [Question("may_buy", "?", Answer.yes_no(), requires=["adult"])])
    r = s.ask({"age": 9}, store=False)["may_buy"]
    assert r.answer is None and r.status == "abstain"
    assert r.why == "hard check adult is false: age 9 < 18 and no answer is set for it"
