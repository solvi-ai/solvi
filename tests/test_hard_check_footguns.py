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
        s = System(_idle_catalog(), [Question("ok_to_ship", "ok?", Answer.yes_no())],
                   allow_unreachable_hard_checks=True)
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


def test_default_constructor_refuses_an_unreachable_hard_check_without_executing_callbacks():
    with pytest.raises(ValueError, match="price_positive.*never runs"):
        System(_idle_catalog(), [Question("ok_to_ship", "ok?", Answer.yes_no())])


@pytest.mark.parametrize("flag", [None, 0, 1, "yes", []])
def test_diagnostic_waiver_requires_a_strict_bool(flag):
    with pytest.raises(TypeError, match="must be a bool"):
        System(_idle_catalog(), [Question("ok_to_ship", "ok?", Answer.yes_no())],
               allow_unreachable_hard_checks=flag)


def test_late_orphan_check_blocks_before_cached_answer_or_rule_callbacks():
    calls = []
    catalog = Catalog()

    @catalog.rule("answer")
    def answer(x):
        calls.append(x)
        return "yes"

    system = System(catalog, [Question("answer", "?", Answer.yes_no())], cache=2)
    assert system.ask({"x": 1}, store=False)["answer"].answer == "yes"

    @catalog.check(hard=True)
    def never_called(price):
        calls.append("check")
        return False

    with pytest.raises(ValueError, match="never_called.*never runs"):
        system.ask({"x": 1}, store=False)
    assert calls == [1]


def test_explicit_waiver_keeps_legacy_skip_visible_not_claimed_as_executed():
    with pytest.warns(UserWarning, match="never runs"):
        system = System(_idle_catalog(), [Question("ok_to_ship", "?", Answer.yes_no())],
                        allow_unreachable_hard_checks=True)
    result = system.ask({"days": 3, "price": -1}, store=False)
    assert result["ok_to_ship"].answer == "yes"
    assert "price_positive" not in result.values
    assert any(f.code == "hard_check_never_runs" and f.level == "error" for f in lint(system).findings)


def test_an_unreachable_reader_does_not_make_a_hard_check_reachable():
    catalog = _idle_catalog()

    @catalog.fn
    def unused(price_positive):
        return price_positive

    with pytest.raises(ValueError, match="price_positive.*never runs"):
        System(catalog, [Question("ok_to_ship", "?", Answer.yes_no())])


def test_transitive_dependencies_and_automatic_computed_fact_checks_remain_valid():
    catalog = Catalog()

    @catalog.fn
    def amount(price):
        return price * 2

    @catalog.check(hard=True)
    def positive(amount):
        return amount > 0

    @catalog.rule("answer")
    def answer(amount):
        return "yes"

    system = System(catalog, [Question("answer", "?", Answer.yes_no())])
    result = system.ask({"price": -1}, store=False)
    assert result["answer"].status == "abstain"
    assert result.values["positive"] is False
