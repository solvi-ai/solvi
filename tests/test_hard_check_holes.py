"""A failed hard check always overrides — the ways it used to fail open: a check that returns a falsy value that is not
False, `cat.check(f, hard=True)` with the function passed directly, an input key named like the check (in a System,
over HTTP, as a fact given to the agent guard)."""
import pytest

from solvi import Answer, Catalog, Question, System


def paying(body, typed=False, direct=False):
    cat = Catalog()
    if direct:
        def cap(amount, limit):
            return body(amount, limit)
        cat.check(cap, hard=True, then={"pay": "no"})
    elif typed:
        @cat.check(hard=True, then={"pay": "no"})
        def cap(amount, limit) -> bool:
            return body(amount, limit)
    else:
        @cat.check(hard=True, then={"pay": "no"})
        def cap(amount, limit):
            return body(amount, limit)

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    return cat, System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), checkpoints=["cap"])])


OVER = {"amount": 9000, "limit": 100}                  # the cap is broken: "yes" is never right


@pytest.mark.parametrize("value", [0, None, [], "", 0.0])
def test_a_check_that_returns_a_falsy_non_bool_does_not_count_as_passed(value):
    """`return amount and amount <= limit`, a forgotten return, `VENDORS.get(v)`: the step is rejected with the reason and
    the question abstains — it used to answer yes [ok]."""
    r = paying(lambda a, l: value)[1].ask(OVER)
    assert r["pay"].answer is None and r["pay"].status == "abstain"
    step = next(x for x in r.trace.records if x.name == "cap")
    assert "a check returns True or False" in step.error and type(value).__name__ in step.error


def test_a_truthy_non_bool_is_not_a_passed_check_either():
    r = paying(lambda a, l: "ok")[1].ask({"amount": 5, "limit": 100})
    assert r["pay"].status == "abstain"


def test_a_numpy_bool_is_a_bool():
    np = pytest.importorskip("numpy")
    _, s = paying(lambda a, l: np.float64(a) <= l)
    assert s.ask(OVER)["pay"].answer == "no" and s.ask(OVER)["pay"].status == "forced"
    r = s.ask({"amount": 5, "limit": 100})
    assert r["pay"].answer == "yes" and r.trace.replay(s)["ok"]
    assert type(next(x for x in r.trace.records if x.name == "cap").value) is bool


def test_a_bool_check_is_untouched_typed_or_not():
    for typed in (False, True):
        _, s = paying(lambda a, l: a <= l, typed)
        assert (s.ask(OVER)["pay"].answer, s.ask(OVER)["pay"].status) == ("no", "forced")
        assert s.ask({"amount": 5, "limit": 100})["pay"].answer == "yes"


def test_check_with_the_function_passed_directly_keeps_its_options():
    cat, s = paying(lambda a, l: a <= l, direct=True)
    assert cat.parts["cap"].hard and cat.parts["cap"].then == {"pay": "no"}
    assert (s.ask(OVER)["pay"].answer, s.ask(OVER)["pay"].status) == ("no", "forced")


@pytest.mark.parametrize("value", [True, False, None])
def test_an_input_key_named_like_a_check_does_not_replace_the_check(value):
    _, s = paying(lambda a, l: a <= l, typed=True)
    with pytest.raises(ValueError, match="'cap' .*check.*cannot stand in"):
        s.ask({**OVER, "cap": value})


def test_an_input_key_named_like_a_computed_fact_is_refused_too():
    cat = Catalog()

    @cat.fn
    def risk(amount) -> int:
        return 9 if amount > 100 else 1

    @cat.rule("pay")
    def pay(risk):
        return "yes" if risk < 5 else "no"

    s = System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]))])
    assert s.ask({"amount": 9000})["pay"].answer == "no"
    with pytest.raises(ValueError, match="'risk' .*fn"):
        s.ask({"amount": 9000, "risk": 0})


def test_over_http_the_key_is_a_bad_request():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    _, s = paying(lambda a, l: a <= l, typed=True)
    c = TestClient(create_app(system=s))
    assert c.post("/ask", json={"state": OVER}).json()["results"]["pay"]["answer"] == "no"
    for path, body in (("/ask", {"state": {**OVER, "cap": True}}), ("/ask/pay", {**OVER, "cap": True})):
        r = c.post(path, json=body)
        assert r.status_code == 422 and "cap" in r.text and '"yes"' not in r.text, (path, r.status_code, r.text)


def test_a_fact_named_like_a_policy_does_not_switch_the_policy_off():
    from solvi.agents import Guard
    paid = []
    g = Guard(facts={"role": str})

    @g.tool
    def send_payment(amount: float) -> str:
        """Pay."""
        paid.append(amount)
        return "paid"

    @g.policy("send_payment")
    def under_cap(amount) -> bool:
        return amount <= 100

    call, ctx = {"name": "send_payment", "arguments": {"amount": 1000000.0}}, [{"role": "user", "content": "pay 1000000"}]
    assert g.call(call, context=ctx, facts={"role": "finance"}).outcome == "deny"
    with pytest.raises(ValueError, match="under_cap"):
        g.call(call, context=ctx, facts={"role": "finance", "under_cap": None})
    assert paid == []


@pytest.mark.parametrize("body", [lambda amount: None, lambda amount: 0, lambda amount: []])
def test_an_untyped_policy_that_returns_a_falsy_non_bool_does_not_allow_the_call(body):
    from solvi.agents import Guard
    paid = []
    g = Guard()

    @g.tool
    def send_payment(amount: float) -> str:
        """Pay."""
        paid.append(amount)
        return "paid"

    @g.policy("send_payment")
    def under_cap(amount):
        return body(amount)

    d = g.call({"name": "send_payment", "arguments": {"amount": 1000000.0}}, context=[{"role": "user", "content": "pay 1000000"}])
    assert d.outcome != "allow" and paid == []
