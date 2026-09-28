"""Counterfactual explanations: the smallest change of the given inputs that changes an answer, found by re-running the
deterministic flow only — model parts held at their recorded proposals, never called."""
import datetime
import enum

import pytest

from solvi import Answer, Catalog, Decision, JSONLStorage, Question, System


# ---------------------------------------------------------------- a refund window
def refunds():
    cat = Catalog()

    @cat.fn
    def days_since_purchase(purchase_date, today):
        return (today - purchase_date).days

    @cat.check(hard=True, then={"refund": "no"})
    def within_window(days_since_purchase):
        return days_since_purchase <= 30

    @cat.rule("refund")
    def refund(opened):
        return "no" if opened else "yes"

    return System(cat, [Question("refund", "Refund?", Answer.choice(["yes", "no"]), checkpoints=["within_window"])])


def test_refund_window_date():
    s = refunds()
    res = s.ask({"purchase_date": datetime.date(2026, 8, 10), "today": datetime.date(2026, 9, 19), "opened": False})
    assert res["refund"].answer == "no" and res["refund"].status == "forced"
    cf = res.counterfactual("refund")
    texts = [str(c) for c in cf]
    assert "yes if purchase_date ≥ 2026-08-20 (now 2026-08-10)" in texts      # 30 days before today
    assert "yes if today ≤ 2026-09-09 (now 2026-09-19)" in texts
    assert cf.best.changes[0].fact in ("purchase_date", "today")
    assert all(len(c.changes) == 1 for c in cf)
    assert "opened" in cf.searched                   # opened=True keeps "no": not a change of the answer
    assert not any(c.changes[0].fact == "opened" for c in cf)
    assert cf.held == [] and "refund = no [forced]" in str(cf)


def test_refund_bool_flip_and_target():
    s = refunds()
    res = s.ask({"purchase_date": datetime.date(2026, 9, 10), "today": datetime.date(2026, 9, 19), "opened": True})
    assert res["refund"].answer == "no" and res["refund"].status == "ok"
    cf = res.counterfactual("refund")
    assert str(cf.best) == "yes if opened = False (now True)"
    assert res.counterfactual("refund", target="no").found == []


# ---------------------------------------------------------------- lending-style limits, a model held fixed
class Scorer:
    model_id = "test/scorer"
    version = "1"


CALLS = {"n": 0}


def lending(margin=0.0):
    cat = Catalog()

    @cat.fn
    def dti(debt, income):
        return debt / income

    @cat.fn(model=Scorer(), options=["low", "high"])
    def risk(history):
        CALLS["n"] += 1
        return Decision("high" if "late" in history else "low", {"high": 0.7, "low": 0.3} if "late" in history
                        else {"high": 0.2, "low": 0.8})

    @cat.check(hard=True, then={"approve": "decline"})
    def adult(age):
        return age >= 18

    @cat.rule("approve")
    def approve(amount, dti, risk):
        if risk == "high":
            return "decline"
        return "approve" if amount <= 1000 and dti < 0.4 else "decline"

    return System(cat, [Question("approve", "Approve the loan?", Answer.choice(["approve", "decline"]),
                                 checkpoints=["adult"])])


STATE = {"amount": 1200.0, "debt": 1000, "income": 5000, "history": "on time", "age": 30}


def test_amount_threshold_and_model_never_called():
    s = lending()
    res = s.ask(STATE)
    assert res["approve"].answer == "decline"
    n = CALLS["n"]
    cf = res.counterfactual("approve")
    assert CALLS["n"] == n                           # no model call
    assert str(cf.best) == "approve if amount ≤ 1000 (now 1200)"
    assert cf.held == ["risk"] and "held at their recorded proposals (no model called): risk" in str(cf)
    assert "history" in cf.not_searched              # a string without a domain
    assert cf.evals > 0 and not cf.exhausted


def test_strict_and_integer_bounds_and_two_changes():
    s = lending()
    res = s.ask({**STATE, "amount": 900.0, "debt": 2500})           # dti 0.5 ≥ 0.4
    cf = res.counterfactual("approve", over=["debt", "income"])
    got = {str(c) for c in cf}
    assert "approve if debt ≤ 1999 (now 2500)" in got               # int: 1999/5000 < 0.4
    assert "approve if income ≥ 6251 (now 5000)" in got
    res2 = s.ask({**STATE, "debt": 2500})                           # amount and dti both block
    one = res2.counterfactual("approve", max_changes=1)
    assert not one and "no change of" in str(one)
    two = res2.counterfactual("approve", over=["amount", "debt"])
    best = two.best
    assert best is not None and len(best.changes) == 2 and best.answer == "approve"
    by = {c.fact: c for c in best.changes}
    assert by["amount"].op == "≤" and by["amount"].to == 1000
    assert by["debt"].op == "≤" and by["debt"].to == 1999
    assert str(best) == "approve if amount ≤ 1000 (now 1200) and debt ≤ 1999 (now 2500)" or \
        str(best) == "approve if debt ≤ 1999 (now 2500) and amount ≤ 1000 (now 1200)"


def test_float_strict_bound():
    cat = Catalog()

    @cat.rule("ok")
    def ok(ratio) -> bool:
        return ratio < 0.4
    s = System(cat, [Question("ok", "OK?", Answer.yes_no())])
    cf = s.ask({"ratio": 0.55}).counterfactual("ok")
    assert str(cf.best) == "yes if ratio < 0.4 (now 0.55)"


def test_model_proposal_held_even_when_its_input_changes():
    s = lending()
    res = s.ask({**STATE, "amount": 500.0, "history": "late twice"})
    assert res["approve"].answer == "decline"
    n = CALLS["n"]
    cf = res.counterfactual("approve", domains={"history": ["on time"]})
    assert CALLS["n"] == n
    assert not cf.found                               # risk stays "high": the model's proposal is held
    assert "history" in cf.searched


def test_domain_bounds_the_search():
    res = lending().ask(STATE)
    assert not res.counterfactual("approve", over=["amount"], domains={"amount": (1100, 5000)})
    assert str(res.counterfactual("approve", over=["amount"], domains={"amount": (950, 5000)}).best) == \
        "approve if amount ≤ 1000 (now 1200)"


def test_model_part_that_did_not_run_has_no_proposal():
    s = lending()
    res = s.ask({**STATE, "amount": 500.0, "age": 16})               # the hard check failed first: risk never ran
    assert res["approve"].status == "forced" and not any(r.name == "risk" for r in res.trace.records)
    n = CALLS["n"]
    cf = res.counterfactual("approve", over=["age"])
    assert CALLS["n"] == n and cf.unavailable == ["risk"] and not cf.found
    assert "without a recorded proposal (did not run): risk" in str(cf)


class Tier(enum.Enum):
    BASIC = "basic"
    GOLD = "gold"


def test_enum_values_and_stored_response(tmp_path):
    cat = Catalog()

    @cat.rule("priority")
    def priority(tier, open_tickets) -> bool:
        return tier == Tier.GOLD or open_tickets > 3
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = System(cat, [Question("priority", "Priority?", Answer.yes_no())], storage=store)
    res = s.ask({"tier": Tier.BASIC, "open_tickets": 1})
    cf = res.counterfactual("priority")
    assert {str(c) for c in cf} == {"yes if tier = gold (now basic)", "yes if open_tickets ≥ 4 (now 1)"}
    assert cf.best.cost == 1.0
    d = cf.to_dict()
    assert d["question"] == "priority" and len(d["found"]) == 2
    back = store.get(res.stored_id, s)
    assert {str(c) for c in back.counterfactual("priority", over=["open_tickets"])} == {"yes if open_tickets ≥ 4 (now 1)"}


def test_errors():
    s = lending()
    res = s.ask(STATE)
    with pytest.raises(KeyError):
        res.counterfactual("nope")
    with pytest.raises(ValueError):
        res.counterfactual("approve", max_changes=3)
    res._system = None
    with pytest.raises(ValueError):
        res.counterfactual("approve")
    cf = s.ask(STATE).counterfactual("approve", max_evals=3)
    assert cf.exhausted and cf.evals == 3
