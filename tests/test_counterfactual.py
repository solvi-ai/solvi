"""Counterfactual explanations: the smallest change of the given inputs that changes an answer, found by re-running the
deterministic flow only — model parts held at their recorded proposals, never called."""
import datetime
import enum

import pytest

from solvi import Answer, Catalog, Decision, JSONLStorage, Question, System
from solvi.experimental.counterfactual import search as counterfactual


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

    return System(cat, [Question("refund", "Refund?", Answer.choice(["yes", "no"]), requires=["within_window"])])


def test_refund_window_date():
    s = refunds()
    res = s.ask({"purchase_date": datetime.date(2026, 8, 10), "today": datetime.date(2026, 9, 19), "opened": False})
    assert res["refund"].answer == "no" and res["refund"].status == "forced"
    cf = counterfactual(res, "refund")
    texts = [str(c) for c in cf]
    assert "yes if purchase_date ≥ 2026-08-20 (now 2026-08-10)" in texts      # 30 days before today
    assert "yes if today ≤ 2026-09-09 (now 2026-09-19)" in texts
    assert cf.best.changes[0].fact in ("purchase_date", "today")
    assert all(len(c.changes) == 1 for c in cf)
    assert "opened" in cf.searched                   # opened=True keeps "no": not a change of the answer
    assert not any(c.changes[0].fact == "opened" for c in cf)
    assert cf.held == [] and "refund = no [forced]" in str(cf)


def test_a_date_change_is_sized_in_days_not_against_the_dates_ordinal():
    """A date change cost days / 739,000, so any date shift — ten years too — ranked before any change of a number."""
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "reject"})
    def short_stay(start, end):
        return (end - start).days <= 30

    @cat.rule("approve")
    def approve(balance):
        return "approve" if balance >= 5 else "reject"
    s = System(cat, [Question("approve", "Approve?", Answer.choice(["approve", "reject"]), requires=["short_stay"])])
    start = datetime.date(2026, 9, 1)
    res = s.ask({"start": start, "end": datetime.date(2026, 10, 31), "balance": 9})       # 60 days: forced reject
    cf = counterfactual(res, "approve")
    end = next(c for c in cf if c.changes[0].fact == "end")
    assert str(end) == "approve if end ≤ 2026-10-01 (now 2026-10-31)" and end.cost == 1.0
    res = s.ask({"start": start, "end": datetime.date(2026, 10, 3), "balance": 3})         # 32 days, low balance
    cf = counterfactual(res, "approve", max_changes=2)
    assert str(cf.best) == "approve if balance ≥ 5 (now 3) and end ≤ 2026-10-01 (now 2026-10-03)"
    assert cf.best.kind == cf.to_dict()["kind"] == "choice" and cf.to_dict()["found"][0]["kind"] == "choice"
    ranked = counterfactual(s.ask({"start": start, "end": datetime.date(2026, 9, 20), "balance": 4}), "approve")
    assert str(ranked.best) == "approve if balance ≥ 5 (now 4)"                           # 25%, not "move a date"
    assert ranked.not_searched == {}


def test_inputs_read_only_by_a_part_that_did_not_run_are_listed_not_left_out():
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "reject"})
    def funded(balance) -> bool:
        return balance >= 5

    @cat.check
    def enough_notice(start, today) -> bool:
        return (start - today).days >= 14

    @cat.rule("approve")
    def approve(enough_notice):
        return "approve" if enough_notice else "needs_manager"
    s = System(cat, [Question("approve", "Approve?", Answer.choice(["approve", "needs_manager", "reject"]),
                              requires=["funded"])])
    res = s.ask({"balance": 3, "start": datetime.date(2026, 10, 19), "today": datetime.date(2026, 9, 25)})
    cf = counterfactual(res, "approve")
    assert str(cf.best) == "approve if balance ≥ 5 (now 3)"
    assert cf.not_searched == {"start": "read only by a part that did not run in this decision",
                               "today": "read only by a part that did not run in this decision"}


def test_refund_bool_flip_and_target():
    s = refunds()
    res = s.ask({"purchase_date": datetime.date(2026, 9, 10), "today": datetime.date(2026, 9, 19), "opened": True})
    assert res["refund"].answer == "no" and res["refund"].status == "ok"
    cf = counterfactual(res, "refund")
    assert str(cf.best) == "yes if opened = False (now True)"
    assert counterfactual(res, "refund", target="no").found == []


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
                                 requires=["adult"])])


STATE = {"amount": 1200.0, "debt": 1000, "income": 5000, "history": "on time", "age": 30}


def test_amount_threshold_and_model_never_called():
    s = lending()
    res = s.ask(STATE)
    assert res["approve"].answer == "decline"
    n = CALLS["n"]
    cf = counterfactual(res, "approve")
    assert CALLS["n"] == n                           # no model call
    assert str(cf.best) == "approve if amount ≤ 1000 (now 1200)"
    assert cf.held == ["risk"] and "held at their recorded proposals (no model called): risk" in str(cf)
    assert "history" in cf.not_searched              # a string without a domain
    assert cf.evals > 0 and not cf.exhausted


def test_strict_and_integer_bounds_and_two_changes():
    s = lending()
    res = s.ask({**STATE, "amount": 900.0, "debt": 2500})           # dti 0.5 ≥ 0.4
    cf = counterfactual(res, "approve", over=["debt", "income"])
    got = {str(c) for c in cf}
    assert "approve if debt ≤ 1999 (now 2500)" in got               # int: 1999/5000 < 0.4
    assert "approve if income ≥ 6251 (now 5000)" in got
    res2 = s.ask({**STATE, "debt": 2500})                           # amount and dti both block
    one = counterfactual(res2, "approve", max_changes=1)
    assert not one and "no change of" in str(one)
    two = counterfactual(res2, "approve", over=["amount", "debt"])
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
    cf = counterfactual(s.ask({"ratio": 0.55}), "ok")
    assert str(cf.best) == "yes if ratio < 0.4 (now 0.55)"


def test_model_proposal_held_even_when_its_input_changes():
    s = lending()
    res = s.ask({**STATE, "amount": 500.0, "history": "late twice"})
    assert res["approve"].answer == "decline"
    n = CALLS["n"]
    cf = counterfactual(res, "approve", domains={"history": ["on time"]})
    assert CALLS["n"] == n
    assert not cf.found                               # risk stays "high": the model's proposal is held
    assert "history" in cf.searched


def test_domain_bounds_the_search():
    res = lending().ask(STATE)
    assert not counterfactual(res, "approve", over=["amount"], domains={"amount": (1100, 5000)})
    assert str(counterfactual(res, "approve", over=["amount"], domains={"amount": (950, 5000)}).best) == \
        "approve if amount ≤ 1000 (now 1200)"


def _alert(center, band):
    cat = Catalog()

    @cat.rule("alert")
    def alert(value):
        return abs(value - center) > band
    return System(cat, [Question("alert", "Alert?", Answer.yes_no())])


def test_a_domain_that_does_not_hold_the_current_value_is_refused_not_searched_both_ways():
    """value 2.656 with domains={"value": (1.8, 2.3)} used to print "no if value ≤ 2.5" and "no if value ≥ 2.5"."""
    res = _alert(2.0, 0.5).ask({"value": 2.656})
    assert str(counterfactual(res, "alert").best) == "no if value ≤ 2.5 (now 2.656)"
    cf = counterfactual(res, "alert", domains={"value": (1.8, 2.3)})
    assert not cf and cf.searched == [] and "outside the domain given for it (1.8, 2.3)" in cf.not_searched["value"]


def test_the_band_of_a_two_sided_rule_is_found_and_a_miss_is_not_a_flat_no_change():
    """abs(value + 20) > 5 at 40: the doubling probes stepped over the band [-25, -15] and the result said "no change of
    value changes the answer"."""
    res = _alert(-20.0, 5).ask({"value": 40.0})
    cf = counterfactual(res, "alert", domains={"value": (-100.0, 100.0)})
    assert str(cf.best) == "no if value ≤ -15 (now 40)"
    cf = counterfactual(res, "alert")                                 # non-negative by default: the band is out of reach
    assert not cf and "was found that changes the answer" in str(cf) and "narrower band can be missed" in str(cf)
    assert "can be missed" not in str(counterfactual(lending().ask({**STATE, "amount": 900}), "approve", over=["opened"]))


def test_model_part_that_did_not_run_has_no_proposal():
    s = lending()
    res = s.ask({**STATE, "amount": 500.0, "age": 16})               # the hard check failed first: risk never ran
    assert res["approve"].status == "forced" and not any(r.name == "risk" for r in res.trace.records)
    n = CALLS["n"]
    cf = counterfactual(res, "approve", over=["age"])
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
    cf = counterfactual(res, "priority")
    assert {str(c) for c in cf} == {"yes if tier = gold (now basic)", "yes if open_tickets ≥ 4 (now 1)"}
    assert cf.best.cost == 1.0
    d = cf.to_dict()
    assert d["question"] == "priority" and len(d["found"]) == 2
    back = store.get(res.stored_id, s)
    assert {str(c) for c in counterfactual(back, "priority", over=["open_tickets"])} == {"yes if open_tickets ≥ 4 (now 1)"}


def test_errors():
    s = lending()
    res = s.ask(STATE)
    with pytest.raises(KeyError):
        counterfactual(res, "nope")
    with pytest.raises(ValueError):
        counterfactual(res, "approve", max_changes=3)
    res._system = None
    with pytest.raises(ValueError):
        counterfactual(res, "approve")
    cf = counterfactual(s.ask(STATE), "approve", max_evals=3)
    assert cf.exhausted and cf.evals == 3
