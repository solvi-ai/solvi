"""solvi check: catalog lint — hard checks outside their question's flow, unused parts, unanswerable questions, cycles,
type conflicts, constraints that cannot all hold, silent defaults in functions that read the input; and the command's
exit statuses."""
import json
from typing import Literal

from solvi import Answer, Catalog, Question, System
from solvi.check import lint, silent_defaults
from solvi.cli import main


def refund_catalog(checkpoint=True):
    cat = Catalog()

    @cat.fn
    def days_since(purchase_day: int, today: int) -> int:
        return today - purchase_day

    @cat.check(hard=True, then={"refund": "no"})
    def known_customer(customer: str) -> bool:
        return customer != "blocked"

    @cat.rule("refund")
    def refund(days_since: int) -> bool:
        return days_since <= 30
    q = Question("refund", "Refund?", requires=["known_customer"] if checkpoint else [])
    return System(cat, [q])


def test_a_clean_catalog_has_no_findings():
    rep = lint(refund_catalog())
    assert rep.ok and rep.findings == [], str(rep)


class DropCheck:
    """A strategist of one's own that plans as the deterministic one, then leaves a check out of every flow."""

    def __init__(self, name):
        self.name = name

    def plan(self, catalog, questions, init_keys, heads=None):
        from solvi.core.plan.strategist import plan
        flow = plan(catalog, questions, init_keys, heads)
        flow.steps = [st for st in flow.steps if st.part.name != self.name]
        flow.per_question = {q: [f for f in fs if f != self.name] for q, fs in flow.per_question.items()}
        return flow


def test_a_hard_check_with_then_is_wired_into_its_questions_flow():
    """1.0: a hard check whose `then` names a question runs in that question's flow without requires= (0.9 answered
    the question as if the check had passed — the bug `then_not_in_flow` caught)."""
    s = refund_catalog(checkpoint=False)
    assert lint(s).findings == []
    res = s.ask({"purchase_day": 1, "today": 5, "customer": "blocked"})
    assert res["refund"].answer == "no" and res["refund"].status == "forced"
    assert "known_customer" in res.flow.per_question["refund"]
    assert any(r == "hard check for refund (then)" for st in res.flow.steps for r in st.reasons)


def test_a_strategist_that_leaves_out_a_then_check_is_refused():
    import pytest
    cat = refund_catalog(checkpoint=False).catalog
    with pytest.raises(ValueError, match=r"hard check known_customer sets `then=` for 'refund'.*does not run it"):
        System(cat, [Question("refund", "Refund?")], strategist=DropCheck("known_customer"))
    s = System(cat, [Question("refund", "Refund?")])
    s.strategist = DropCheck("known_customer")       # set after construction: `solvi check` still tells
    rep = lint(s)
    assert rep.codes("error") == ["then_not_in_flow"] and not rep.ok
    assert "requires=['known_customer']" in rep.errors[0].message


def test_a_hard_check_on_a_computed_fact_is_in_the_flow_without_a_checkpoint():
    s = refund_catalog()

    @s.catalog.check(hard=True, then={"refund": "no"})
    def not_in_future(days_since: int) -> bool:
        return days_since >= 0
    assert lint(s).findings == []
    assert s.ask({"purchase_day": 9, "today": 5, "customer": "ann"})["refund"].status == "forced"


def test_then_mistakes_and_soft_checks():
    cat = Catalog()

    @cat.check(hard=True, then={"refund": "maybe", "ghost": "no"})
    def known(customer) -> bool:
        return customer != "x"

    @cat.check(then={"refund": "no"})
    def polite(text) -> bool:
        return "please" in text

    @cat.rule("refund")
    def refund(known, polite) -> bool:
        return polite
    rep = lint(System(cat, [Question("refund", "Refund?")]))
    assert sorted(rep.codes("error")) == ["then_bad_answer", "then_unknown"]
    assert rep.codes("warning") == ["then_on_soft_check"]


def test_cycles_unused_parts_and_unanswerable_questions():
    cat = Catalog()

    @cat.fn
    def net(gross, tax):
        return gross - tax

    @cat.fn
    def gross(net, tax):
        return net + tax

    @cat.fn
    def spare(tax):
        return tax * 2

    @cat.rule("big")
    def big(net) -> bool:
        return net > 100

    @cat.rule("pick")
    def pick(tax, big) -> Literal["a", "b"]:
        return "a"
    qs = [Question("big", "Big?"), Question("pick", "Pick?"),
          Question("where", "Where?", Answer.span()), Question("later", "Later?", Answer.yes_no(), uses=["tax"])]
    rep = lint(System(cat, qs))
    by = {(f.code, f.where) for f in rep.findings}
    assert ("cycle", "gross → net → gross") in by
    assert ("unanswerable", "big") in by and ("unanswerable", "where") in by
    assert ("unused_part", "spare") in by
    assert ("question_as_fact", "pick") in by
    assert ("no_rule", "later") in by and rep.of("note")[0].level == "note"
    assert ("unused_part", "net") not in by                        # in a cycle: reported once, as the cycle


def test_type_conflicts():
    from pydantic import BaseModel

    class Order(BaseModel):
        amount: str
        qty: int
    cat = Catalog()

    @cat.rule("ok")
    def ok(amount: float, qty: int, note: str) -> bool:
        return amount > 0 and qty > 0

    @cat.fn
    def shout(note: int) -> int:
        return note * 2

    @cat.fn(provides="size")
    def size_a(qty: int) -> int:
        return qty

    @cat.fn(provides="size")
    def size_b(qty: int) -> list:
        return [qty]

    @cat.rule("small")
    def small(size) -> bool:
        return True
    s = System(cat, [Question("ok", "OK?"), Question("small", "Small?"), Question("loud", "Loud?", Answer.yes_no(),
                                                                                  uses=["shout"])], input_model=Order)
    rep = lint(s)
    errs = [f for f in rep.errors if f.code == "type_conflict"]
    assert len(errs) == 1 and errs[0].where == "amount" and "Order" in errs[0].message
    warn = {f.where for f in rep.warnings if f.code == "reader_types_differ"}
    assert warn == {"note", "size"}


def test_constraints_that_cannot_hold():
    cat = Catalog()

    @cat.rule("verdict")
    def verdict(x) -> Literal["safe", "unsafe"]:
        return "safe"

    @cat.rule("harm")
    def harm(x) -> Literal["none", "low", "high"]:
        return "none"

    @cat.constraint
    def unsafe_if_harm(verdict, harm):
        return harm == "none" or verdict == "unsafe"

    @cat.constraint
    def never_high(harm):
        return harm != "high"

    qs = [Question("verdict", "?"), Question("harm", "?")]
    rep = lint(System(cat, qs))
    assert rep.ok and [(f.code, f.where) for f in rep.findings] == [("dead_option", "harm")]
    assert "'high'" in rep.findings[0].message

    @cat.constraint
    def always_safe(verdict):
        return verdict == "safe"

    @cat.constraint
    def some_harm(harm):
        return harm != "none"
    rep = lint(System(cat, qs))
    assert ("constraints_conflict", "unsafe_if_harm, never_high, always_safe, some_harm") in \
        {(f.code, f.where) for f in rep.findings}

    @cat.constraint
    def impossible(verdict):
        return verdict == "maybe"

    s = System(cat, qs)

    @cat.constraint                                    # added after the System was built (System(...) refuses it)
    def ghost(verdict, mood):
        return True
    rep = lint(s)
    assert ("constraint_never_holds", "impossible") in {(f.code, f.where) for f in rep.findings}
    assert ("constraint_unknown_question", "ghost") in {(f.code, f.where) for f in rep.findings}


def test_a_raising_constraint_and_multi_label_domains():
    cat = Catalog()

    @cat.rule("tags")
    def tags(x) -> list[Literal["refund", "legal", "spam"]]:
        return []

    @cat.constraint
    def not_spam_and_legal(tags):
        return not ({"spam", "legal"} <= set(tags))

    @cat.constraint
    def first_tag(tags):
        return tags[0] != "spam"                      # IndexError on no tags
    rep = lint(System(cat, [Question("tags", "?")]))
    assert ("constraint_raises", "first_tag") in {(f.code, f.where) for f in rep.findings}
    assert rep.ok


def test_silent_defaults_in_functions_that_read_the_input():
    cat = Catalog()
    TABLE = {"a": 1}

    @cat.fn
    def limit(customer):
        return TABLE.get(customer, 100)                # a lookup in a constant table: not a default for the input

    @cat.fn
    def note_text(note):
        return note or ""

    @cat.fn
    def total(order):
        return order.get("total", 0) + (order["tax"] or 0)

    @cat.fn
    def spread(order):
        import math
        return math.sqrt(len(order)) or 1e-9           # a default on a computed value

    @cat.fn
    def ok_default(note):
        return note or None

    @cat.fn
    def accepted(customer):
        return TABLE.get(customer, 0)  # solvi: ok

    @cat.fn
    def computed(limit):
        return {"x": limit}.get("y", 5)                # reads a computed fact only: not an input parser

    @cat.rule("big")
    def big(limit, note_text, ok_default, accepted, computed, total, spread) -> bool:
        return limit > 10
    rep = lint(System(cat, [Question("big", "?")]))
    found = sorted(f.where.rsplit("(", 1)[1].rstrip(")") for f in rep.findings if f.code == "silent_default")
    assert found == ["note_text", "total", "total"]
    assert all(f.where.startswith("tests/") or "test_check.py" in f.where for f in rep.findings
               if f.code == "silent_default")
    assert silent_defaults(len) == []


def test_a_catalog_alone_is_linted_without_question_checks():
    cat = Catalog()

    @cat.fn
    def a(b):
        return b

    @cat.fn
    def b(a):
        return a
    rep = lint(cat)
    assert rep.codes() == ["cycle"]


CATALOG = '''
from solvi import Catalog, Question, System

cat = Catalog()


@cat.check(hard=True, then={"refund": "no"})
def known_customer(customer) -> bool:
    return customer != "blocked"


@cat.rule("refund")
def refund(days) -> bool:
    return days <= 30


def good():
    return System(cat, [Question("refund", "Refund?", requires=["known_customer"])])


class DropCheck:
    def plan(self, catalog, questions, init_keys, heads=None):
        from solvi.core.plan.strategist import plan
        flow = plan(catalog, questions, init_keys, heads)
        flow.per_question = {q: [f for f in fs if f != "known_customer"] for q, fs in flow.per_question.items()}
        return flow


def wired():
    return System(cat, [Question("refund", "Refund?")])      # 1.0: the `then` check is in the flow without requires


def bad():
    s = System(cat, [Question("refund", "Refund?")])
    s.strategist = DropCheck()                    # a strategist that leaves the `then` check out (System() refuses it)
    return s


def warns():
    @cat.fn
    def unused(days):
        return days or 0
    return good()
'''


def test_a_validate_that_reads_what_no_producer_reads():
    """validate gets the value and, by name, inputs of the fact's producers. One that requires anything else cannot run:
    it raised TypeError on every call and every output of its producer was rejected ("validate raised TypeError")."""
    import pytest
    from solvi import Decision

    def build(validate):
        cat = Catalog()

        @cat.fn(provides="pick", validate=validate)
        def pick_by_model(amount):
            return Decision("a", {"a": 0.9, "b": 0.1})

        @cat.fn(provides="pick")
        def pick_by_rule(amount, limit):
            return "b"

        @cat.rule("choice")
        def choice(pick) -> str:
            return pick

        return cat, [Question("choice", "Which?", Answer.choice(["a", "b"]))]

    cat, qs = build(lambda v, amount, dominance: dominance > 0.5)
    rep = lint(cat)
    assert rep.codes("error") == ["validate_reads_unknown"] and "dominance" in rep.errors[0].message
    with pytest.raises(ValueError, match="validate of pick_by_model reads \\['dominance'\\], which no producer of pick"):
        System(cat, qs)
    for ok in (lambda v, amount, limit: v == "a" and amount < limit,       # an input of another producer of the fact
               lambda v, amount, dominance=0.5: True, lambda v, **kw: True, lambda v, *a: True, lambda v: True):
        cat, qs = build(ok)
        assert lint(cat).ok
        res = System(cat, qs).ask({"amount": 5, "limit": 10})
        assert res["choice"].answer == "a"
    cat, qs = build(lambda v: True)                                         # a producer added after the System was built
    s = System(cat, qs)

    @cat.fn(provides="pick", cost=0, validate=lambda v, dominance: True)
    def pick_first(amount):
        return "a"

    from solvi.core import validated
    assert s.ask({"amount": 5, "limit": 10})["choice"].answer == "a"
    assert validated(cat.alternative("pick", "pick_first"), "a", {"amount": 5, "limit": 10}) == \
        "validate cannot run: it reads dominance, which is not an input of pick"
    with pytest.raises(ValueError, match="validate of amount_ok reads \\['limit'\\], which amount_ok does not take"):
        @cat.fn(validate=lambda v, limit: v < limit)                        # not an alternative producer: refused at once
        def amount_ok(amount):
            return amount


def test_solvi_check_exit_statuses(tmp_path, capsys):
    f = tmp_path / "refunds.py"
    f.write_text(CATALOG)
    assert main(["check", f"{f}:good"]) == 0
    assert "no problems found" in capsys.readouterr().out
    assert main(["check", f"{f}:wired"]) == 0
    assert "no problems found" in capsys.readouterr().out
    assert main(["check", f"{f}:bad"]) == 1
    assert "then_not_in_flow" in capsys.readouterr().out
    assert main(["check", f"{f}:bad", "--json"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert d["ok"] is False and d["findings"][0]["code"] == "then_not_in_flow"
    assert main(["check", f"{f}:cat"]) == 0                          # a Catalog: no question checks
    capsys.readouterr()
    f2 = tmp_path / "refunds2.py"
    f2.write_text(CATALOG)
    assert main(["check", f"{f2}:warns"]) == 0
    assert "unused_part" in capsys.readouterr().out
    assert main(["check", f"{f2}:warns", "--strict"]) == 1              # the unused part and its silent default fail now
    for bad in (["check"], ["check", "nocolon"], ["check", f"{tmp_path}/missing.py:x"]):
        try:
            code = main(bad)
        except SystemExit as e:
            code = e.code
        assert code == 2


def _approve(**kw):
    return [Question("approve", "Approve?", Answer.choice(["approve", "reject"]), **kw)]


def test_a_rule_that_returns_a_literal_outside_its_questions_options():
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount, vip):
        if vip:
            return "approve"
        if amount is None:
            return None                                      # abstains on purpose: not a finding
        label = "aprove"

        def helper():
            return "whatever"                                # a function inside the rule: not the rule's return
        return "aprove" if amount < 10 else label if amount < 20 else "reject"
    s = System(cat, _approve())
    rep = lint(s)
    assert rep.codes() == ["rule_returns_non_option"] and not rep.ok
    assert "returns 'aprove'" in rep.errors[0].message and "(approve)" in rep.errors[0].where
    assert s.ask({"amount": 5, "vip": False})["approve"].guard == "outside_options"   # the abstention it warns of

    typed = Catalog()

    @typed.rule("ok")
    def ok(amount) -> bool:
        return "yes" if amount else False
    multi = Catalog()

    @multi.rule("tags")
    def tags(amount):
        return "a" if amount else "z"
    assert lint(System(typed, [Question("ok", "OK?")])).findings == []
    rep = lint(System(multi, [Question("tags", "Tags?", Answer.multi(["a", "b"]))]))
    assert rep.codes() == ["rule_returns_non_option"] and "'z'" in rep.errors[0].message


def test_a_hard_check_without_a_bool_type_that_plainly_returns_a_non_bool():
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "reject"})
    def enough(balance):
        if balance is None:
            return                                           # rejected at run time: the question abstains
        return 0 if balance < 0 else True

    @cat.check(hard=True)
    def known(customer):
        return customer != "blocked"                         # not a literal: nothing to say

    @cat.check
    def soft(balance):
        return None                                          # a soft check: not this finding

    @cat.rule("approve")
    def approve(balance, customer, soft):
        return "approve"
    s = System(cat, _approve(requires=["enough", "known"]))
    rep = lint(s)
    assert rep.codes() == ["hard_check_untyped", "hard_check_untyped"] and not rep.ok
    assert ["returns None" in f.message or "returns 0" in f.message for f in rep.errors] == [True, True]
    r = s.ask({"balance": -5, "customer": "ann"})["approve"]
    assert (r.status, r.guard) == ("abstain", "hard_check")


def test_a_rule_for_a_question_the_system_does_not_ask():
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount):
        return "approve"

    @cat.rule("ghost")
    def ghost(amount):
        return "x"
    rep = lint(System(cat, _approve()))
    assert [(f.level, f.code, f.where) for f in rep.findings] == [("warning", "unused_rule", "ghost")]
    assert lint(cat).findings == []                           # a bare catalog has no questions to compare with


def test_names_that_nothing_computes_and_the_input_model_does_not_declare():
    from pydantic import BaseModel, ConfigDict

    class Inputs(BaseModel):
        amount: float
        balance: float

    class Closed(Inputs):
        model_config = ConfigDict(extra="forbid")

    def build(inputs, **kw):
        cat = Catalog()

        @cat.fn
        def left(balance: float, amout: float) -> float:      # a typo: `amout`
            return balance - amout

        @cat.rule("approve")
        def approve(left: float):
            return "approve" if left >= 0 else "reject"
        return System(cat, _approve(**kw) + [Question("later", "Later?", Answer.yes_no(), uses=["left", "balance", "nope"])],
                      input_model=inputs)
    rep = lint(build(Inputs))
    got = {(f.code, f.where): f.message for f in rep.warnings}
    assert set(got) == {("input_not_declared", "amout"), ("uses_unknown", "later")} and rep.ok
    assert "left reads amout" in got[("input_not_declared", "amout")] and "extra key" in got[("input_not_declared", "amout")]
    assert "'nope'" in got[("uses_unknown", "later")]
    rep = lint(build(Closed))
    assert all("forbids extra keys, so it can never be given" in f.message for f in rep.warnings) and len(rep.warnings) == 2
    assert [f.code for f in lint(build(None)).warnings] == []  # no input model: every such name is a given fact


def test_solvi_check_takes_a_task_directory_or_file_as_solvi_test_does(capsys):
    """`solvi check gallery/01_support_triage/task.py:system` exited 2 (the task has no `system`), and task.py:cat skipped
    every question check."""
    assert main(["check", "gallery/01_support_triage"]) == 0
    assert main(["check", "gallery/09_credit_adverse_action/task.py"]) == 0
    assert "no_rule" in capsys.readouterr().out                    # its questions were checked
