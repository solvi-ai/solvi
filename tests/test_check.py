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
    q = Question("refund", "Refund?", checkpoints=["known_customer"] if checkpoint else [])
    return System(cat, [q])


def test_a_clean_catalog_has_no_findings():
    rep = lint(refund_catalog())
    assert rep.ok and rep.findings == [], str(rep)


def test_a_hard_check_with_then_outside_its_questions_flow():
    s = refund_catalog(checkpoint=False)
    rep = lint(s)
    assert rep.codes("error") == ["then_not_in_flow"] and not rep.ok
    assert "checkpoints=['known_customer']" in rep.errors[0].message
    assert s.ask({"purchase_day": 1, "today": 5, "customer": "blocked"})["refund"].answer == "yes"   # the bug it catches


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
                                                                                  uses=["shout"])], inputs=Order)
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

    @cat.constraint
    def ghost(verdict, mood):
        return True
    rep = lint(System(cat, qs))
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
        return TABLE.get(customer, 100)

    @cat.fn
    def note_text(note):
        return note or ""

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
    def big(limit, note_text, ok_default, accepted, computed) -> bool:
        return limit > 10
    rep = lint(System(cat, [Question("big", "?")]))
    found = sorted(f.where.rsplit("(", 1)[1].rstrip(")") for f in rep.findings if f.code == "silent_default")
    assert found == ["limit", "note_text"]
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
    return System(cat, [Question("refund", "Refund?", checkpoints=["known_customer"])])


def bad():
    return System(cat, [Question("refund", "Refund?")])


def warns():
    @cat.fn
    def unused(days):
        return days or 0
    return good()
'''


def test_solvi_check_exit_statuses(tmp_path, capsys):
    f = tmp_path / "refunds.py"
    f.write_text(CATALOG)
    assert main(["check", f"{f}:good"]) == 0
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
