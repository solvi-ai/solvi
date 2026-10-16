"""A rule's `why` names the branch it took and the numbers it used (solvi.core.because), and a fact that has no value
says why (solvi.core.missing) instead of a NaN stand-in; both are recorded in the trace and checked on replay."""
from solvi import Answer, Catalog, Question, System
from solvi.core import because, missing


def auditor():
    cat = Catalog()

    @cat.fn
    def proposed(text: str):
        if "=" not in text:
            return missing("the program did not run")
        return because(float(text.split("=")[1]), "parsed the program's output", raw=text)

    @cat.fn
    def p_abstain(direct: float):
        return because(0.38, "two of three checks agree", agree=2, of=3)

    @cat.rule("verdict")
    def verdict(direct: float, proposed, p_abstain: float) -> str:
        if proposed is None:
            return because("unknown", "no program answer", direct=direct)
        if proposed == direct:
            return because("ok", "B = P", p=p_abstain)
        return because("suspect", "B ≠ P", gap=abs(proposed - direct))
    return System(cat, [Question("verdict", "Is the label right?", Answer.choice(["ok", "suspect", "unknown"]))])


def test_the_rule_branch_and_its_numbers_are_in_why():
    s = auditor()
    res = s.ask({"text": "x=192", "direct": 176.0})
    r = res["verdict"]
    assert r.answer == "suspect"
    assert r.why.startswith("because B ≠ P (gap=16.0)")
    assert "proposed = 192.0 (parsed the program's output; raw='x=192')" in r.why
    assert "p_abstain = 0.38 (two of three checks agree; agree=2, of=3)" in r.why


def test_a_missing_fact_carries_its_reason_not_nan():
    s = auditor()
    res = s.ask({"text": "no output", "direct": 176.0})
    r = res["verdict"]
    assert r.answer == "unknown"
    assert "proposed = missing (the program did not run)" in r.why
    assert "nan" not in r.why
    rec = next(x for x in res.trace.records if x.name == "proposed")
    assert rec.value is None and rec.extra == {"missing": "the program did not run"}


def test_replay_checks_the_recorded_reasons():
    s = auditor()
    res = s.ask({"text": "x=192", "direct": 176.0})
    assert res.trace.replay(s.catalog)["ok"]
    rec = next(x for x in res.trace.records if x.name == "answer:verdict")
    rec.extra["because"] = "B = P"
    assert not res.trace.replay(s.catalog)["ok"]


def test_plain_rules_keep_their_why():
    cat = Catalog()

    @cat.rule("approve")
    def approve(income: float) -> bool:
        return income >= 10000
    s = System(cat, [Question("approve", "Approve?", Answer.yes_no())])
    assert s.ask({"income": 2000.0})["approve"].why == "income = 2000.0"
