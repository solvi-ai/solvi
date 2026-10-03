"""Grounded decisions: provenance of every fact and answer, model identity in the trace, grounding of model outputs, the
audit, and the lifetime safeguard stats. Fuzzy proposes, deterministic decides, everything is in the trace."""
import io
import random
import re
import runpy
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from solvi import Answer, Catalog, Decision, Question, Quote, System
from solvi.core.provenance import digest, fingerprint, matches
from solvi.core.runtime import MISSING, vhash

DOC = "Invoice 7781\nVendor: Acme Tools GmbH\nTotal: 1,250.00 EUR\nDue: 2026-10-30\n"
OLD_KEYS = {"step", "kind", "name", "inputs", "value", "quote", "error", "prev"}


class FakeExtractor:
    """A stand-in for a model extractor: a pattern, a version (its fingerprint) and a switch to hallucinate."""
    model_id = "test/fake-extractor"

    def __init__(self, version="1", hallucinate=None):
        self.version, self.hallucinate = version, hallucinate

    def fingerprint(self):
        return digest("FakeExtractor", self.version)

    def field(self, name, pattern, convert=None):
        def f(doc):
            m = re.search(pattern, doc)
            if not m:
                return None
            v = self.hallucinate if self.hallucinate is not None else m.group(1)
            return Quote(convert(v) if convert else v, m.start(1), m.end(1), confidence=0.9)
        f.__name__ = name
        f.__solvi_model__ = self
        f.__solvi_provenance__ = "quoted"
        return f


def model_catalog(ex, fallback=False):
    cat = Catalog()
    if fallback:
        cat.extract(ex.field("vendor_model", r"Vendor:\s*(.+)"), provides="vendor")

        @cat.extract(provides="vendor")
        def vendor_regex(doc):
            m = re.search(r"Vendor:\s*(.+)", doc)
            return Quote(m.group(1), m.start(1), m.end(1))
    else:
        cat.extract(ex.field("vendor", r"Vendor:\s*(.+)"))

    @cat.fn
    def known_vendor(vendor, approved_vendors):
        return vendor in approved_vendors

    @cat.rule("pay")
    def pay(known_vendor):
        return known_vendor
    return cat


QS = [Question("pay", "Pay the invoice?", Answer.yes_no())]
STATE = {"doc": DOC, "approved_vendors": ["Acme Tools GmbH"]}


# ---------------------------------------------------------------- 1. provenance
def test_plain_catalog_records_hash_as_before_and_provenance_is_default():
    cat = Catalog()

    @cat.extract
    def total(doc):
        m = re.search(r"Total:\s*([0-9,.]+)", doc)
        return Quote(float(m.group(1).replace(",", "")), m.start(1), m.end(1))

    @cat.fn
    def big(total):
        return total > 1000

    @cat.rule("pay")
    def pay(big):
        return not big

    res = System(cat, QS).ask({"doc": DOC})
    for r in res.trace.records:
        assert set(r.body()) == OLD_KEYS                       # nothing new is hashed for plain parts
        assert r.model is None and r.probs is None
    origins = {r.name: r.origin for r in res.trace.records}
    assert origins == {"total": "quoted", "big": "computed", "answer:pay": "computed"}
    assert res["pay"].provenance == "computed" and res["pay"].source == "pay"
    assert "given" in str(res.audit()) and res.audit("pay").share_deterministic == 1.0


def test_model_backed_parts_record_model_identity_and_provenance():
    ex = FakeExtractor()
    cat = model_catalog(ex)
    assert cat.parts["vendor"].model is ex                     # the field() function brought its model along
    res = System(cat, QS).ask(STATE)
    rec = next(r for r in res.trace.records if r.name == "vendor")
    assert rec.origin == "quoted"
    assert rec.model == {"type": "FakeExtractor", "id": "test/fake-extractor", "fp": ex.fingerprint()}
    assert "model" in rec.body()                               # model-backed records hash their model identity
    assert "model FakeExtractor test/fake-extractor" in res.state_text()
    assert res["pay"].answer == "yes"


def test_declared_provenance_and_validation():
    cat = Catalog()

    @cat.fn(provenance="proposed")
    def plan(x):
        return x + 1

    @cat.rule("q")
    def q(plan):
        return plan > 1
    res = System(cat, [Question("q", "", Answer.yes_no())]).ask({"x": 1})
    rec = res.trace.records[0]
    assert rec.origin == "proposed" and rec.body()["provenance"] == "proposed"
    assert res.audit("q").fuzzy == 1
    with pytest.raises(ValueError):
        Catalog().fn(provenance="given")(lambda x: x)
    with pytest.raises(ValueError):
        Catalog().fn(provenance="magic")(lambda x: x)


def test_decision_with_probabilities_and_closed_set():
    class Clf:
        model_id, version = "test/clf", "3"

    for rogue, ok in ((False, True), (True, False)):
        cat = Catalog()

        @cat.fn(model=Clf(), options=["low", "high"])
        def risk(amount):
            return Decision("extreme" if rogue else ("high" if amount > 100 else "low"), {"low": 0.2, "high": 0.8})  # noqa: B023

        @cat.rule("risk_q")
        def risk_q(risk):
            return risk
        s = System(cat, [Question("risk_q", "", Answer.choice(["low", "high"]))])
        res = s.ask({"amount": 500})
        rec = res.trace.records[0]
        assert rec.origin == "decided" and rec.probs == {"low": 0.2, "high": 0.8}
        if ok:
            assert res["risk_q"].answer == "high" and abs(res["risk_q"].confidence - 0.8) < 1e-9
            assert s.stats["outside_options"] == 0
        else:
            assert rec.value is MISSING and "outside the options" in rec.error
            assert res["risk_q"].status == "abstain"
            assert s.stats["outside_options"] == 1
        assert res.trace.replay(cat)["ok"]


# ---------------------------------------------------------------- 3. grounding
def test_hallucinating_extractor_is_caught_and_the_question_abstains():
    ex = FakeExtractor(hallucinate="Acme Tools GmbH & Partners")   # not what the document says at those offsets
    cat = model_catalog(ex)
    s = System(cat, QS)
    res = s.ask(STATE)
    rec = next(r for r in res.trace.records if r.name == "vendor")
    assert rec.value is MISSING and rec.error.startswith("not grounded")
    assert "Acme Tools GmbH & Partners" in rec.error                # the claim stays visible
    assert res["pay"].status == "abstain"                           # never silently an answer
    assert s.stats["grounding_rejected"] == 1 and s.stats["model_outputs"] == 1
    assert res.trace.replay(cat)["ok"]                              # replay re-runs the model and agrees it is rejected
    au = res.audit("pay")
    assert au.safeguards[0]["kind"] == "grounding" and "REJECTED" in str(au)


def test_hallucination_falls_back_to_the_next_producer():
    ex = FakeExtractor(hallucinate="Globex Corporation")
    cat = model_catalog(ex, fallback=True)
    s = System(cat, QS)
    res = s.ask(STATE)
    rec = next(r for r in res.trace.records if r.name == "vendor")
    assert rec.producer == "vendor_regex" and rec.tried[0][1].startswith("not grounded")
    assert rec.model is None and res["pay"].answer == "yes"
    assert s.stats["grounding_rejected"] == 1 and s.stats["fallbacks"] == 1
    kinds = [e["kind"] for e in res.audit("pay").safeguards]
    assert kinds == ["grounding", "fallback"]
    assert res.trace.replay(cat)["ok"]


def test_numbers_are_grounded_as_written_and_other_types_are_not_compared():
    assert matches("Acme  Tools\nGmbH", "Acme Tools GmbH") and not matches("Acme", "Acme Tools")
    assert matches(1250.0, "1,250.00") and matches(1250, "EUR 1 250") and not matches(1250.5, "1,250.00")
    assert matches(True, "yes") is None
    ex = FakeExtractor(hallucinate="9,999.00")
    for halluc, grounded in ((None, True), ("9,999.00", False)):
        ex.hallucinate = halluc
        cat = Catalog()
        cat.extract(ex.field("total", r"Total:\s*([0-9,.]+)", lambda s: float(s.replace(",", ""))))

        @cat.rule("pay")
        def pay(total):
            return total < 5000
        res = System(cat, QS).ask({"doc": DOC})
        assert (res["pay"].status == "ok") is grounded


def test_hand_written_quotes_may_derive_a_value_unless_exact():
    for exact in (None, True):
        cat = Catalog()

        @cat.extract(exact=exact)
        def currency(doc):
            i = doc.index("EUR")
            return Quote("euro", i, i + 3)                          # a label for the quoted text, by plain code

        @cat.rule("pay")
        def pay(currency):
            return currency == "euro"
        res = System(cat, QS).ask({"doc": DOC})
        if exact:
            assert res["pay"].status == "abstain"
        else:
            assert res["pay"].answer == "yes"
            q = res.audit("pay").quoted[0]
            assert q["match"] is False and "derived from 'EUR'" in str(res.audit("pay"))


def test_quote_outside_the_text_is_not_used_downstream():
    cat = Catalog()

    @cat.extract
    def total(doc):
        return Quote(5.0, 10, 10_000)

    @cat.rule("pay")
    def pay(total):
        return total < 10
    s = System(cat, QS)
    res = s.ask({"doc": DOC})
    assert res.trace.records[0].value is MISSING and res["pay"].status == "abstain"
    assert s.stats["grounding_rejected"] == 1
    assert res.trace.replay(cat)["ok"]


# ---------------------------------------------------------------- 2. model identity and replay
def test_replay_reports_a_changed_model_and_verifies_trusted_outputs():
    ex = FakeExtractor(version="1")
    cat = model_catalog(ex)
    res = System(cat, QS).ask(STATE)
    rep = res.trace.replay(cat)
    assert rep["ok"] and rep["models"] == [(1, "vendor", "recomputed")]
    assert res.trace.replay(cat, trust_models=True)["models"] == [(1, "vendor", "trusted")]

    ex.version = "2"                                                 # retrained after the decision
    rep = res.trace.replay(cat)
    assert not rep["ok"] and "model changed since this decision" in rep["mismatches"][0][2]
    assert rep["models"] == [(1, "vendor", "changed")]

    ex.version = "1"
    cat.parts["vendor"].model = None                                 # the model is not available to this replay
    rep = res.trace.replay(cat)
    assert rep["ok"] and rep["models"] == [(1, "vendor", "unavailable")]
    cat.parts["vendor"].model = ex


def test_trusted_replay_still_catches_an_ungrounded_recorded_value():
    ex = FakeExtractor()
    cat = model_catalog(ex)
    res = System(cat, QS).ask(STATE)
    tr = res.trace
    r0 = tr.records[0]
    r0.value = "Initech"                                             # consistent tampering: the chain is rebuilt
    prev = tr.init_hash
    for r in tr.records:
        r.prev = prev
        r.hash = vhash(r.body())
        prev = r.hash
    rep = tr.replay(cat, trust_models=True)                          # the model is not re-run, the quote is checked
    assert not rep["ok"] and any("not grounded" in m[2] for m in rep["mismatches"])


def test_answer_head_decisions_are_in_the_trace_with_their_fingerprint():
    cat = Catalog()

    @cat.fn
    def amount(x):
        return x

    q = Question("big", "", Answer.yes_no())
    s = System(cat, [q])
    rng = random.Random(0)
    xs = [rng.uniform(0, 100) for _ in range(60)]
    head = s.fit("big", [({"x": x}, x > 50) for x in xs], select=False)
    res = s.ask({"x": 80})
    hr = res.trace.records[-1]
    assert hr.kind == "head" and hr.name == "answer:big" and hr.origin == "learned"
    assert hr.model["type"] == "FastHead" and hr.model["fp"] == fingerprint(head)
    assert res["big"].provenance == "learned" and res["big"].source == "FastHead"
    assert res.trace.replay(s)["ok"] and res.trace.replay(s)["models"][-1][2] == "recomputed"
    assert res.trace.replay(cat)["models"][-1][2] == "unavailable"   # the catalog alone has no heads
    s.teach("big", {"x": 49}, "yes")                                # the head changes
    rep = res.trace.replay(s)
    assert not rep["ok"] and "model changed" in rep["mismatches"][0][2]
    au = res.audit("big")
    assert au.learned and au.share_deterministic < 1


def test_learned_rule_records_its_rule_list():
    cat = Catalog()

    @cat.fn
    def word(text):
        return text.split()[0]

    s = System(cat, [Question("zone", "", Answer.choice(["a", "b"]))])
    rl = s.learn_rule("zone", [({"text": t}, z) for t, z in [("north x", "a")] * 5 + [("south y", "b")] * 5], features=["word"])
    res = s.ask({"text": "north z"})
    rr = res.trace.records[-1]
    assert rr.origin == "learned" and rr.model["type"] == "RuleList" and rr.model["fp"] == fingerprint(rl)
    assert res["zone"].provenance == "learned"
    assert res.trace.replay(cat)["ok"]


# ---------------------------------------------------------------- 4-5. audit and stats
def test_audit_of_a_no_model_decision_vs_a_model_decision():
    plain = Catalog()

    @plain.extract
    def vendor(doc):
        m = re.search(r"Vendor:\s*(.+)", doc)
        return Quote(m.group(1), m.start(1), m.end(1))

    @plain.fn
    def known_vendor(vendor, approved_vendors):
        return vendor in approved_vendors

    @plain.rule("pay")
    def pay(known_vendor):
        return known_vendor

    a = System(plain, QS).ask(STATE).audit()
    b = System(model_catalog(FakeExtractor()), QS).ask(STATE).audit()
    assert a["pay"].answer == b["pay"].answer == "yes"                      # the same decision...
    assert a["pay"].share_deterministic == 1.0 and a["pay"].fuzzy == 0       # ...one fully deterministic
    assert b["pay"].counts["quoted_by_model"] == 1 and b["pay"].share_deterministic < 1
    ta, tb = str(a), str(b)
    assert "doc[" in ta and "literal 'Acme Tools GmbH'" in ta and "FakeExtractor" not in ta
    assert "[FakeExtractor test/fake-extractor #" in tb
    assert "100% deterministic" in ta and "from models" in tb
    d = b["pay"].to_dict()
    assert d["quoted"][0]["model"].startswith("FakeExtractor") and d["share_deterministic"] < 1


def test_stats_count_hard_checks_constraints_and_low_confidence():
    cat = Catalog()

    @cat.fn
    def score(x):
        return x

    @cat.check(hard=True, then={"a": "no"})
    def positive(score):
        return score > 0

    @cat.rule("a")
    def a(score):
        return score > 5

    @cat.constraint
    def b_needs_a(a, b):
        return b == "no" or a == "yes"

    q_b = Question("b", "", Answer.yes_no(), min_confidence=0.55)
    s = System(cat, [Question("a", "", Answer.yes_no(), requires=["positive"]), q_b])
    rng = random.Random(1)
    s.fit("b", [({"x": v}, "yes" if v > 3 else "no") for v in [rng.uniform(-5, 10) for _ in range(80)]], select=False)
    r1 = s.ask({"x": -1})                                            # hard check forces a = no
    assert r1["a"].status == "forced" and r1["a"].guard == "hard_check"
    assert s.stats["forced_by_hard_check"] == 1
    r2 = s.ask({"x": 4})                                             # a = no (rule); b learned yes → repaired to no
    assert r2["b"].repaired == ("yes", ["b_needs_a"])                # joint decoding changed it (then it was too unsure)
    assert s.stats["constraint_repairs"] == 1
    s.questions["b"].min_confidence = 1.01                           # nothing is that sure
    r3 = s.ask({"x": 9})
    assert r3["b"].status == "abstain" and r3["b"].guard == "low_confidence"
    assert "would have answered" in r3["b"].why
    assert s.stats["low_confidence"] >= 1 and s.stats["asks"] == 3
    assert "grounding rejected" in s.safeguard_summary()


def test_show_prints_the_compact_audit():
    from solvi.show import show
    res = System(model_catalog(FakeExtractor()), QS).ask(STATE)
    out = io.StringIO()
    with redirect_stdout(out):
        show(res, res.catalog)
    text = out.getvalue()
    assert "── audit" in text and "pay:" in text and "quoted by model" in text and "model steps: vendor recomputed" in text


def test_example_12_runs():
    out = io.StringIO()
    with redirect_stdout(out):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "12_grounded_audit.py"), run_name="__main__")
    text = out.getvalue()
    assert "not grounded: '488.60'" in text and "total_regex used after total_model rejected" in text
    assert "outside the options" in text and "model changed since this decision" in text
    assert "100% deterministic" in text and "low confidence 0.40 < 0.5" in text


def test_a_model_rule_whose_quote_is_not_in_the_text_abstains_with_guard_grounding():
    class M:
        model_id, version = "demo/m", "1"
    cat = Catalog()

    @cat.rule("team", model=M())
    def team(doc):
        return Decision("billing", {"billing": 0.9, "shipping": 0.1}, evidence=[Quote("charged twice", 500, 513, "doc")])

    @cat.rule("where", model=M())
    def where(doc):
        return Quote("nothing here", 0, 12, "doc")
    s = System(cat, [Question("team", "Team?", Answer.choice(["billing", "shipping"])),
                     Question("where", "Where?", Answer.span(source="doc"))])
    r = s.ask({"doc": "I was charged twice"})
    for q in ("team", "where"):
        assert r[q].status == "abstain" and r[q].guard == "grounding" and r[q].answer is None
    assert [e["kind"] for e in r.safeguards] == ["grounding", "grounding"]     # one event per answer, not two
    assert s.stats["grounding_rejected"] == 2


def test_a_models_quote_of_a_decimal_a_numpy_number_or_a_date_is_compared_with_the_text():
    """Quote(Decimal("999.99"), 0, 5) over "hello" was accepted from a model-backed part: only str, int and float values
    were compared with the text at the offsets."""
    import datetime
    import fractions
    from decimal import Decimal
    import numpy as np
    from solvi.core.provenance import matches
    assert matches(Decimal("999.99"), "hello") is False and matches(Decimal("12.50"), "total 12.50") is True
    assert matches(Decimal("1250.5"), "1,250.50 EUR") is True and matches(Decimal("12.5"), "12.51") is False
    assert matches(fractions.Fraction(1, 2), "0.5") is True and matches(np.float64(12.5), "12.50") is True
    assert matches(np.int64(7), "8 items") is False and matches(Decimal("NaN"), "12") is False
    assert matches(datetime.date(2020, 1, 1), "hello") is False and matches(datetime.date(2020, 1, 1), "on 2020-01-01") is True
    assert matches(datetime.date(2026, 9, 12), "12 September 2026") is True
    assert matches(datetime.date(2026, 9, 13), "12 September 2026") is False
    assert matches(datetime.date(2026, 9, 12), "12 Sep") is None and matches(True, "yes") is None   # cannot be compared

    class M:
        version = "1"
    cat = Catalog()

    @cat.extract(model=M())
    def total(doc):
        return Quote(Decimal("999.99"), 0, 5)

    @cat.rule("big")
    def big(total):
        return total > 100
    res = System(cat, [Question("big", "?", Answer.yes_no())]).ask({"doc": "hello world, total 12.50"})
    assert res["big"].status == "abstain" and "not grounded" in res.trace.records[0].error
