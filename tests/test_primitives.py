"""Answer primitives: "not stated" (solvi.Unknown) vs abstain, evidence quotes, spans, rankings, estimates — each a value with
a confidence — from plain rules, model-backed rules and the typed decider (a stub with the L14g contract); types → questions,
grounding, constraints and joint decoding, the audit, overall confidence, JSON round trip and replay, backward compatibility."""
import dataclasses
import json
import re
from typing import Literal

import numpy as np
import pytest

from solvi import (Answer, Catalog, Claim, Decision, Estimate, FactTypeError, Maybe, Question, Quote, Rank, Response, Span,
                   System, Unknown)
from solvi.decide import DecideModel, Item, capabilities, decode_pointer, plan_batches, pointer_evidence
from solvi.typed import Bins

DOC = "Invoice 17. Total due: 1,250.50 EUR. Paid by card on 2026-09-01. Delivery in about 12 days."


# --------------------------------------------------------------------------------------------------- types → questions
def test_types_declare_primitive_questions():
    at = Answer.from_type(Maybe[bool])
    assert at.kind == "yes_no" and at.unknown and at.primitive
    at = Answer.from_type(Maybe[Literal["a", "b"]])
    assert at.kind == "choice" and at.options == ["a", "b"] and at.unknown
    at = Answer.from_type(Span[float])
    assert at.kind == "span" and at.type is float and at.source == "doc" and not at.unknown
    assert Answer.from_type(Span[str, "notes"]).source == "notes" and Answer.from_type(Span[str]).type is None
    at = Answer.from_type(Rank[Literal["x", "y", "z"], 2])
    assert at.kind == "rank" and at.options == ["x", "y", "z"] and at.k == 2
    at = Answer.from_type(Estimate[0, 7, 14])
    assert at.kind == "estimate" and at.options == ["less than 0", "0–6", "7–13", "14 or more"] and at.coverage == 0.8
    from typing import Annotated
    at = Answer.from_type(Annotated[float, Bins([0, 0.5, 1.0], coverage=0.9, unit="kg")])
    assert at.options == ["less than 0 kg", "0 to 0.5 kg", "0.5 to 1 kg", "1 or more kg"] and at.coverage == 0.9
    assert Answer.from_type(Maybe[Estimate[1, 2]]).unknown
    assert not Answer.from_type(Literal["a", "b"]).primitive          # ordinary answer types are untouched


def test_questions_take_primitive_types_from_rules_and_check_them():
    cat = Catalog()

    @cat.rule("paid")
    def paid(doc: str) -> Maybe[bool]:
        return Unknown

    s = System(cat, [Question("paid", "Paid?")])
    assert s.questions["paid"].answer.unknown
    with pytest.raises(FactTypeError, match="does not allow it"):
        System(cat, [Question("paid", "Paid?", Answer.yes_no())])

    @cat.rule("total")
    def total(doc: str) -> Span[float]:
        return "1"

    with pytest.raises(FactTypeError, match="span"):
        System(cat, [Question("total", "Total?", Answer.choice(["1", "2"]))])


# --------------------------------------------------------------------------------------------------- not stated vs abstain
def _flags():
    cat = Catalog()

    @cat.rule("signed")
    def signed(doc) -> Maybe[bool]:
        if "signed" in doc:
            return "not signed" not in doc
        return Unknown                          # the text does not say: an answer, not an abstention

    @cat.rule("vat")
    def vat(doc) -> Maybe[Literal["standard", "reduced"]]:
        return None                             # the rule refuses: an abstention

    @cat.rule("paid")
    def paid(doc):
        return Unknown                          # not allowed for this question → outside the options

    @cat.rule("sent")
    def sent(doc) -> bool:
        return Unknown                          # and not a bool: the typed rule's output is rejected

    return cat, [Question("signed", "Signed?"), Question("vat", "VAT rate?"), Question("paid", "Paid?", Answer.yes_no()),
                 Question("sent", "Sent?")]


def test_unknown_is_an_answer_distinct_from_no_and_from_abstain():
    cat, qs = _flags()
    s = System(cat, qs)
    r = s.ask({"doc": DOC})
    a = r["signed"]
    assert a.answer is Unknown and a.not_stated and a.status == "ok" and a.confidence == 1.0 and not a.answer
    assert r["vat"].status == "abstain" and r["vat"].answer is None and r["vat"].guard == "rule_abstained"
    assert r["paid"].status == "abstain" and r["paid"].guard == "outside_options"
    assert r["sent"].status == "abstain" and "type rejected" in r["sent"].why
    assert s.ask({"doc": "The contract is not signed."})["signed"].answer == "no"
    assert r.not_stated == ["signed"] and r.overall["not_stated"] == ["signed"] and not r.complete
    assert r.overall["answered"] == 1 and set(r.overall["abstained"]) == {"vat", "paid", "sent"}
    text = str(r.audit())
    assert "signed = not stated  [ok]" in text and "not stated: signed" in text
    assert repr(Unknown) == "Unknown" and str(Unknown) == "<not stated>" and Unknown is type(Unknown)()
    import copy
    import pickle
    assert copy.deepcopy(Unknown) is Unknown and pickle.loads(pickle.dumps(Unknown)) is Unknown


def test_constraints_see_unknown_and_joint_decoding_can_choose_it():
    class Clf:
        model_id, version = "demo/clf", "1"

    cat = Catalog()

    @cat.rule("signed", model=Clf())
    def signed(doc):
        return Decision("yes", {"yes": 0.6, "no": 0.1, Unknown: 0.3})

    @cat.rule("draft")
    def draft(doc) -> bool:
        return "DRAFT" in doc

    @cat.constraint
    def drafts_are_not_signed(signed, draft):
        return draft == "no" or signed != "yes"

    @cat.constraint
    def unknown_is_seen(signed):
        return signed is Unknown or signed in ("yes", "no")

    qs = [Question("signed", "Signed?", Answer.maybe(Answer.yes_no())), Question("draft", "Draft?")]
    s = System(cat, qs)
    r = s.ask({"doc": "DRAFT contract"})
    assert r["signed"].answer is Unknown and r["signed"].repaired[0] == "yes" and r.feasible
    assert r["signed"].confidence == pytest.approx(0.3)
    assert s.ask({"doc": "final contract"})["signed"].answer == "yes"
    assert "constraint repair" in str(r.audit("signed"))


# --------------------------------------------------------------------------------------------------- evidence
def _evidence_catalog(evidence, conf=1.0):
    cat = Catalog()

    @cat.rule("paid")
    def paid(doc) -> bool:
        return Claim("Paid by card" in doc, evidence=evidence, confidence=conf)

    return cat


def test_evidence_is_located_checked_and_shown():
    cat = _evidence_catalog(["Paid by card"], 0.9)
    s = System(cat, [Question("paid", "Paid?", require_evidence=True)])
    r = s.ask({"doc": DOC})
    a = r["paid"]
    assert a.answer == "yes" and a.confidence == 0.9 and a.provenance == "computed"
    (q,) = a.evidence
    assert (q.value, q.source) == ("Paid by card", "doc") and DOC[q.start:q.end] == "Paid by card"
    au = r.audit("paid")
    assert au.evidence[0]["verified"] and not au.evidence[0]["by_model"] and au.counts["quoted"] == 1
    assert "evidence    doc[" in str(au) and "'Paid by card'  verified" in str(au)
    rec = r.trace.records[-1]
    assert rec.extra == {"evidence": [[q.start, q.end, "doc", "Paid by card"]]}
    assert r.trace.replay(cat)["ok"]


def test_evidence_not_in_the_text_is_rejected_as_grounding():
    for ev in (["Paid in cash"], [Quote("Paid by card", 0, 12)]):     # a string not in the text; wrong offsets
        cat = _evidence_catalog(ev)
        s = System(cat, [Question("paid", "Paid?")])
        r = s.ask({"doc": DOC})
        assert r["paid"].status == "abstain" and "not grounded" in r["paid"].why
        assert [e["kind"] for e in r.safeguards] == ["grounding"] and s.stats["grounding_rejected"] == 1
        assert "grounding rejected" in str(r.audit("paid")) and r.trace.replay(cat)["ok"]


def test_evidence_rejection_falls_back_to_the_next_producer():
    cat = Catalog()

    @cat.fn(provides="method")
    def method_model(doc):
        return Claim("card", evidence=["paid with a unicorn"])

    @cat.fn(provides="method")
    def method_regex(doc):
        return Claim("card", evidence=[m.group(0) for m in re.finditer(r"by card", doc)])

    @cat.rule("card")
    def card(method) -> bool:
        return method == "card"

    s = System(cat, [Question("card", "Paid by card?")])
    r = s.ask({"doc": DOC})
    assert r["card"].answer == "yes" and {e["kind"] for e in r.safeguards} == {"grounding", "fallback"}
    rec = next(x for x in r.trace.records if x.name == "method")
    assert rec.producer == "method_regex" and rec.extra["evidence"][0][3] == "by card"
    assert r.trace.replay(cat)["ok"]


def test_require_evidence_abstains_without_a_quote():
    cat = _evidence_catalog([])
    s = System(cat, [Question("paid", "Paid?", require_evidence=True)])
    r = s.ask({"doc": DOC})
    assert r["paid"].status == "abstain" and r["paid"].guard == "evidence_missing" and "would have answered 'yes'" in r["paid"].why
    assert s.stats["evidence_missing"] == 1 and "evidence missing" in s.safeguard_report()
    assert "evidence missing" not in System(cat, [Question("paid", "Paid?")]).safeguard_report()   # listed once it fires
    assert System(cat, [Question("paid", "Paid?")]).ask({"doc": DOC})["paid"].answer == "yes"


def test_replay_catches_tampered_evidence():
    cat = _evidence_catalog(["Paid by card"])
    r = System(cat, [Question("paid", "Paid?")]).ask({"doc": DOC})
    rec = r.trace.records[-1]
    rec.extra = {"evidence": [[0, 7, "doc", "Invoice"]]}
    rec.hash = __import__("solvi.runtime", fromlist=["vhash"]).vhash(rec.body())      # a consistent forgery
    bad = r.trace.replay(cat)["mismatches"]
    assert any("evidence" in why for _, _, why in bad)


# --------------------------------------------------------------------------------------------------- spans
def _spans(total_out, currency_out="EUR"):
    cat = Catalog()

    @cat.rule("total")
    def total(doc: str) -> Span[float]:
        return total_out(doc) if callable(total_out) else total_out

    @cat.rule("currency")
    def currency(doc: str) -> Span[str]:
        return currency_out

    return cat, [Question("total", "Total?"), Question("currency", "Currency?")]


def test_span_answers_are_grounded_and_typed():
    doc = DOC.replace("1,250.50", "1250.50")
    cat, qs = _spans(lambda doc: Quote("1250.50", doc.index("1250.50"), doc.index("1250.50") + 7))
    r = System(cat, qs).ask({"doc": doc})
    assert r["total"].answer == 1250.5 and r["total"].span.value == "1250.50" and r["currency"].answer == "EUR"
    sp = r["currency"].span
    assert doc[sp.start:sp.end] == "EUR" and r["currency"].kind == "span"
    assert "span        doc[" in str(r.audit("currency")) and r.trace.replay(cat)["ok"]


def test_span_failures_abstain_with_their_safeguard():
    cat, qs = _spans("EUR", "USD")               # the currency is not in the text; "EUR" is in it, and is not a float
    s = System(cat, qs)
    r = s.ask({"doc": DOC})
    assert r["currency"].status == "abstain" and r["currency"].guard == "grounding" and "not grounded" in r["currency"].why
    assert r["total"].status == "abstain" and r["total"].guard == "type_rejected" and "is not float" in r["total"].why
    assert s.stats["grounding_rejected"] == 1 and s.stats["type_rejected"] == 1
    cat2, qs2 = _spans(Quote("999", 0, 7))       # a quote whose text is not at its offsets
    assert System(cat2, qs2).ask({"doc": DOC})["total"].guard == "grounding"


# --------------------------------------------------------------------------------------------------- rankings
CH = ["email", "phone", "letter"]


def test_rank_from_a_key_function_and_from_a_list():
    cat = Catalog()
    cost = {"email": 1, "phone": 3, "letter": 5}

    @cat.rule("cheapest")
    def cheapest(doc) -> Rank[Literal["email", "phone", "letter"], 2]:
        return {o: -cost[o] for o in CH}                  # a key function's scores: higher first

    @cat.rule("listed")
    def listed(doc) -> Rank[Literal["email", "phone", "letter"]]:
        return ["letter", "email", "phone"]

    @cat.rule("broken")
    def broken(doc) -> Rank[Literal["email", "phone", "letter"]]:
        return ["fax", "email", "phone"]

    r = System(cat, [Question("cheapest", "?"), Question("listed", "?"), Question("broken", "?")]).ask({"doc": DOC})
    assert r["cheapest"].answer == ("email", "phone") and r["cheapest"].scores == {"email": -1, "phone": -3, "letter": -5}
    assert r["cheapest"].confidence == 1.0 and r["listed"].answer == ("letter", "email", "phone")
    assert r["broken"].status == "abstain" and r["broken"].guard == "outside_options"
    assert "scores      email -1.00" in str(r.audit("cheapest"))


def test_model_ranking_confidence_and_joint_decoding():
    class Ranker:
        model_id, version = "demo/ranker", "1"

    cat = Catalog()

    @cat.rule("channels", model=Ranker())
    def channels(doc):
        return Decision(("letter", "email"), {"letter": 0.5, "email": 0.3, "phone": 0.2})

    @cat.rule("paperless")
    def paperless(doc) -> bool:
        return "paperless" in doc

    @cat.constraint
    def no_letters_when_paperless(channels, paperless):
        return paperless == "no" or "letter" not in channels

    qs = [Question("channels", "Best channels?", Answer.rank(CH, k=2)), Question("paperless", "Paperless?")]
    s = System(cat, qs)
    r = s.ask({"doc": "normal"})
    assert r["channels"].answer == ("letter", "email")
    assert r["channels"].confidence == pytest.approx(0.5 * 0.3 / 0.5)          # Plackett–Luce: 0.5 · 0.3 / (1 − 0.5)
    r2 = s.ask({"doc": "paperless customer"})
    assert r2["channels"].answer == ("email", "phone") and r2["channels"].repaired[0] == ("letter", "email")
    assert r2["channels"].confidence == pytest.approx(0.3 * 0.2 / 0.7) and r2.feasible


# --------------------------------------------------------------------------------------------------- estimates
def test_estimates_from_numbers_and_distributions():
    cat = Catalog()

    @cat.rule("days")
    def days(doc) -> Estimate[0, 7, 14, 30]:
        return 12

    @cat.rule("days_dist")
    def days_dist(doc) -> Estimate[0, 7, 14, 30]:
        return {"0–6": 0.05, "7–13": 0.7, 3: 0.2, "30 or more": 0.05}     # bin labels or bin indices

    @cat.rule("days_list")
    def days_list(doc) -> Estimate[0, 7, 14, 30]:
        return [0, 0.5, 0.5, 0, 0]

    @cat.rule("days_bad")
    def days_bad(doc) -> Estimate[0, 7, 14, 30]:
        return {"next week": 1.0}

    qs = [Question(n, "?") for n in ("days", "days_dist", "days_list", "days_bad")]
    r = System(cat, qs).ask({"doc": DOC})
    assert r["days"].answer == 12 and r["days"].interval == (12, 12) and r["days"].confidence == 1.0
    d = r["days_dist"]
    assert d.answer == 10.5 and d.interval == (7, 30) and d.confidence == pytest.approx(0.9)   # bins 7–13 and 14–29
    assert d.probs["7–13"] == pytest.approx(0.7) and d.extra["coverage"] == 0.8
    assert r["days_list"].answer == 3.5 and r["days_list"].interval == (0, 14)
    assert r["days_bad"].status == "abstain" and r["days_bad"].guard == "outside_options"
    assert "10.5 (80% interval: [7, 30))" in str(r.audit("days_dist"))


def test_estimate_open_bins_and_coverage():
    from solvi.primitives import estimate_of
    at = Answer.estimate([10, 20], coverage=0.5)
    assert at.options == ["less than 10", "10–19", "20 or more"]
    v, iv, mass = estimate_of(at, [0.1, 0.3, 0.6])
    assert v == 20 and iv == [10, None] and mass == pytest.approx(0.9)       # median in the open bin: its edge
    v, iv, mass = estimate_of(at, [0.8, 0.1, 0.1])
    assert v == 10 and iv == [None, 10]
    assert Answer.estimate(lo=0, hi=1, step=0.5).options == ["less than 0", "0 to 0.5", "0.5 to 1", "1 or more"]


# --------------------------------------------------------------------------------------------------- overall, JSON, replay
def _all():
    cat = Catalog()

    @cat.rule("signed")
    def signed(doc) -> Maybe[bool]:
        return Unknown

    @cat.rule("paid")
    def paid(doc) -> bool:
        return Claim(True, evidence=["Paid by card"], confidence=0.9)

    @cat.rule("total")
    def total(doc: str) -> Span[str]:
        return "1,250.50"

    @cat.rule("channels")
    def channels(doc) -> Rank[Literal["email", "phone", "letter"], 2]:
        return {"email": 2, "phone": 3, "letter": 1}

    @cat.rule("days")
    def days(doc) -> Estimate[0, 7, 14, 30]:
        return {"7–13": 0.6, "14–29": 0.4}

    qs = [Question(n, n) for n in ("signed", "paid", "total", "channels", "days")]
    return cat, qs


def test_overall_confidence_covers_every_kind():
    cat, qs = _all()
    r = System(cat, qs).ask({"doc": DOC})
    confs = {q: x.confidence for q, x in r.results.items()}
    assert r.confidence == pytest.approx(np.prod(list(confs.values()))) and r.complete
    assert confs["signed"] == 1.0 and confs["paid"] == 0.9 and confs["days"] == pytest.approx(1.0)
    bk = r.overall["by_kind"]
    assert set(bk) == {"yes_no", "span", "rank", "estimate"} and bk["yes_no"]["answered"] == 2
    assert bk["yes_no"]["confidence"] == pytest.approx(0.9) and r.overall["not_stated"] == ["signed"]
    assert r.weakest == ("paid", 0.9)


def test_json_round_trip_and_replay_of_every_primitive():
    cat, qs = _all()
    s = System(cat, qs)
    r = s.ask({"doc": DOC})
    j = r.to_json()
    d = json.loads(j)
    assert d["results"]["signed"]["not_stated"] and d["results"]["signed"]["answer"] is None
    assert d["results"]["paid"]["evidence"][0]["text"] == "Paid by card" and d["results"]["days"]["kind"] == "estimate"
    r2 = Response.from_json(j, catalog=s)
    for q in r.results:
        a, b = r[q], r2[q]
        assert (a.answer, a.status, a.kind, a.evidence, a.extra, a.confidence) == (b.answer, b.status, b.kind, b.evidence,
                                                                                    b.extra, b.confidence), q
    assert r2["signed"].answer is Unknown and r2["channels"].answer == ("phone", "email")
    assert r2.trace.replay(cat)["ok"] and r.trace.replay(cat)["ok"]
    assert r2.overall["not_stated"] == ["signed"]
    for q in s.questions.values():                               # answer types and questions round-trip too
        assert Question.from_json(q.to_json()) == q
    assert System(cat, qs).response_schema()["$defs"]


def test_old_answer_types_and_questions_dump_as_before():
    assert Answer.choice(["a", "b"]).model_dump() == {"kind": "choice", "options": ["a", "b"], "descriptions": {}}
    assert "require_evidence" not in Question("q", "?", Answer.yes_no()).model_dump()


# --------------------------------------------------------------------------------------------------- the decider (L14g)
L14G = {"format": "solvi_decide v2", "subformat": "l14g typed v2", "multi_question": {"layout": "block", "max_questions": 6},
        "temperature": {"choice": 1.0, "score": 1.0, "noul": 1.0, "rank": 1.0, "number": 1.0}}
TEXT = "Invoice. Total 1250.50 EUR. Contact via phone, phone preferred, not email. Delivery in 9 days."


class L14gStub:
    """Keyword logits for every kind; "not stated" logit high when the text says "unclear"; a pointer over whitespace
    tokens that points at the token after the task's key word ("total" → the amount)."""
    model_id = "test/l14g-stub"

    def __init__(self):
        self.calls = []

    def fingerprint(self):
        return "l14g-stub-1"

    def _one(self, it, text):
        low = text.lower()
        toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        o = {"act": 3.0, "unknown": 4.0 if "unclear" in low else -4.0}
        if it.mode == "span":
            o["logits"] = np.zeros(0)
        elif it.mode == "number":
            m = re.search(r"(\d+) days", low)
            o["logits"] = np.array([3.0 if m and lab.startswith(str(int(m.group(1)) // 7 * 7)) else 0.0 for lab in it.options])
        elif it.mode == "noul":
            o["logits"] = np.array([2.0 if "paid" in low else -2.0])
        else:
            o["logits"] = np.array([2.0 * low.count(x.lower()) for x in it.options])
        if it.pointer:
            key = it.task.split()[-1].strip("?").lower()
            st, en = np.full(len(toks), -3.0), np.full(len(toks), -3.0)
            for i, (a, b) in enumerate(toks[:-1]):
                if text[a:b].lower().startswith(key):
                    st[i + 1] = en[i + 1] = 4.0
            null = [5.0, 5.0] if "unclear" in low else [0.0, 0.0]
            o["pointer"] = {"start": st, "end": en, "null": null, "offsets": toks}
        return o

    def logits(self, items):
        self.calls.append(("seq", [it.mode for it in items], [it.pointer for it in items]))
        return [self._one(it, it.text) for it in items]

    def logits_pass(self, passes):
        self.calls.append(("pass", [[it.mode for it in p.items] for p in passes]))
        return [[self._one(it, p.text) for it in p.items] for p in passes]


def l14g():
    return DecideModel(L14gStub(), L14G)


def test_l14g_capabilities_and_old_formats_unchanged():
    c = capabilities(L14G)
    assert c["version"] == 3 and c["unknown"]["column"] == 3 and c["pointer"]["start"] == 4 and c["pointer"]["end"] == 5
    assert c["markers"]["rank"] == "[unused5]" and c["markers"]["span"] == "[unused7]"
    assert c["multi_question"]["pointer"] is False and "number" in c["modes"]
    assert capabilities({"format": "l14g typed v2"})["version"] == 3
    old_keys = {"format", "version", "modes", "markers", "columns", "noul_labels", "serialization", "state_format", "act",
                "multi_question", "max_questions"}
    for meta in ({"format": "l14f typed v1", "multi_question": {"layout": "block"}}, {"format": "solvi_decide v2"}, {}):
        c = capabilities(meta)
        assert set(c) == old_keys and "rank" not in c["markers"] and "pointer" not in (c["multi_question"] or {})
    m = DecideModel(L14gStub(), {"format": "l14f typed v1"})
    assert set(m.temperatures) == {"single", "multi", "score", "noul", "act"} and not m.has_unknown and not m.has_pointer
    assert m.wire("rank") == "single" and m.wire("number") == "score"


def test_decider_refuses_what_the_checkpoint_cannot_give():
    m = DecideModel(L14gStub(), {"format": "l14f typed v1"})
    with pytest.raises(ValueError, match="cannot point"):
        m.decision("total", "Total?", "doc", Span[float])
    with pytest.raises(ValueError, match="cannot point"):
        m.decision("team", "Team?", "doc", Literal["a", "b"], evidence=True)
    with pytest.raises(ValueError, match="not stated"):
        m.decision("paid", "Paid?", "doc", Maybe[bool])
    r = m.decision("r", "Rank?", "doc", Rank[Literal["a", "b"]])          # rank / number: asked as a choice / a score
    assert r.spec.kind == "rank" and m._item(r.spec, "x").mode == "single"


def _decider_system():
    m = l14g()
    cat = Catalog()
    qs = [m.decision("total", "What is the total?", "doc", Span[float]).question(cat),
          m.decision("channel", "Preferred channel via", "doc", Maybe[Literal["email", "phone", "letter"]],
                     evidence=True).question(cat, require_evidence=True),
          m.decision("ranked", "Rank the channels", "doc", Rank[Literal["email", "phone", "letter"], 2]).question(cat),
          m.decision("days", "Delivery time in days?", "doc", Maybe[Estimate[0, 7, 14, 21]]).question(cat),
          m.decision("paid", "Is it paid?", "doc", Maybe[bool]).question(cat),
          m.decision("topics", "Topics?", "doc", Maybe[list[Literal["invoice", "delivery"]]]).question(cat)]
    return m, cat, System(cat, qs)


def test_decider_primitives_from_one_checkpoint():
    m, cat, s = _decider_system()
    r = s.ask({"doc": TEXT})
    t = r["total"]
    assert t.answer == 1250.5 and t.span.source == "doc" and TEXT[t.span.start:t.span.end] == "1250.50"
    assert 0.9 < t.confidence < 1 and t.provenance == "decided"
    c = r["channel"]
    assert c.answer == "phone" and c.evidence and all(TEXT[e.start:e.end] == e.value for e in c.evidence)
    assert c.probs[Unknown] < 0.01
    rk = r["ranked"]
    assert rk.answer == ("phone", "email") and set(rk.scores) == {"email", "phone", "letter"}
    assert rk.confidence == pytest.approx(rk.scores["phone"] * rk.scores["email"] / (1 - rk.scores["phone"]), rel=1e-3)
    d = r["days"]
    assert d.answer == 10.5 and d.interval[0] is not None and d.confidence > 0.5
    assert r["paid"].answer == "no" and r["topics"].answer == ("invoice", "delivery")
    au = r.audit("channel")
    assert au.evidence and au.evidence[0]["by_model"] and au.counts["quoted_by_model"] == len(c.evidence)
    rep = r.trace.replay(s)
    assert rep["ok"] and {v for *_, v in rep["models"]} == {"recomputed"}
    assert r.trace.replay(s, trust_models=True)["ok"]
    r2 = Response.from_json(r.to_json(), catalog=s)
    assert r2.trace.replay(s)["ok"] and r2["total"].answer == 1250.5 and r2["channel"].evidence == c.evidence


def test_decider_not_stated_and_the_null_span():
    m, cat, s = _decider_system()
    r = s.ask({"doc": "unclear note, Total ??? whatever"})
    for q in ("channel", "days", "paid", "topics"):
        assert r[q].answer is Unknown and r[q].status == "ok" and r[q].confidence > 0.9, q
    assert r["channel"].evidence == []                            # "not stated" needs no quote (require_evidence holds)
    # not Maybe: the best real span is "???", and the pointer itself prefers "no span" — escalated, never answered alone
    assert r["total"].status == "abstain" and r["total"].guard == "escalated" and "the text may not state it" in r["total"].why
    d = m.decision("who", "Who signed for the total?", "doc", Span[str])("unclear note, Total ??? whatever")
    assert d.value.value == "???" and d.extra["p_null"] > 0.8 and d.escalate.startswith("model escalated: the text may not")
    sure = m.decision("who", "Who signed for the total?", "doc", Span[str])("the Total 45 EUR")
    assert sure.value.value == "45" and sure.escalate is None
    assert set(r.not_stated) == {"channel", "days", "paid", "topics"} and r.trace.replay(s)["ok"]
    m2 = l14g()
    cat2 = Catalog()
    q2 = m2.decision("total", "What is the total?", "doc", Maybe[Span[float]]).question(cat2)
    r2 = System(cat2, [q2]).ask({"doc": "unclear note, Total ??? whatever"})
    assert r2["total"].answer is Unknown and r2["total"].probs[Unknown] > 0.8


def test_pointer_questions_use_the_full_layout_and_are_not_batched():
    m, cat, s = _decider_system()
    r = s.ask({"doc": TEXT})
    assert r.flow.batches and all(n not in ("answer:total", "answer:channel") for b in r.flow.batches for n in b)
    seq = [c for c in m.scorer.calls if c[0] == "seq"]
    assert seq and all(all(p) for _, _, p in seq)                  # only pointer questions go one per sequence
    assert all("span" not in modes for c in m.scorer.calls if c[0] == "pass" for modes in c[1])
    from solvi.strategist import Step
    assert plan_batches([Step(cat.rules["total"], []), Step(cat.rules["ranked"], [])]) == []


def test_decider_number_teach_and_fingerprints():
    m = l14g()
    p = m.decision("days", "Delivery time in days?", "doc", Estimate[0, 7, 14, 21])
    assert p.spec.options == ["less than 0", "0–6", "7–13", "14–20", "21 or more"] and p.options is None
    fp = p.fingerprint()
    p.teach("Delivery in 3 days.", 16)                             # a number → its bin: the shift is adapted
    assert p.fingerprint() != fp and p.adaptation.examples[0][1] == "14–20"
    plain = m.decision("team", "Team?", "doc", Literal["a", "b"])
    assert "unknown" not in plain.spec.describe() and "evidence" not in plain.spec.describe()


def test_pointer_decoding_matches_the_l14g_rules():
    text = "  paid  by card "
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    ptr = decode_pointer({"start": [3.0, 0.0, 0.0], "end": [0.0, 0.0, 3.0], "null": [0.0, 0.0], "offsets": toks}, text)
    best = ptr["spans"][0]
    assert best[1:] == (2, 15, "paid  by card") and 0 < ptr["null"] < best[0]
    assert sum(p for p, *_ in ptr["spans"]) + ptr["null"] <= 1 + 1e-9
    ev = pointer_evidence(ptr, threshold=0.01, max_spans=3)
    assert ev and all(text[q.start:q.end] == q.value for q in ev)
    assert all(a.end <= b.start for a, b in zip(ev, ev[1:]))       # non-overlapping, in text order
    assert pointer_evidence({"null": 0.9, "spans": [(0.05, 0, 4, "paid")]}) == []
    assert decode_pointer({"start": [], "end": [], "null": [0, 0], "offsets": []}, "") == {"null": 1.0, "spans": []}


def test_pointer_temperature_divides_every_score():
    text = "paid by card"
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    raw = {"start": [3.0, 0.0, 1.0], "end": [0.0, 1.0, 3.0], "null": [1.0, 0.5], "offsets": toks}
    hot = decode_pointer(raw, text, temperature=2.0)
    same = decode_pointer({"start": [1.5, 0.0, 0.5], "end": [0.0, 0.5, 1.5], "null": [0.5, 0.25], "offsets": toks}, text)
    assert hot["null"] == pytest.approx(same["null"]) and [s[1:] for s in hot["spans"]] == [s[1:] for s in same["spans"]]
    assert hot["spans"][0][0] < decode_pointer(raw, text)["spans"][0][0]      # a higher temperature: less sure


def test_typed_span_trims_the_best_span_never_jumps_outside_it():
    from solvi.decide import _typed_span
    spans = [(0.6, 10, 20, "149.90 EUR"), (0.2, 0, 3, "313"), (0.1, 10, 16, "149.90"), (0.05, 10, 13, "149")]
    k, mass = _typed_span(spans, float)
    assert k == 2 and mass == pytest.approx(0.7)                   # '149.90' and every span up to '149.90 EUR'
    assert _typed_span([(0.6, 0, 12, "twenty euros"), (0.3, 20, 23, "313")], float) == (0, None)   # → type rejected
    assert _typed_span(spans, str) == (0, None) and _typed_span([(0.9, 0, 2, "20")], float) == (0, None)
    # a piece of the number is not its value: 'EUR 18,851.12' is never trimmed to '851.12' (it was, and answered 851.12)
    money = [(0.5, 0, 13, "EUR 18,851.12"), (0.3, 7, 13, "851.12"), (0.1, 4, 6, "18"), (0.05, 11, 13, "12")]
    assert _typed_span(money, float) == (0, None)
    import datetime
    dated = [(0.7, 0, 12, "21 July 2026"), (0.2, 0, 2, "21"), (0.1, 8, 12, "2026")]
    assert _typed_span(dated, datetime.date) == (0, None) and _typed_span(dated, int) == (1, pytest.approx(0.9))


def test_a_typed_span_reads_dates_and_numbers_as_people_write_them():
    """Span[date] / Span[float] validated the quote with pydantic alone: '21 July 2026' and '41,908.56 USD' were "type
    rejected". The type's own reading comes first; then solvi.textin's parsers, which refuse rather than guess."""
    import datetime
    from decimal import Decimal
    from solvi.typed import span_value
    d = datetime.date
    for text, want in (("2026-07-21", d(2026, 7, 21)), ("21 July 2026", d(2026, 7, 21)), ("July 21, 2026", d(2026, 7, 21)),
                       ("21.07.2026", d(2026, 7, 21)), ("12 сентября 2026", d(2026, 9, 12)),
                       ("18 октября 2026 г.", d(2026, 10, 18))):
        assert span_value(d, text) == want, text
    for text, why in (("12.09.2026", "day or month first"), ("03/04/2026", "day or month first"), ("21 July", "the year is not stated"),
                      ("21/07/26", "two-digit year"), ("between 1 May 2026 and 3 May 2026", "more than one date"),
                      ("next week", "no date")):
        with pytest.raises(ValueError, match=why):
            span_value(d, text)
    for typ, text, want in ((float, "149.90", 149.9), (float, "41,908.56 USD", 41908.56), (float, "EUR 18,851.12", 18851.12),
                            (float, "1 500 000 руб", 1500000.0), (float, "1.5 million", 1500000.0), (float, "1.0000", 1.0),
                            (int, "1,200", 1200), (int, "2k", 2000), (Decimal, "€12,50", Decimal("12.5"))):
        assert span_value(typ, text) == want, text
    for typ, text in ((float, "5%"), (float, "3 100"), (float, "about 20 or 30"), (int, "12.5"), (int, "twelve"),
                      (bool, "1,500"), (str, 5)):
        with pytest.raises((ValueError, AttributeError)):
            span_value(typ, text)
    with pytest.raises(ValueError):
        span_value(float, "41,908.56 USD", strict=True)                     # strict: the type's own reading only

    def system(typ, text, at):
        cat = Catalog()

        def rule(doc):
            return Quote(text, at, at + len(text), "doc")
        rule.__annotations__ = {"return": Span[typ]}
        cat.rule("value")(rule)
        return System(cat, [Question("value", "What?")])

    doc = "Invoice dated 21 July 2026. Total amount payable: EUR 18,851.12. Delivered 03/04/2026."
    r = system(float, "EUR 18,851.12", doc.index("EUR")).ask({"doc": doc})
    assert r["value"].answer == 18851.12 and r["value"].span.value == "EUR 18,851.12"
    assert r.trace.replay(system(float, "EUR 18,851.12", doc.index("EUR")).catalog)["ok"]
    r = system(d, "21 July 2026", doc.index("21 July")).ask({"doc": doc})
    assert r["value"].answer == d(2026, 7, 21) and r["value"].span.value == "21 July 2026"
    r = system(d, "03/04/2026", doc.index("03/04")).ask({"doc": doc})
    assert r["value"].status == "abstain" and "type rejected" in r["value"].why and "day or month first" in r["value"].why


def test_act_features_always_carry_the_l14g_features():
    from solvi.decide import _Spec, act_features
    x = act_features(_Spec("?", ["yes", "no"], kind="noul"), Decision("yes", {"yes": 0.9, "no": 0.1}, confidence=0.9), 1.0)
    assert x["p_unknown"] == 0.0 and x["kind=span"] == 0.0 and x["kind=noul"] == 1.0


def test_item_keeps_its_positional_fields():
    it = Item("t", ("a",), None, "x", False, "")
    assert it.pointer is False and dataclasses.replace(it, pointer=True).pointer


def test_example_16_runs(capsys, monkeypatch):
    import runpy
    from pathlib import Path
    monkeypatch.delenv("SOLVI_DECIDE_MODEL", raising=False)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "16_primitives.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "identical after JSON: True" in out and "decider ok=True, loaded from JSON ok=True" in out
    assert "abstain [evidence_missing]" in out and "signed   not stated" in out


def test_decider_questions_from_a_pydantic_model_of_primitives():
    from pydantic import BaseModel, Field

    class Claim_(BaseModel):
        total: Span[float] = Field(description="What is the total?")
        channel: Maybe[Literal["email", "phone", "letter"]] = Field(description="Preferred channel via",
                                                                     json_schema_extra={"evidence": True})
        ranked: Rank[Literal["email", "phone", "letter"], 2] = Field(description="Rank the channels")
        days: Maybe[Estimate[0, 7, 14, 21]] = Field(description="Delivery days?")

    cat = Catalog()
    qs = l14g().questions(cat, Claim_, "doc")
    assert [(q.answer.kind, q.answer.unknown) for q in qs] == [("span", False), ("choice", True), ("rank", False),
                                                                ("estimate", True)]
    r = System(cat, qs).ask({"doc": TEXT})
    assert r["total"].answer == 1250.5 and r["channel"].evidence and r["ranked"].answer == ("phone", "email")
    assert Claim_.model_validate_json('{"total": 1, "channel": "<not stated>", "ranked": ["email"], "days": 2}').channel is Unknown


# ---------------------------------------------------------------- evidence strings are whole words and numbers
def test_an_evidence_string_is_found_as_whole_words_and_numbers_not_inside_longer_ones():
    from solvi.core import find_whole
    t = "North Beach to Chinatown: 30. Fee 3.5, total 1,300 (category A); was charged twice, 12%."
    for text in ("3", "5", "1", "300", "cat", "harged twice", "charged twi", "Chinatow", "0"):
        assert find_whole(text, t) == -1, text
    for text in ("30", "3.5", "1,300", "category", "charged twice", "charged twice,", "12", "12%", "(category A)", ": 30."):
        assert t[find_whole(text, t):].startswith(text), text
    assert find_whole("3", "30 or 3") == 6 and find_whole("", t) == -1 and find_whole("7", "7") == 0
    assert find_whole("два", "двадцать два") == 9                  # any script


def test_a_wrong_number_is_not_verified_by_a_longer_number_that_contains_it():
    """Claim(3, evidence=["3"]) over "...: 30." used to be accepted: the evidence was matched as a substring."""
    class M:
        model_id, version, deterministic = "m", "1", False

    def build(value, quote):
        cat = Catalog()

        @cat.fn(model=M())
        def minutes(doc: str) -> int:
            return Claim(value, evidence=[quote], source="doc")

        @cat.rule("short")
        def short(minutes: int) -> bool:
            return minutes < 10
        return System(cat, [Question("short", "short?", Answer.yes_no())])
    doc = {"doc": "North Beach to Chinatown: 30."}
    res = build(3, "3").ask(doc)
    assert res["short"].status == "abstain" and res["short"].answer is None
    assert [e["kind"] for e in res.safeguards] == ["grounding"]
    assert "not grounded: evidence '3' is not in doc" in res.trace.records[0].error
    ok = build(30, "30").ask(doc)
    assert (ok["short"].answer, ok["short"].status) == ("no", "ok")
    assert ok.trace.records[0].extra["evidence"] == [[26, 28, "doc", "30"]] and ok.trace.replay(build(30, "30"))["ok"]
    by_offsets = build(3, Quote("3", 26, 27)).ask(doc)             # a Quote's offsets inside the number 30: not "3"
    assert (by_offsets["short"].status, by_offsets["short"].answer) == ("abstain", None)
    assert "not grounded: evidence '3' is not the text at doc[26:27] ('30')" in by_offsets.trace.records[0].error
    assert build(30, Quote("30", 26, 28)).ask(doc)["short"].answer == "no"


def test_a_model_quote_with_offsets_inside_a_number_is_rejected():
    from solvi.core import cuts_number
    t = "Fee 3.5, total 1,300 and 30 or 3; Chinatown"
    assert cuts_number(t, 4, 5) and cuts_number(t, 6, 7) and cuts_number(t, 17, 20) and cuts_number(t, 25, 26)
    assert not cuts_number(t, 4, 7) and not cuts_number(t, 15, 20) and not cuts_number(t, 25, 27) \
        and not cuts_number(t, 31, 32) and not cuts_number(t, 34, 39)                # a word may be cut: only numbers

    class M:
        model_id, version, deterministic = "m", "1", False
    cat = Catalog()

    @cat.extract(model=M())
    def minutes(doc: str):
        return Quote(3, 26, 27)

    @cat.rule("short")
    def short(minutes) -> bool:
        return minutes < 10
    res = System(cat, [Question("short", "short?", Answer.yes_no())]).ask({"doc": "North Beach to Chinatown: 30."})
    assert res["short"].status == "abstain" and "is not the text at [26:27] ('30')" in res.trace.records[0].error


def test_a_typed_span_does_not_read_an_ambiguous_number():
    """Span[float] read "2.500" as 2.5 while the guarded parser, and the docstring, refuse it as ambiguous."""
    from decimal import Decimal
    from solvi.typed import span_value
    for t in ("2.500", "1.000", "12.345"):
        for vtype in (float, Decimal):
            with pytest.raises(ValueError, match="ambiguous"):
                span_value(vtype, t)
    assert span_value(float, "2.50") == 2.5 and span_value(float, "1.000,50") == 1000.5 and span_value(int, "1,000") == 1000
    assert span_value(float, "2.5000") == 2.5 and span_value(float, "1e3") == 1000.0 and span_value(float, "2.500", strict=True) == 2.5
