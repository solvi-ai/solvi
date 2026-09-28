"""solvi.longdoc and decisions with long="retrieve": a long synthetic contract is split into sections, BM25 (and optionally
the decider's own relevance) selects the ones that bear on the question, the decider reads only those, spans map back into
the whole contract, and the sections read are in the trace. Stand-in scorers as in test_primitives."""
from typing import Literal

import numpy as np
import pytest

from solvi import Catalog, Span, System
from solvi.decide import DecideModel
from solvi.longdoc import BM25, LongDocument, approx_tokens, terms
from test_primitives import L14G, L14gStub

FILLER = ("The parties shall cooperate in good faith and keep accurate records of every order, shipment and invoice "
          "exchanged under this Agreement, and each party shall bear its own costs unless stated otherwise. ")
CLAUSES = [
    ("DEFINITIONS", "In this Agreement the Supplier means Nordwerk GmbH, a company registered in Germany, and the Buyer "
                    "means the purchasing company named in the order form."),
    ("SCOPE", "The Supplier sells and the Buyer buys the products listed in Schedule 1."),
    ("DELIVERY", "Products are delivered to the Buyer's warehouses in Germany within 14 days of the order, and delivery "
                 "costs to Germany are borne by the Supplier."),
    ("PRICES", "Prices are fixed for the first year and may change once a year with 30 days' warning."),
    ("PAYMENT", "The Buyer pays every invoice within 45 days of receipt by bank transfer."),
    ("WARRANTY", "The Supplier warrants the products against defects in materials for 24 months."),
    ("LIABILITY", "Neither party is liable for indirect or consequential loss; the total liability is capped at the fees "
                  "paid in the preceding twelve months."),
    ("CONFIDENTIALITY", "Each party keeps the other's confidential information secret for five years."),
    ("TERMINATION", "Either party may terminate this Agreement for convenience on notice 90 days in writing to the other "
                    "party, and at once for a material breach that is not remedied."),
    ("FORCE MAJEURE", "A party is excused for delays caused by events beyond its reasonable control."),
    ("ASSIGNMENT", "Neither party may assign this Agreement without the other's written consent."),
    ("NOTICES", "All notices shall be sent to the addresses in the order form, by registered mail or e-mail."),
    ("GOVERNING LAW", "This Agreement is governed by the law of France, and the courts of Paris, France have exclusive "
                      "jurisdiction."),
    ("ENTIRE AGREEMENT", "This Agreement is the entire agreement between the parties on its subject."),
]


def contract(filler=4):
    out = ["SUPPLY AGREEMENT", ""]
    for i, (h, body) in enumerate(CLAUSES, 1):
        out += [f"{i}. {h}", "", body + " " + FILLER * filler, ""]
    return "\n".join(out)


TEXT = contract()


class ContractStub(L14gStub):
    """L14gStub (choice by option counts, a pointer at the token after the task's last word) with a yes / no relevance:
    yes when the passage says "governed"."""
    model_id = "test/contract-stub"

    def _one(self, it, text):
        self.__dict__.setdefault("texts", []).append(text)
        o = super()._one(it, text)
        if it.mode == "noul":
            o["logits"] = np.array([3.0 if "governed" in text.lower() else -3.0])
        return o


def model():
    return DecideModel(ContractStub(), {**L14G, "max_len": 256})


# --------------------------------------------------------------------------------------------------- the document
def test_sections_cover_the_text_fit_and_keep_headings():
    assert approx_tokens(TEXT) > 2000
    doc = LongDocument(TEXT, max_tokens=120)
    assert len(doc) > len(CLAUSES)
    heads = {s.heading for s in doc.sections}
    assert "13. GOVERNING LAW" in heads and "9. TERMINATION" in heads
    prev = 0
    for s in doc.sections:
        assert approx_tokens(doc.section_text(s)) <= 120 and s.start >= prev
        assert TEXT[prev:s.start].strip() == ""                 # only whitespace between sections
        prev = s.end
    assert TEXT[prev:].strip() == ""
    one = LongDocument("no headings here. " * 400, max_tokens=50)
    assert len(one) > 10 and all(approx_tokens(one.section_text(s)) <= 50 for s in one.sections)


def test_bm25_and_selection():
    bm = BM25([terms("the law of France governs"), terms("delivery to Germany"), terms("payment in 45 days")])
    sc = bm.score(terms("which law governs"))
    assert sc[0] > 0 and sc[1] == 0 == sc[2]
    doc = LongDocument(TEXT, max_tokens=120)
    top = doc.select("Which law governs this agreement?", k=2)
    assert top[0][0].heading == "13. GOVERNING LAW" and len(top) == 2 and top[0][1] >= top[1][1]
    win = doc.window([s for s, _ in top])
    i = win.text.index("France")
    a, b = win.to_doc(i, i + 6)
    assert TEXT[a:b] == "France"
    assert win.to_doc(0, len(win.text)) is None               # a range across two sections does not map back
    rr = doc.select("law", k=1, rerank=lambda xs: [1.0 if "Paris" in x else 0.0 for x in xs])
    assert "Paris" in doc.section_text(rr[0][0])
    tight = doc.select("law agreement", k=5, budget=150)
    assert sum(approx_tokens(doc.section_text(s)) for s, _ in tight) <= 150 or len(tight) == 1


# --------------------------------------------------------------------------------------------------- decisions
LAW = Literal["England", "France", "Germany"]


def test_retrieve_finds_the_clause_truncation_would_miss():
    m = model()
    full = m.decision("law", "Which country's law governs this agreement?", "contract", LAW)
    assert full(contract=TEXT).value == "Germany"             # the whole text: Germany is named more often
    part = m.decision("law", "Which country's law governs this agreement?", "contract", LAW, long="retrieve", top_k=1)
    d = part(contract=TEXT)
    assert d.value == "France"
    lg = d.extra["long"]
    assert lg["read"] == 1 and lg["of"] > 14 and lg["by"] == "bm25" and lg["sections"][0][2] == "13. GOVERNING LAW"
    a, b = lg["sections"][0][:2]
    assert "governed by the law of France" in TEXT[a:b]
    assert max(len(t) for t in m.scorer.texts[1:]) < len(TEXT) / 5    # after the full-text call: only the window
    assert part.budget() < 256 and part.fingerprint() != full.fingerprint()
    ev = m.decision("law", "Which country's law is this agreement governed by", "contract", LAW, long="retrieve",
                    top_k=1, evidence=True)(contract=TEXT)
    assert ev.evidence and all(TEXT[e.start:e.end] == e.value and e.source == "contract" for e in ev.evidence)
    assert all(a <= e.start and e.end <= b for e in ev.evidence)          # inside the section that was read
    short = "This Agreement is governed by the law of England."
    assert "long" not in part(contract=short).extra and part(contract=short).value == "England"


def test_span_maps_back_into_the_whole_contract_and_the_trace_records_the_sections():
    m = model()
    cat = Catalog()
    q = m.decision("notice", "Termination for convenience needs notice", "contract", Span[str], long="retrieve",
                   top_k=2).question(cat)
    s = System(cat, [q])
    res = s.ask({"contract": TEXT})
    r = res["notice"]
    assert r.answer == "90" and TEXT[r.span.start:r.span.end] == "90" and r.span.source == "contract"
    assert r.span.start > TEXT.index("9. TERMINATION")
    rec = next(x for x in res.trace.records if x.name == "answer:notice")
    lg = rec.extra["long"]
    assert lg["read"] == 2 and any(h == "9. TERMINATION" for _, _, h, _ in lg["sections"])
    assert "read 2 of" in str(res.audit("notice"))
    rep = res.trace.replay(s)
    assert rep["ok"], rep["mismatches"]
    import dataclasses
    from solvi.runtime import vhash
    forged = dataclasses.replace(rec, extra={**rec.extra, "long": {**lg, "sections": lg["sections"][:1]}})
    forged.hash = vhash(forged.body())                       # a consistent record that claims other sections were read
    res.trace.records[res.trace.records.index(rec)] = forged
    assert any("long" in why for *_, why in res.trace.replay(s)["mismatches"])


def test_rerank_with_the_deciders_relevance():
    m = model()
    part = m.decision("law", "Which country's law applies?", "contract", LAW, long="retrieve", top_k=1, rerank=True)
    d = part(contract=TEXT)
    assert d.value == "France" and d.extra["long"]["by"] == "bm25+decider"
    assert any("noul" in str(c[1]) for c in m.scorer.calls)          # one yes / no relevance question per candidate


def test_long_parts_in_a_shared_pass_and_bad_option():
    m = model()
    cat = Catalog()
    q1 = m.decision("law", "Which country's law governs this agreement?", "contract", LAW, long="retrieve",
                    top_k=1).question(cat)
    q2 = m.decision("paid", "Is it paid?", "contract", bool).question(cat)
    s = System(cat, [q1, q2])
    res = s.ask({"contract": TEXT})
    assert res["law"].answer == "France" and res.trace.replay(s)["ok"]
    with pytest.raises(ValueError, match="long"):
        m.decision("x", "t", "contract", LAW, long="summarize")


# --- fixes before 0.7
def test_short_lowercase_lines_are_not_headings():
    from solvi.longdoc import LongDocument
    text = "ARTICLE 1. DEFINITIONS\n\nthe goods are delivered\nshort line here\n\nPAYMENT TERMS\n\nthe buyer pays\n"
    doc = LongDocument(text, max_tokens=8)
    heads = {s.heading for s in doc.sections}
    assert heads == {"ARTICLE 1. DEFINITIONS", "PAYMENT TERMS"}


def test_splitting_a_megabyte_is_fast():
    import time

    from solvi.longdoc import LongDocument
    para = ("The supplier shall deliver the goods within thirty days of the order and\ninvoice the buyer at the agreed "
            "price.\nshort line here\n\n")
    heads = "".join(f"## Section {i}\n\nThe buyer pays within {i} days.\n\n" for i in range(12000))
    for text in ("ARTICLE 1. DEFINITIONS\n\n" + para * 8000, heads):
        assert len(text) > 500_000
        t0 = time.perf_counter()
        doc = LongDocument(text)
        assert time.perf_counter() - t0 < 1.0, len(doc)
    assert len({s.heading for s in LongDocument("ARTICLE 1. DEFINITIONS\n\n" + para * 8000).sections}) == 1
