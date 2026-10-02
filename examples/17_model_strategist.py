"""The model strategist (experimental): plans where producers are ambiguous, a model proposes, code verifies; name matching.

A payments team keeps a catalog with alternative producers of the same fact (`provides=`) — some need inputs nobody gives
here (a dead end), some are shortcuts that skip a step, some are slow. A second team wrote the approval rule with its own
names (`INV_TOTAL`, `FX_RATE`), so they match no fact.

  1. the deterministic strategist needs the inputs of EVERY producer of a fact: one dead end and the question abstains
  2. CostStrategist() — the same plan in declaration order, dead ends dropped: it answers; the plan is in the trace
  3. producers="equivalent": the producers are interchangeable, so the cheapest verified plan (declared costs)
  4. undeclared costs: the segment model (or, without one, a stand-in reading the docstrings) proposes which producer to use
     — every proposal is checked; a bad one falls back to the code plan (see the rejected proposal)
  5. names that match no fact: a matcher proposes aliases, examples (and targeted questions) decide, the catalog is rewired

The trained model runs when SOLVI_STRATEGIST points to a strategist checkpoint (docs/strategist.md; its
weights are not published with 0.5.0); otherwise stand-ins with the same interfaces play their part.

Run:  uv run python examples/17_model_strategist.py"""
from __future__ import annotations

import os
import re
from typing import Literal

import numpy as np

from solvi import Catalog, Question, System
from solvi.aliases import accept, apply, propose, unresolved
from solvi.strategy import CostStrategist, ModelStrategist


def catalog(costs=True):
    cat = Catalog()
    c = (lambda x: {"cost": x}) if costs else (lambda x: {})

    @cat.fn(**c(1))
    def invoice_amount(net: float, vat_rate: float) -> float:
        """Gross amount of the invoice."""
        return round(net * (1 + vat_rate), 2)

    @cat.fn(provides="fx_rate", **c(2))
    def fx_from_partner_feed(partner_feed: dict, currency: str) -> float:
        """Rate from the partner's live feed. Fast."""
        return partner_feed[currency]

    @cat.fn(provides="fx_rate", **c(40))
    def fx_from_archive(currency: str, value_date: str) -> float:
        """Slow: queries the rate archive and waits for it."""
        return {"EUR": 1.08, "GBP": 1.27}[currency]

    @cat.fn(provides="fx_rate", **c(3))
    def fx_from_table(currency: str) -> float:
        """Reads the treasury table (cached)."""
        return {"EUR": 1.08, "GBP": 1.27}[currency]

    @cat.fn(**c(1))
    def amount_usd(invoice_amount: float, fx_rate: float) -> float:
        """Gross amount in USD."""
        return round(invoice_amount * fx_rate, 2)

    @cat.check(hard=True, then={"approve": "reject"}, **c(1))
    def within_limit(amount_usd: float) -> bool:
        """Invoices above 50 000 USD are rejected here."""
        return amount_usd <= 50_000

    @cat.rule("approve")
    def approve(amount_usd: float) -> Literal["pay", "review", "reject"]:
        return "pay" if amount_usd < 10_000 else "review"
    return cat


QUESTIONS = [Question("approve", "Pay this invoice?", None)]
STATE = {"net": 7_000.0, "vat_rate": 0.2, "currency": "EUR", "value_date": "2026-09-27"}


class DocStandIn:
    """A stand-in with the segment model's interface: proposes the producer whose docstring sounds cheapest (and, once, a
    wrong part — to show that a bad proposal is caught)."""
    fingerprint = "standin00000000"

    def info(self):
        return {"type": "ModelStrategist", "id": "docstring stand-in", "fp": self.fingerprint}

    def propose(self, segs):
        out = []
        for s in segs:
            prod = [c for c in s["candidates"] if c["fact"] == s["fact"]]
            score = {c["name"]: len(re.findall(r"fast|cached|cheap", c["doc"].lower()))
                     - len(re.findall(r"slow|waits|archive", c["doc"].lower())) for c in prod}
            ready = {x for x, _, _ in s["available"]}
            best = max(prod, key=lambda c: score[c["name"]])                         # may be a dead end: code rejects it
            ok = max((c for c in prod if all(x in ready for x, _ in c["params"])), key=lambda c: score[c["name"]])
            out.append({"nodes": [best["name"]], "score": -0.1, "second": [ok["name"]]})
        return out


def strategist_model():
    path = os.environ.get("SOLVI_STRATEGIST")
    if path:
        from solvi.segment_model import SegmentModel
        return SegmentModel.load(path)
    return DocStandIn()


def main():
    print("1. deterministic strategist (needs the inputs of every producer of fx_rate — partner_feed is not given)")
    r = System(catalog(), QUESTIONS).ask(STATE)
    print(f"   approve = {r['approve'].answer!r} ({r['approve'].status}): {r['approve'].why}")

    print("\n2. CostStrategist(): declaration order, dead ends dropped")
    ms = CostStrategist()
    s = System(catalog(), QUESTIONS, strategist=ms)
    r = s.ask(STATE)
    print(f"   approve = {r['approve'].answer!r}; fx_rate from {next(x.producer for x in r.trace.records if x.name == 'fx_rate')}")
    print("   plan record:", r.trace.records[-1].name, r.trace.records[-1].provenance, "— replay ok:",
          r.trace.replay(s, r.flow)["ok"])

    print("\n3. producers='equivalent' with declared costs: the cheapest verified plan")
    ms = CostStrategist(producers="equivalent")
    r = System(catalog(), QUESTIONS, strategist=ms).ask(STATE)
    print(f"   chosen: {ms.last['choice']['fx_rate']} (plan cost {ms.last['cost']:g}); mandatory checks {ms.last['mandatory']}")

    print("\n4. costs not declared: the model proposes, code verifies")
    model = strategist_model()
    ms = ModelStrategist(model, producers="equivalent")
    s = System(catalog(costs=False), QUESTIONS, strategist=ms)
    r = s.ask(STATE)
    for seg in ms.last["segments"]:
        verdict = seg.get("accepted") or ("rejected: " + seg.get("rejected", ""))
        first = f" (first proposal rejected: {seg['first_rejected']})" if seg.get("first_rejected") else ""
        print(f"   {seg['fact']}: code {seg['code']} · model {seg['proposed']}{first} → {seg['by']}: {verdict}")
    rec = r.trace.records[-1]
    print(f"   answer {r['approve'].answer!r}; plan record provenance {rec.provenance}, model {rec.model}")
    print("   replay ok:", r.trace.replay(s, r.flow)["ok"])

    print("\n5. another team's names: INV_TOTAL, FX match no fact")
    cat = catalog()

    @cat.rule("approve")
    def approve(INV_TOTAL: float, FX: float) -> Literal["pay", "review", "reject"]:
        return "pay" if INV_TOTAL * FX < 10_000 else "review"
    init = set(STATE)
    print("   unresolved:", sorted(unresolved(cat, init)))
    matcher = _matcher()
    types = {"net": float, "vat_rate": float, "currency": str, "value_date": str}
    props = propose(cat, QUESTIONS, init, matcher, init_types=types)
    states = [{**STATE, "net": n, "currency": c} for n, c in [(1_000, "EUR"), (7_000, "GBP"), (9_000, "EUR"), (3_000, "GBP"),
                                                              (8_500, "GBP"), (500, "EUR")]]
    truth = System(catalog(), QUESTIONS, strategist=CostStrategist())

    def label(st):
        return {"approve": truth.ask(st)["approve"].answer}
    got = accept(cat, QUESTIONS, props, [(st, label(st)) for st in states[:3]], probes=states[3:], oracle=label, active=(2, 3))
    print(f"   accepted: {got.aliases} — {got.why} ({got.labels} labels)")
    if got.aliases:
        cat2 = apply(cat, got.aliases)
        r = System(cat2, QUESTIONS, strategist=CostStrategist()).ask(STATE)
        print(f"   approve = {r['approve'].answer!r}; catalog aliases {cat2.aliases}")


def _matcher():
    path = os.environ.get("SOLVI_STRATEGIST")
    if path and os.path.isdir(os.path.join(path, "matcher")):
        from solvi.aliases import NameMatcher
        return NameMatcher.load(os.path.join(path, "matcher"))

    class LetterMatcher:                              # stand-in: names as bags of letters, plus a synonym for FX
        T = 0.05

        def embed(self, texts, names):
            out = []
            for t, n in zip(texts, names):
                w = (n or t.split("|")[0]).lower().replace("fx", "fx rate").replace("inv", "invoice")
                v = np.zeros(27)
                for ch in w:
                    if ch.isalpha():
                        v[ord(ch) - 97] += 1
                out.append(v / (np.linalg.norm(v) + 1e-9))
            return np.array(out)
    return LetterMatcher()


if __name__ == "__main__":
    main()
