"""The code strategist: plans where a fact has several producers — dead ends dropped, the cheapest verified plan.

A payments team keeps a catalog with alternative producers of the same fact (`provides=`) — some need inputs nobody gives
here (a dead end), some are slow.

  1. the deterministic strategist needs the inputs of EVERY producer of a fact: one dead end and the question abstains
  2. CostStrategist() — the same plan in declaration order, dead ends dropped: it answers; the plan is in the trace
  3. producers="equivalent": the producers are interchangeable, so the cheapest verified plan (declared costs)

No model is involved. Run:  uv run python examples/17_cost_strategist.py"""
from __future__ import annotations

from typing import Literal

from solvi import Catalog, Question, System
from solvi.core.plan.cost import CostStrategist


def catalog():
    cat = Catalog()
    c = lambda x: {"cost": x}                         # the declared cost of a part

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


if __name__ == "__main__":
    main()
