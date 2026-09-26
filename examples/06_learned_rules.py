"""Rules learned from examples: route a parcel to a delivery zone from a free-form address. Nobody writes the rule — solvi learns a
short, readable decision list ("if the address has postcode starting 43 -> North") from labeled addresses, and it then works like
any hand-written rule: deterministic, explained, re-checkable.

Run:  uv run python examples/06_learned_rules.py"""
from __future__ import annotations

import random

from solvi import Answer, Catalog, Question, System
from solvi.show import show

ZONES = {"North": (["Aldmoor", "Kestrel Bay", "Northwick"], ["43", "44"]),
         "Central": (["Midtown", "Harlow", "Crossfield"], ["50", "51", "52"]),
         "South": (["Portsea", "Southend", "Rivermouth"], ["80", "81"]),
         "Islands": (["Isle of Fen", "Gullrock"], ["97"])}
STREETS = ["High St", "Mill Lane", "Station Rd", "Church Way", "Harbour Rd", "Oak Ave"]

cat = Catalog()


@cat.fn
def address_upper(address):
    return address.upper()


@cat.check
def looks_complete(address):
    """soft: an address should carry a 5-digit postcode"""
    return any(t.isdigit() and len(t) == 5 for t in address.replace(",", " ").split())


QUESTIONS = [Question("zone", "Which delivery zone?", Answer.choice(list(ZONES)))]


def make(rng):
    zone = rng.choice(list(ZONES))
    cities, prefixes = ZONES[zone]
    city = rng.choice(cities)
    parts = [f"{rng.randint(1, 250)} {rng.choice(STREETS)}"]
    if rng.random() < 0.85:                                   # sometimes the city is missing
        parts.append(city if rng.random() < 0.9 else city.lower())
    if rng.random() < 0.9:                                    # sometimes the postcode is missing
        parts.append(rng.choice(prefixes) + f"{rng.randint(0, 999):03d}")
    return {"address": ", ".join(parts)}, zone


if __name__ == "__main__":
    rng = random.Random(0)
    system = System(cat, QUESTIONS)
    train = [make(rng) for _ in range(800)]
    rules = system.learn_rule("zone", train, ["address"])
    print(f"learned from {len(train)} labeled addresses — {len(rules.rules)} rules:\n{rules}\n")
    test = [make(rng) for _ in range(1000)]
    acc = sum(system.ask(s)["zone"].answer == z for s, z in test) / len(test)
    print(f"accuracy on 1000 new addresses: {acc:.3f}\n")
    for addr in ("17 Harbour Rd, Portsea", "4 Mill Lane, 97012"):
        answer, fired = rules.predict({"address": addr})
        print(f"{addr!r} -> {answer}   (rule: if {fired['if']})" if fired else f"{addr!r} -> {answer}   (default)")
    print()
    show(system.ask({"address": "17 Harbour Rd, Portsea"}), cat)
