"""One entry point: a question, labelled examples, a promise and a slow path in — a ready System 1 + dispatcher out.

`solvi.auto.build` fits System 1 from the examples (here a ridge head over the facts the catalog computes), splits
the examples so that every promise is calibrated on examples the fitted part did not see, calibrates System 1's
guarantee, then who answers each slice System 1 hands over (System 1's own guess, the slow path, the slow path when it
agrees, or a person), and stores every decision. `explain()` says what it chose; `report()` reads the store.

The slow path here is a stand-in for a slower reader (an LLM in practice: `slow=llm(...)`, with `price=` and a budget):
it also reads the agent's free-text notes, which decide the requests between 40% and 70% of the order.

    uv run python examples/24_one_entry_point.py
"""
import random
import tempfile
from pathlib import Path

from solvi import Answer, Catalog, Question
from solvi.auto import build

cat = Catalog()


@cat.fn
def refund_share(amount, order_total) -> float:            # how much of the order is asked back
    return amount / order_total


@cat.fn
def delivered_late(days_late) -> bool:
    return days_late > 2


def notes_reader(state):                                    # the slow path: also reads the agent's free-text notes
    share = state["amount"] / state["order_total"]
    return "yes" if state["days_late"] > 2 or share < 0.4 or (share < 0.7 and "loyal" in state["notes"]) else "no"


def request(rng, n):
    total = round(rng.uniform(20, 400), 2)
    state = {"amount": round(total * rng.uniform(0.05, 1.0), 2), "order_total": total,
             "days_late": rng.choice([0, 0, 0, 1, 3, 6]),
             "notes": f"order {n}: customer since {rng.randint(2012, 2026)}, " + rng.choice(["loyal", "new", "returning"])}
    return state, notes_reader(state)                       # the policy a person applies


rng = random.Random(7)
examples = [request(rng, n) for n in range(2000)]
question = Question("refund", "Refund without asking a person?", Answer.yes_no())

with tempfile.TemporaryDirectory() as tmp:
    s = build(question, examples, catalog=cat, max_risk=0.02, slow=notes_reader, storage=Path(tmp) / "decisions.jsonl")
    print(s.explain())
    print()
    for state, _ in [request(rng, n) for n in range(5000, 5006)]:
        res = s.ask(state)
        print(f"{res.by:5} {res.answer!s:4} {res.reasons[-1][:90]}")
    print()
    out = [s.ask(state) for state, _ in [request(rng, n) for n in range(6000, 6500)]]
    print({by: sum(r.by == by for r in out) for by in ("s1", "s2", "human")}, "replay failures:", len(s.replay_all()))
