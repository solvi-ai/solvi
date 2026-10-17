"""Updating a `solvi.build` system through the change pipeline: each re-build on new labels is a candidate change, checked
on held-out groups, for honesty and for refusals before it goes live; every version is journaled and can be rolled back.

The system: refund requests from twelve shops (the group), a promise of at most 2% answered alone and wrong, the rest
handed to a person; a refund larger than the order always goes to a person (a hard check). Shops s10 and s11 are never
given to `build`: their requests are the held-out check every update is measured on. Three updates are proposed:

  1. more labelled requests from the same shops           → promoted (version 1) when no held-out shop gets worse by
     more than the 2% tolerance: here on every build seed 0–4 (SEED); on other draws of the data a re-fit sometimes
     costs one shop more than that, and the slice guard refuses it — the script prints what happened
  2. a batch whose labels say "yes" to every late delivery → refused by the per-shop slices: the re-built system
     loses 0.06–0.11 per request on each held-out shop (build seeds 0–4), which its own calibration, on the same bad
     labels, cannot see
  3. the same batch again                                  → refused by the failure memory, nothing re-built

then version 1 is rolled back with a reason, and proposing it again is refused as a known failure. The stages (solvi.experimental.changes):
  - SliceGuard: per held-out shop, paired losses of the version in force and of the candidate (1 for a wrong answer
    given alone, 0.2 for a hand-over, 0 for a right answer), also against the original version, so small steps do not
    add up;
  - Canaries.drawn: the confident-error rate on 100 held-out requests drawn per change by the stage's secret, against the
    promise with one request of slack (100 draws cannot tell 2% from 3%; with no slack the verdict turned on one canary);
  - RefusalRegression (always on): requests that must go to a person (here: refunds larger than the order) stay with a
    person in every version;
  - the order is cheap first (Pipeline(order="cheap_first")), so a candidate the failure memory knows is not re-built.

What is missing for this in the library (on the roadmap): a ready target for a `build` system (here a dozen lines,
`BuildHead`), the per-group losses read from the decisions store instead of re-asking, and a ShipWatch on the live
version's stored outcomes.

    uv run python examples/26_build_head_update.py
"""
import random
import tempfile
import warnings
from pathlib import Path

from solvi import Answer, Catalog, Question
from solvi.solutions.decisions import build

with warnings.catch_warnings():
    warnings.simplefilter("ignore")                           # the changes module is experimental
    from solvi.experimental.changes import Canaries, Change, KnownFailures, Pipeline, SliceGuard

cat = Catalog()


@cat.fn
def refund_share(amount, order_total) -> float:
    return amount / order_total


@cat.check(hard=True)                                         # false: handed to a person, whatever the data say
def within_order(amount, order_total) -> bool:
    return amount <= order_total


@cat.fn
def delivered_late(days_late) -> bool:
    return days_late > 2


def policy(state):                                            # what a person decides
    share = state["amount"] / state["order_total"]
    return "yes" if (state["days_late"] > 2 and share < 0.8) or share < 0.4 else "no"


def request(rng, shop):
    total = round(rng.uniform(20, 400), 2)
    s = {"amount": round(total * rng.uniform(0.05, 1.0), 2), "order_total": total,
         "days_late": rng.choice([0, 0, 0, 1, 3, 6]), "shop": f"s{shop}"}
    return s, policy(s)


rng = random.Random(3)
SEEN, HELD_OUT = range(10), (10, 11)
DATA = {"v0": [request(rng, rng.choice(SEEN)) for _ in range(1200)]}
DATA["more"] = DATA["v0"] + [request(rng, rng.choice(SEEN)) for _ in range(600)]
DATA["late_yes"] = DATA["v0"] + [({**s, "days_late": 6}, "yes") for s, _ in
                                 (request(rng, rng.choice(SEEN)) for _ in range(600))]
HELD = {f"s{g}": [request(rng, g) for _ in range(150)] for g in HELD_OUT}
POOL = [{"id": f"{g}-{i}", "state": s, "label": y} for g, xs in HELD.items() for i, (s, y) in enumerate(xs)]
MUST_HAND_OVER = [{**c["state"], "amount": round(c["state"]["order_total"] * rng.uniform(1.05, 2.0), 2)}
                  for c in POOL[::15]]                        # 20 refunds larger than the order
QUESTION = Question("refund", "Refund without asking a person?", Answer.yes_no())


SEED = 0                                                      # build's seed: the outcomes above hold for 0–4


def make(name):
    return build(QUESTION, DATA[name], catalog=cat, max_risk=0.02, groups="shop", seed=SEED)


class BuildHead:
    """A change target for a `build` system: an "update" names the labelled data to re-build on; the candidate is built
    once and kept until it is promoted or dropped; promote swaps it in, rollback puts the one before back."""
    artifact = "update"

    def __init__(self, name):
        self.live, self.name, self._built = make(name), name, {}

    def system(self, change):
        if change is None:
            return self.live
        key = change.payload["data"]
        if key not in self._built:
            self._built[key] = make(key)
        return self._built[key]

    def answerer(self, change, ctx):                          # None: handed to a person
        s = self.system(change)
        return lambda state: s.ask(state).answer

    def promote(self, change, measured):
        before = self.name
        self.live, self.name = self.system(change), change.payload["data"]
        return {"data": self.name, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.live, self.name = make(ref["before"]), ref["before"]


def loss(answer, label):
    return 0.2 if answer is None else float(answer != label)


def slices(change, ctx):                                      # per held-out shop: (in force, candidate) losses
    now, cand = (ctx.pipeline.answerer(c, ctx, change) for c in (None, change))
    return {g: ([loss(now(s), y) for s, y in xs], [loss(cand(s), y) for s, y in xs]) for g, xs in HELD.items()}


def honesty(change, ctx, cases):                              # confident-error rate on the drawn canaries
    ans = ctx.pipeline.answerer(change, ctx, change)
    wrong = sum(1 for c in cases if (a := ans(c["state"])) is not None and a != c["label"])
    return {"n": len(cases), "confident_error_rate": wrong / len(cases)}


def update(data, why):
    return Change("outcome", artifact="update", payload={"data": data, "fingerprint": data}, proposer="labeller",
                  reason=why, change_type="rule")


if __name__ == "__main__":
    head = BuildHead("v0")
    stages = [KnownFailures(), SliceGuard(slices, tol=0.02, min_n=30),
              Canaries.drawn(honesty, POOL, 100, baseline={"confident_error_rate": 0.02}, tolerance=0.01, perfect=None)]
    with tempfile.TemporaryDirectory() as tmp:
        pipe = Pipeline(stages, Path(tmp) / "changes.jsonl", targets=[head], approval={"update": "gate"},
                        refusals=MUST_HAND_OVER, order="cheap_first")
        for data, why in [("more", "600 more labelled requests"), ("late_yes", "a batch from the late-delivery desk"),
                          ("late_yes", "the same batch, proposed again")]:
            res = pipe.propose(update(data, why))
            print(f"{why:32} → {res.status}" + (f" (version {res.version})" if res.promoted else f" by {res.stage}"))
            for v in res.verdicts:
                print(f"    {'ok ' if v.ok else 'NO '} {v.stage}: {v.reason[:110]}")
        print("versions:", [(v["version"], v["in_force"]) for v in pipe.versions()], "live data:", head.name)
        if head.name == "more":
            print("rollback undid:", pipe.rollback(0, by="ops", why="complaints from shop s3"), "live data:", head.name)
            res = pipe.propose(update("more", "600 more labelled requests"))
            print("again after the rollback →", res.status, "by", res.stage)
