"""Several models, one decision: a cascade, a vote and a route over deciders, with one guarantee for the whole.

A support desk routes emails to a team. It has a small decider (fast, ~45 ms, often sure on easy emails), a large one
(better, ~137 ms) and a decider of another family (different mistakes). Each is a decision part; solvi.multi combines
them with plain code, and act_guard puts one guarantee on the combination: P(answered alone and wrong) ≤ 10%.

  1. each model alone under the guarantee: how much it answers, its error, its cost;
  2. a cascade small → large: the large model only when the small one escalates — about the large model's answers at a
     fraction of its cost where the small one is often sure;
  3. a vote of two families: an answer only when both agree and both are sure — fewer errors among the answers;
  4. a route: code picks the model per input (long emails to the large model);
  5. one email through the cascade in a catalog: the audit lists each stage, the trace replays.

The measurements behind these modes (research note L25, real deciders, 300 calibration questions per set): the risk
stayed ≤ 10% for every mode; the cascade answered like the large model at ~half its cost on sets where the small model is
often sure; voting lowered the error among automatic answers from 2.1% to 0.4% on JSON questions.

The models are keyword stand-ins, so the example runs anywhere. With checkpoints, set SOLVI_DECIDE_SMALL and
SOLVI_DECIDE_MODEL (a folder or a Hugging Face id each; `solvi[onnx]` or `solvi[model]`).

Run:  uv run python examples/18_several_models.py"""
from __future__ import annotations

import os
import random
import zlib

import numpy as np

from solvi import Catalog, System
from solvi.decide import DecideModel
from solvi.multi import Cascade, Route, Vote

TASK = "Which team should handle this support email?"
TEAMS = {"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
         "shipping": "delivery, tracking, parcels"}
CORE = {"billing": ["I was charged twice for order {n}, please refund one payment.", "My invoice {n} shows the wrong amount.",
                    "Why did my card get billed again? I cancelled.", "The payment failed but the money left my account."],
        "technical": ["The app crashes when I open the settings.", "I get error {n} when I upload a photo.",
                      "The page freezes after the last update.", "The search button does nothing, I see an error."],
        "shipping": ["Where is my parcel {n}? The tracking has not moved.", "My order {n} was delivered damaged.",
                     "The courier says delivered but no package arrived.", "My delivery is three days late."]}
MIXED = ["The app crashed while I paid, and now I was charged twice for order {n}.",
         "My parcel {n} never arrived and the refund page shows an error.",
         "The tracking page crashes every time; where is my delivery?",
         "I was billed for order {n} but the courier lost the package."]
MIXED_GOLD = ["billing", "shipping", "shipping", "shipping"]
LONG = " I have written to you three times already about this and nobody answered, which is very disappointing."


def dataset(seed, n):
    """Support emails: most name their team plainly, one in four mixes two teams' words (the hard ones)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 4 == 3:
            k = rng.randrange(len(MIXED))
            text, team = MIXED[k].format(n=rng.randint(1000, 9999)), MIXED_GOLD[k]
        else:
            team = list(TEAMS)[i % 3]
            text = rng.choice(CORE[team]).format(n=rng.randint(1000, 9999))
        out.append((text + (LONG if rng.random() < 0.3 else ""), team))
    return out


class StandIn:
    """A keyword decider: a logit per option from its keywords, a label bias and deterministic noise of its own."""
    KEYWORDS = {"billing": ["charged", "refund", "invoice", "payment", "billed", "paid", "money"],
                "technical": ["crash", "error", "freez", "button", "update", "app"],
                "shipping": ["parcel", "tracking", "deliver", "courier", "package", "arrive"]}

    def __init__(self, name, weight, noise, bias=None, favour=None):
        self.model_id, self.weight, self.noise, self.bias = name, weight, noise, dict(bias or {})
        self.favour = favour or {}                      # a family's own reading of some words

    def fingerprint(self):
        return f"{self.model_id}-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = []
            for o in it.options:
                hits = sum(low.count(k) for k in self.KEYWORDS.get(o, [])) + sum(low.count(k) for k in self.favour.get(o, []))
                n = (zlib.crc32((self.model_id + o + it.text).encode()) % 1000) / 1000 - 0.5
                z.append(self.weight * hits + self.bias.get(o, 0.0) + self.noise * n)
            out.append(np.array(z))
        return out


def load(env, stand_in):
    src = os.environ.get(env)
    if src:
        return DecideModel.load(os.path.expanduser(src))
    return DecideModel(stand_in, meta={"format": "stand-in", "temperature": 1.0})


COST = {"small": 45.0, "large": 137.0, "other": 90.0}          # ms per question on a CPU (L19 measurements for S / L)


def risk_on(comb, test):
    """What happened on new emails: share answered alone, error among them, P(answered alone and wrong)."""
    ds = comb.decide([x for x, _ in test])
    auto = np.array([d.escalate is None for d in ds])
    wrong = np.array([d.value != y for d, (_, y) in zip(ds, test)])
    return auto.mean(), (wrong[auto].mean() if auto.any() else 0.0), (auto & wrong).mean()


def line(label, info, test_numbers, cost):
    a, e, r = test_numbers
    print(f"  {label:26s} answered alone {a:4.0%}, error among them {e:5.1%}, risk {r:5.1%}   "
          f"cost {cost:5.0f} ms/question   (calibration: threshold {info['threshold']:.2f})")


if __name__ == "__main__":
    small_m = load("SOLVI_DECIDE_SMALL", StandIn("small", 1.6, 4.0, bias={"technical": 0.6}))
    large_m = load("SOLVI_DECIDE_MODEL", StandIn("large", 2.4, 2.0))
    other_m = DecideModel(StandIn("other-family", 2.0, 3.0, favour={"shipping": ["lost", "late"], "billing": ["cancel"]}),
                          meta={"format": "stand-in", "temperature": 1.0})
    small = small_m.decision("team", TASK, "email", TEAMS)
    large = large_m.decision("team", TASK, "email", TEAMS)
    other = other_m.decision("team", TASK, "email", TEAMS)
    calib, test = dataset(1, 300), dataset(2, 1200)
    print(f"models: {small_m.model_id}, {large_m.model_id}, {other_m.model_id}; {len(calib)} labelled emails to calibrate on, "
          f"{len(test)} new ones")

    print("\n=== 1. each model alone, P(answered alone and wrong) ≤ 10% ===")
    for name, part in (("small", small), ("large", large), ("other", other)):
        one = Cascade([part], name="team")             # a one-part combination: the same calibration as below
        line(name, one.act_guard(calib, risk=0.10), risk_on(one, test), COST[name])

    print("\n=== 2. cascade small → large: the large model only when the small one escalates ===")
    cascade = Cascade([small, large], costs=[COST["small"], COST["large"]])
    info = cascade.act_guard(calib, risk=0.10)
    line("cascade small → large", info, risk_on(cascade, test), info["cost"])
    print(f"  answered by the small model {info['answered_by'][0]:.0%}, by the large {info['answered_by'][1]:.0%}; "
          f"{info['calls']:.2f} models called per question")

    print("\n=== 3. vote of two families: answer only when both agree and both are sure ===")
    vote = Vote([large, other], rule="all")
    info = vote.act_guard(calib, risk=0.10)
    line("vote large + other", info, risk_on(vote, test), COST["large"] + COST["other"])
    x, y = next((x, y) for x, y in test if "disagree" in (vote.decide(x).escalate or ""))
    print(f"  {x[:60]!r} (a {y} email): {vote.decide(x).escalate}")

    print("\n=== 4. route: long emails to the large model, the rest to the small one ===")

    def long_email(email):
        return len(email) > 120
    route = Route({long_email: large}, default=small)
    info = route.act_guard(calib, risk=0.10)
    share_long = np.mean([long_email(x) for x, _ in test])
    line("route by length", info, risk_on(route, test), share_long * COST["large"] + (1 - share_long) * COST["small"])

    print("\n=== 5. the cascade in a catalog: the audit lists each stage, the trace replays ===")
    cat = Catalog()
    q = cascade.question(cat, "team", "Which team handles the email?")
    system = System(cat, [q])
    for email in ("I was charged twice on invoice 5521, please refund the payment.", MIXED[1].format(n=7710)):
        res = system.ask({"email": email})
        print(res.audit("team"))
        print(f"  replay: {res.trace.replay(cat)['ok']}\n")
