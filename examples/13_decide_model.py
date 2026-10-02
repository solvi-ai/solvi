"""Decisions with a model: support-email routing by a decider, inside the same safeguards as any other part.

A decider (solvi-decide: a ModernBERT cross-encoder "task + options [SEP] text" → a probability per option) chooses the team
for a support email. It is a catalog part like any other: its answer is always one of the options, its provenance is
`decided`, the trace records the model, and the rest of the catalog is plain code — a refund detector with a rule-based
question, a constraint "refunds go to billing" (joint decoding repairs the team when the model disagrees), and a hard check
that sends legal threats to a person whatever the model says. Then:

  1. zero-shot accuracy on 60 test emails;
  2. label-bias correction from 60 unlabelled emails (no labels: the mean score of each team is subtracted);
  3. few-shot "S" from 16 labelled emails: a per-team shift and scale plus a temperature, so confidence is calibrated
     (ECE and coverage at 90% accuracy from solvi.calibration);
  4. abstention: the question's min_confidence; "other" is not a label the model scores but a threshold;
  5. an audit, a constraint repair, a hard check, and a correction absorbed at once by System.teach;
  6. escalation for a target error rate (calibrate_for), and a JSON ticket instead of a text (read as key paths).

The typed version of this desk — the questions as the fields of a pydantic model, several decided in one forward pass, the
model's own act / escalate signal — is examples/15_typed_decisions.py.

The real model runs when SOLVI_DECIDE_MODEL is set to a checkpoint folder or a Hugging Face id (needs `solvi[onnx]` or
`solvi[model]`); otherwise a small keyword stand-in with a built-in label bias plays its part, so the example runs anywhere.

Run:  uv run python examples/13_decide_model.py
      SOLVI_DECIDE_MODEL=<checkpoint folder> uv run --extra onnx python examples/13_decide_model.py"""
from __future__ import annotations

import json
import os
import random
import re
import zlib

import numpy as np

from solvi import Answer, Catalog, Question, System
from solvi.calibration import evaluate
from solvi.decide import DecideModel

TASK = "Which team should handle this support email?"
TEAMS = {"billing": "payments, invoices, refunds, double charges",
         "technical": "bugs, errors, crashes, the app or website not working",
         "shipping": "delivery, tracking, lost or damaged parcels",
         "account": "profile, email address, password, closing the account",
         "other": "none of the above: a person routes it"}

CORE = {
    "billing": ["I was charged twice for order {n}, please refund the duplicate payment.",
                "My invoice {n} shows the wrong amount, I paid for the basic plan.",
                "Why did my card get billed again this month? I cancelled the subscription.",
                "I need a copy of the invoice for my payment of {n} EUR.",
                "The payment failed but the money left my bank account.",
                "Can I get my money back for the order I returned last week?"],
    "technical": ["The app crashes every time I open the settings page.",
                  "I get error {n} when I try to upload a photo.",
                  "The website does not load on my phone since yesterday.",
                  "The search button does nothing, the page just freezes.",
                  "After the last update the app shows a blank screen.",
                  "Notifications stopped working on my tablet."],
    "shipping": ["Where is my parcel? The tracking number has not updated for a week.",
                 "My order {n} arrived damaged, the box was crushed.",
                 "The courier says delivered but I never received the package.",
                 "Can you ship my order to a different address?",
                 "My delivery is three days late, when will it arrive?",
                 "Only half of my order {n} was in the parcel."],
    "account": ["I want to change the email address on my profile.",
                "Please close my account and delete my data.",
                "I forgot my password and the reset link never arrives.",
                "How do I change the name on my account?",
                "Someone else logged into my account, please lock it.",
                "I would like to merge my two accounts into one."],
    "other": ["Do you sponsor local football clubs?",
              "Are you hiring developers in Berlin?",
              "I love your products, keep up the good work!",
              "Can I visit your office for a school project?",
              "Is your company planning to open a store in Lisbon?",
              "Who designed your logo? It is beautiful."],
}
GREET = ["Hello,", "Hi team,", "Good morning,", "Dear support,", ""]
CLOSE = ["Thanks.", "Best regards, Anna", "Kind regards, Tom", "Cheers", "Thank you in advance."]
LEGAL = " If this is not fixed by Friday my lawyer will take legal action."


def make(rng, team, legal=False):
    body = rng.choice(CORE[team]).format(n=rng.randint(1000, 9999))
    return f"{rng.choice(GREET)} {body}{LEGAL if legal else ''} {rng.choice(CLOSE)}".strip()


def dataset(seed, n):
    rng = random.Random(seed)
    teams = list(TEAMS)
    return [(make(rng, t), t) for t in (teams[i % len(teams)] for i in range(n))]


class StandInScorer:
    """Stands in for the decider when no checkpoint is given: a logit per option from keywords, plus a label bias (it likes
    "technical" and "account" whatever the text — the failure the unlabelled-text correction exists for) and a little noise."""
    model_id = "demo/stand-in-decider"
    KEYWORDS = {"billing": ["charged", "refund", "invoice", "payment", "billed", "money", "paid"],
                "technical": ["crash", "error", "load", "freez", "blank", "stopped working", "button"],
                "shipping": ["parcel", "tracking", "deliver", "courier", "ship", "arrive", "package"],
                "account": ["account", "password", "email address", "profile", "logged", "name on"]}
    BIAS = {"technical": 2.2, "account": 1.4}

    def fingerprint(self):
        return "stand-in-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            out.append(np.array([1.5 * sum(low.count(k) for k in self.KEYWORDS.get(o, [])) + self.BIAS.get(o, 0.0) - 1.0
                                 + 2.0 * ((zlib.crc32((o + it.text).encode()) % 1000) / 1000 - 0.5) for o in it.options]))
        return out


def load_model():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    if src:
        return DecideModel.load(os.path.expanduser(src))
    return DecideModel(StandInScorer(), meta={"format": "stand-in", "temperature": 1.0})


def build(model):
    cat = Catalog()
    team = model.decision("team", TASK, text_fact="email", options=TEAMS)

    def text(email):
        """The email as text: a JSON ticket (section 10) is read through its JSON."""
        return email if isinstance(email, str) else json.dumps(email, ensure_ascii=False)

    @cat.fn
    def mentions_refund(email):
        return bool(re.search(r"\brefund|money back|charged twice\b", text(email), re.I))

    @cat.rule("refund")
    def refund(mentions_refund):
        return "yes" if mentions_refund else "no"

    @cat.fn
    def legal_threat(email):
        return bool(re.search(r"\blawyer|legal action|court\b", text(email), re.I))

    @cat.check(hard=True, then={"team": "other"})
    def no_legal_threat(legal_threat):
        """A legal threat goes to a person, whatever the model says."""
        return not legal_threat

    @cat.constraint
    def refunds_go_to_billing(team, refund):
        return refund == "no" or team in ("billing", "other")

    questions = [team.question(cat, "team", "Which team handles the email?", min_confidence=0.55,
                               checkpoints=["no_legal_threat"]),
                 Question("refund", "Does the customer ask for a refund?", Answer.yes_no())]
    return cat, team, questions


def report(system, test, label):
    ev = evaluate(system, "team", [({"email": x}, y) for x, y in test])
    print(f"  {label:34s} accuracy {ev['accuracy_all']:.0%} of all, {ev['accuracy']:.0%} of answered, "
          f"answered {ev['answered']:.0%}, ECE {ev['ece']:.3f}, coverage at 90% accuracy {ev['coverage_at']:.0%}")
    return ev


if __name__ == "__main__":
    model = load_model()
    print(f"decider: {model.model_id} ({model.backend})")
    cat, team, questions = build(model)
    system = System(cat, questions)
    unlabelled = [x for x, _ in dataset(1, 60)]
    labelled = dataset(2, 16)
    test = dataset(3, 60)

    print("\n=== 1-3. zero-shot, bias correction without labels, 16 labelled examples ===")
    report(system, test, "zero-shot")
    team.adapt(unlabelled)
    print("  bias per team:", {o: round(b, 2) for o, b in zip(team.spec.real, team.adaptation.bias)})
    report(system, test, "+ bias correction (60 unlabelled)")
    team.fit(labelled)
    a = team.adaptation
    print(f"  S: scale {a.scale:.2f}, shift {({o: round(b, 2) for o, b in zip(team.spec.real, a.shift)})}, temperature "
          f"{a.temperature:.2f}, 'other' threshold {a.other_threshold if a.other_threshold is not None else model.other_threshold}")
    report(system, test, "+ S on 16 labelled")

    print("\n=== 4. one email: the audit ===")
    email = "Hello, my card was charged twice for order 5521, please refund one payment. Thanks."
    res = system.ask({"email": email})
    print(res.audit("team"))

    print("\n=== 5. the model and a rule disagree: joint decoding moves the answer to the most probable team the "
          "constraint allows (and the question abstains when that team is below min_confidence) ===")
    tricky = "Hi team, the app crashed with an error during checkout, can I get a refund? Cheers"
    r = system.ask({"email": tricky})
    was = f"the model said {r['team'].repaired[0]!r}, " if r["team"].repaired else "no repair needed, "
    print(f"  refund = {r['refund'].answer!r} by the rule; {was}team = {r['team'].answer!r} [{r['team'].status}] — "
          f"{r['team'].why}")

    print("\n=== 6. a legal threat: the hard check decides, the model is not asked ===")
    r = system.ask({"email": make(random.Random(7), "shipping", legal=True)})
    print(f"  team = {r['team'].answer!r} [{r['team'].status}] — {r['team'].why}")

    print("\n=== 7. unsure emails abstain (min_confidence 0.55); 'other' is a threshold, not a label ===")
    unsure = [(x, y) for x, y in test if system.ask({"email": x}, ["team"])["team"].status == "abstain"]
    print(f"  {len(unsure)} of {len(test)} test emails abstained")
    if unsure:
        x, y = unsure[0]
        print(f"  e.g. {x!r} (truly {y}): {system.ask({'email': x})['team'].why}")

    print("\n=== 8. a correction, absorbed at once ===")
    x, y = next(((x, y) for x, y in unsure if y != "other"), unsure[0] if unsure else test[0])
    before, p0 = team.fingerprint(), team.score(x)[y]
    ms = system.teach("team", {"email": x}, y)
    print(f"  System.teach(team = {y!r}) → the decision's shift updated in {ms:.1f} ms; P({y}) {p0:.2f} → "
          f"{team.score(x)[y]:.2f}; examples kept {team.adaptation.n_labelled}; fingerprint {before[:8]} → {team.fingerprint()[:8]}")
    rep = res.trace.replay(cat)
    print(f"  replay of the audited decision: ok={rep['ok']}; {rep['mismatches'][0][2] if rep['mismatches'] else ''}")

    print("\n=== 9. escalation for a target error rate ===")
    held_out = dataset(4, 40)
    info = team.calibrate_for(held_out, error=0.1)
    print(f"  calibrate_for(error=0.1) on 40 labelled emails: escalate below confidence {info['threshold']:.2f} "
          f"({info['signal']}); there it answers {info['coverage']:.0%} with {info['error']:.0%} errors")
    esc = [x for x, _ in test if system.ask({"email": x}, ["team"])["team"].status == "abstain"]
    print(f"  {len(esc)} of {len(test)} test emails now go to a person; e.g. {system.ask({'email': esc[0]})['team'].why[:90]}"
          if esc else "  no test email escalates")

    print("\n=== 10. a JSON ticket instead of a text: the decider reads it as key paths ===")
    ticket = {"subject": "Double charge", "body": "My card was charged twice for order 5521, please refund one payment.",
              "customer": {"tier": "pro", "since": "2023-04-01"}}
    print("  " + team.text_of({"email": ticket}).replace("\n", "\n  "))
    r = system.ask({"email": ticket}, ["team"])["team"]
    print(f"  team = {r.answer!r} [{r.status}] confidence {r.confidence:.2f} — {r.why[:120]}")

    print("\n=== lifetime safeguard stats ===")
    print(system.safeguard_report())
