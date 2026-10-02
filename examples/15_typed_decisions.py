"""Typed decisions: types declare the questions, the model proposes, checks decide.

A support desk receives tickets as a pydantic model (subject, body, customer). The questions are the fields of another
pydantic model, and each field's type says what kind of question it is:

    team:    Literal["billing", "technical", "shipping", "account"]      one option (choice)
    urgency: Scale[Literal["low", "medium", "high", "critical"]]         ordered levels (score)
    angry:   bool                                                        yes / no (noul)
    topics:  list[Literal["refund", "delay", "bug", "data_loss"]]        every option that applies (multi)

One decider answers all four over the ticket (the state is serialized as key paths) — in ONE forward pass when the model
shares passes (the stand-in does; the published checkpoints answer one question per pass unless loaded with
multi_question=True) — each answer with
probabilities, a calibrated confidence and act / escalate. Around the model, plain code decides: a hard check sends legal
threats to a person as "critical", a constraint sends refunds to billing (joint decoding), a rule reads the customer's tier.

  1. the questions from the types; the ticket as the model reads it
  2. one ticket: four answers, and how many forward passes they took; the audit
  3. rules, a hard check and a constraint override the model
  4. an unclear ticket: the model escalates — abstain, "model escalated" in the audit and the stats
  5. a correction absorbed at once; the traces replay

The real decider runs when SOLVI_DECIDE_MODEL points to a checkpoint (a single-question checkpoint has no act head and
answers one question per pass: the example then escalates by calibrated confidence instead); otherwise a keyword
stand-in with the typed v1 contract (act head, several questions per pass) plays its part.

Run:  uv run python examples/15_typed_decisions.py"""
from __future__ import annotations

import os
import re
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from solvi import Answer, Catalog, Question, Scale, System
from solvi.decide import DecideModel


# --- the input: a ticket
class Customer(BaseModel):
    name: str
    tier: Literal["free", "pro", "enterprise"]
    open_tickets: int = 0


class Ticket(BaseModel):
    subject: str
    body: str
    customer: Customer
    order_id: Optional[str] = None


# --- the questions: one field = one question, its type = its kind, its description = the task
class Triage(BaseModel):
    team: Literal["billing", "technical", "shipping", "account"] = Field(
        description="Which team should handle this ticket?",
        json_schema_extra={"options": {"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
                                       "shipping": "delivery, parcels, couriers", "account": "login, password, profile"}})
    urgency: Scale[Literal["low", "medium", "high", "critical"]] = Field(description="How urgent is this ticket?")
    angry: bool = Field(description="Is the customer angry?")
    topics: list[Literal["refund", "delay", "bug", "data_loss"]] = Field(description="What does the ticket mention?")


class StandIn:
    """A keyword stand-in for a decider with the typed v1 contract: every question kind natively, an act logit per question
    (low when the ticket gives little evidence), several questions per forward pass."""
    model_id = "demo/stand-in-typed-decider"
    META = {"format": "l14f typed v1", "multi_question": {"layout": "block", "max_questions": 6},
            "temperature": {"choice": 1.0, "multi": 1.0, "score": 1.0, "noul": 1.0}, "act": {"threshold": 0.5}}
    KW = {"billing": ["charged", "refund", "invoice", "payment"], "technical": ["crash", "error", "bug", "app"],
          "shipping": ["parcel", "delivery", "courier", "late"], "account": ["password", "login", "profile"],
          "refund": ["refund", "money back"], "delay": ["late", "still waiting", "delay"], "bug": ["crash", "bug", "error"],
          "data_loss": ["lost my data", "deleted", "data loss"]}
    ANGRY = ["furious", "terrible", "unacceptable", "angry", "!!"]

    def fingerprint(self):
        return "stand-in-typed-1"

    def _one(self, it, text):
        low = text.lower()
        hits = lambda o: sum(low.count(k) for k in self.KW.get(o, []))          # noqa: E731
        if it.mode == "noul":
            z = [2.5 if any(w in low for w in self.ANGRY) else -2.5, 0.0]
        elif it.mode == "score":
            u = min(3, low.count("!") // 2 + 2 * any(w in low for w in ("urgent", "asap", "today", "lawyer")))
            z = [-1.5 * abs(k - u) for k in range(len(it.options))]
        else:
            z = [2.0 * hits(o) - (1.0 if it.mode == "multi" else 0.0) for o in it.options]
        evidence = max(hits(o) for o in ("billing", "technical", "shipping", "account"))
        return {"logits": np.array(z, float), "act": 3.0 if evidence else -2.0}

    def logits(self, items):
        return [self._one(it, it.text) for it in items]

    def logits_pass(self, passes):
        return [[self._one(it, p.text) for it in p.items] for p in passes]


def load_model():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    if src:
        return DecideModel.load(os.path.expanduser(src))
    return DecideModel(StandIn(), StandIn.META)


def build(model):
    cat = Catalog()
    reads = ["subject", "body", "customer"]                 # the decider reads these facts as one state
    extra = {} if model.has_act else {"escalate_below": 0.55}
    questions = model.questions(cat, Triage, text_fact=reads, **extra)

    @cat.fn
    def legal_threat(body: str) -> bool:
        return bool(re.search(r"\b(lawyer|legal action|court)\b", body, re.I))

    @cat.check(hard=True, then={"urgency": "critical"})
    def no_legal_threat(legal_threat: bool) -> bool:
        """A legal threat is critical, whatever the model says."""
        return not legal_threat

    @cat.constraint
    def refunds_go_to_billing(team, topics):
        return "refund" not in (topics or ()) or team == "billing"

    @cat.rule("priority_support")
    def priority_support(customer: Customer) -> bool:
        return customer.tier == "enterprise"

    for q in questions:
        if q.name == "urgency":
            q.requires.append("no_legal_threat")
    questions.append(Question("priority_support", "Does the customer get priority support?", Answer.yes_no()))
    return cat, questions


def line(res):
    return "  ".join(f"{q}={r.answer!r}" if r.status == "ok" else f"{q}=— [{r.status}{', ' + r.guard if r.guard else ''}]"
                     if r.status == "abstain" else f"{q}={r.answer!r} [{r.status}]" for q, r in res.results.items())


TICKET = Ticket(subject="Charged twice", body="I was charged twice for order 5521 and want a refund. This is unacceptable!!",
                customer=Customer(name="Anna", tier="enterprise", open_tickets=2), order_id="5521")

if __name__ == "__main__":
    model = load_model()
    print(f"decider: {model.model_id} ({model.backend}); act head: {model.has_act}; "
          f"questions per pass: {model.caps['max_questions'] or 1}; state format: {model.state_format}")
    cat, questions = build(model)
    system = System(cat, questions, input_model=Ticket)

    print("\n=== 1. the questions come from the types ===")
    for q in questions:
        print(f"  {q.name:17s} {q.answer.kind:8s} {q.answer.options}")
    part = cat.rules["team"].func
    print("  the ticket as the model reads it:")
    print("    " + part.text_of(TICKET.model_dump()).replace("\n", "\n    "))

    print("\n=== 2. one ticket: the answers, " + ("one forward pass" if model.batchable else "one forward pass per question")
          + " ===")
    before = model.passes
    res = system.ask(TICKET)
    print("  " + line(res))
    decided = sum(r.origin == "decided" for r in res.trace.records)
    print(f"  forward passes: {model.passes - before} for {decided} decided questions; shared passes {res.flow.batches}")
    o = res.overall
    print(f"  overall: confidence {o['confidence']:.2f} (all answered right), weakest {o['weakest'][0]} "
          f"{o['weakest'][1]:.2f}, {o['answered']} answered, complete {o['complete']}")
    print(res.audit("urgency"))

    print("\n=== 3. plain code overrides the model: a hard check, a constraint, a rule ===")
    legal = TICKET.model_copy(update={"subject": "App crash", "body": "The app crashes and I want a refund, or my lawyer "
                                                                      "will call you.",
                                      "customer": Customer(name="Tom", tier="free")})
    r = system.ask(legal)
    print("  " + line(r))
    print(f"  urgency: {r['urgency'].why}")
    rep = r["team"].repaired or r["topics"].repaired
    print(f"  constraint: {'repaired ' + repr(rep) if rep else 'satisfied'}; priority_support by the rule: "
          f"{r['priority_support'].answer!r}")

    print("\n=== 4. an unclear ticket: the model escalates ===")
    vague = TICKET.model_copy(update={"subject": "Question", "body": "Hello, I have a question about something."})
    r = system.ask(vague)
    print("  " + line(r))
    print(f"  team: {r['team'].why}")
    print("  " + r.audit("team").safeguard_line())
    print(f"  overall: confidence {r.confidence:.2f} over the answered questions; complete {r.complete}, "
          f"abstained {r.overall['abstained']}")

    print("\n=== 5. the traces replay; a correction is absorbed at once ===")
    print(f"  replay: first ticket ok={res.trace.replay(cat)['ok']}, legal threat ok={system.ask(legal).trace.replay(cat)['ok']}, "
          f"unclear ticket ok={r.trace.replay(cat)['ok']}")
    system.teach("urgency", TICKET, "high")                  # the first correction also loads scipy's optimizer
    ms = system.teach("urgency", legal, "critical")
    print(f"  System.teach(urgency = ...) → the decision's shift updated in {ms:.1f} ms; "
          f"{cat.rules['urgency'].func.adaptation.n_labelled} corrections kept")
    rep = res.trace.replay(cat)
    print(f"  replay of the first ticket now: {rep['mismatches'][0][2][:70]}…")

    print("\n=== lifetime safeguard stats ===")
    print(system.safeguard_report())
