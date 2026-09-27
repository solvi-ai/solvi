"""Typed decisions (solvi 0.5): the questions are the fields of a pydantic model, a decider answers them, code keeps the
guarantees.

NO MODEL RUNS HERE. The decider below is `StandIn`: a keyword stand-in with the decider's contract (one logit per option,
an "act" logit that says "I am unsure, escalate", several questions per forward pass). Every confidence you see comes from
keyword counts, not from a trained model. In your own code, swap it for a real checkpoint:
`DecideModel.load("<folder or HF id>")` (pip install "solvi[onnx]"); the catalog does not change.

- `EmailTriage` declares three questions by type: `Literal[...]` → one of five categories, `Scale[...]` → an ordered level,
  `bool` → yes / no. `model.questions(cat, EmailTriage, text_fact="email")` turns the fields into Questions.
- The answer is one of the declared options by construction; an unsure decision (act < 0.5) is ESCALATED: the answer
  abstains and the Audit says "model escalated" — that email goes to a person.
- A hard check the model cannot override: a possible account takeover is `security` / `high`, whatever the model says.

Try: the email "Someone hacked my account and changed the payment card." (the hard check decides), a vague one like
"Hello, quick question about something from last week." (escalated), or "I can't log in to download the invoice."
(torn between account and billing: escalated)."""
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from solvi import Catalog, Scale
from solvi.decide import DecideModel

LABELS = ["billing", "technical", "account", "security", "other"]


class EmailTriage(BaseModel):
    category: Literal["billing", "technical", "account", "security", "other"] = Field(
        description="What is this email about?",
        json_schema_extra={"options": {"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
                                       "account": "login, password, profile", "security": "hacked account, fraud",
                                       "other": "none of the above"}})
    urgency: Scale[Literal["low", "medium", "high"]] = Field(description="How urgent is it?")
    angry: bool = Field(description="Is the sender angry?")


class StandIn:
    """A STAND-IN decider, not a model: keyword logits for every question kind, an act logit that is low when the email
    gives little to go on or is torn between two categories, all questions in one forward pass."""
    model_id = "demo/stand-in-decider"
    META = {"format": "l14f typed v1", "multi_question": {"layout": "block", "max_questions": 4},
            "temperature": {"choice": 1.0, "score": 1.0, "noul": 1.0}, "act": {"threshold": 0.5}}
    KW = {"billing": ["invoice", "charged", "refund", "payment"], "technical": ["error", "crash", "bug", "export"],
          "account": ["password", "login", "log in", "email address", "profile"],
          "security": ["hacked", "suspicious", "fraud"]}

    def fingerprint(self):
        return "stand-in-decider-1"

    def _one(self, it, text):
        low = text.lower()
        hits = {o: sum(low.count(k) for k in self.KW.get(o, [])) for o in LABELS}
        if it.mode == "noul":
            z = [2.5 if re.search(r"unacceptable|!!|ridiculous|third time", low) else -2.5, 0.0]
        elif it.mode == "score":
            u = min(2, 2 * bool(re.search(r"urgent|asap|today|right now", low)) + ("!" in low))
            z = [-1.5 * abs(k - u) for k in range(len(it.options))]
        else:
            z = [2.0 * hits.get(o, 0) for o in it.options]
        top, second = sorted(hits.values(), reverse=True)[:2]
        return {"logits": np.array(z, float), "act": 2.5 * (top - second) - 1.0 if top else -2.0}

    def logits(self, items):
        return [self._one(it, it.text) for it in items]

    def logits_pass(self, passes):
        return [[self._one(it, p.text) for it in p.items] for p in passes]


model = DecideModel(StandIn(), StandIn.META)         # a real checkpoint: DecideModel.load("<folder or HF id>")
cat = Catalog()
QUESTIONS = model.questions(cat, EmailTriage, text_fact="email")


@cat.fn
def account_takeover(email: str) -> bool:
    return bool(re.search(r"\b(?:hacked|didn't log in|not me who logged in|unknown device)\b", email, re.I))


@cat.check(hard=True, then={"category": "security", "urgency": "high"})
def no_account_takeover(account_takeover: bool) -> bool:
    """A possible account takeover goes to security as high urgency, whatever the model says."""
    return not account_takeover


for q in QUESTIONS:
    if q.name in ("category", "urgency"):
        q.checkpoints.append("no_account_takeover")
