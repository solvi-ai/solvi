"""solvi as an agent's tool (solvi 0.5): an LLM PROPOSES an action with quotes, solvi CHECKS and DECIDES.

NO LLM RUNS HERE. `stand_in_llm` is a SCRIPTED stand-in: for each order it returns the tool-call arguments a real LLM might
send (attempt 1, then a corrected attempt 2). In your own agent, the LLM's tool call arrives instead; the catalog does not
change.

- The proposed action must be one of the declared options (`Literal[...]`): "replace_and_voucher" is "outside the options".
- `require_evidence=True`: every quote must be literally in the email; one invented quote and the proposal is "not grounded".
- Order facts (days since delivery, order total) come from OUR records (`ORDERS`), never from the LLM.
- Two hard checks win over any confidence: returns after 30 days → reject; orders over 500 → human_review.

init_state: `order_id` and `attempt` pick the stand-in's proposal. To write your own proposal instead, add
`"proposal": {"action": "refund", "quotes": ["..."], "confidence": 0.9}`.

Try: A-1001 attempt 1 (an invented quote → abstain) then attempt 2 (accepted); A-1002 (44 days: the hard check decides);
A-1003 attempt 1 (an action outside the options); A-1004 (over the limit: a person decides)."""
from typing import Literal

from solvi import Catalog, Claim, Question

ORDERS = {                          # our own records: the LLM never supplies these facts
    "A-1001": {"days_since_delivery": 6, "order_total": 79.0},
    "A-1002": {"days_since_delivery": 44, "order_total": 35.0},
    "A-1003": {"days_since_delivery": 3, "order_total": 120.0},
    "A-1004": {"days_since_delivery": 2, "order_total": 1250.0},
}
EMAILS = {
    "A-1001": "The kettle I got last week leaks from the bottom. I'd like my money back, please.",
    "A-1002": "The lamp stopped working. Please refund it.",
    "A-1003": "The blender arrived cracked. A replacement would be great.",
    "A-1004": "The laptop screen arrived cracked. I want a refund.",
}


def stand_in_llm(order_id, attempt):
    """SCRIPTED stand-in for the LLM: what a real model might send as tool-call arguments."""
    script = {
        "A-1001": [{"action": "refund", "quotes": ["leaks from the bottom", "I want my money back"], "confidence": 0.9},
                   {"action": "refund", "quotes": ["leaks from the bottom", "I'd like my money back"], "confidence": 0.9}],
        "A-1002": [{"action": "refund", "quotes": ["stopped working"], "confidence": 0.95}],
        "A-1003": [{"action": "replace_and_voucher", "quotes": ["arrived cracked"], "confidence": 0.8},
                   {"action": "replace", "quotes": ["arrived cracked", "A replacement would be great"], "confidence": 0.85}],
        "A-1004": [{"action": "refund", "quotes": ["screen arrived cracked"], "confidence": 0.97}],
    }[order_id]
    return script[min(max(int(attempt), 1), len(script)) - 1]


def prepare(state):
    """Build the tool call: the email and order facts from our records, the proposal from the (stand-in) LLM."""
    oid = state["order_id"]
    p = state.get("proposal") or stand_in_llm(oid, state.get("attempt", 1))
    return {"email": EMAILS[oid], **ORDERS[oid], "proposed_action": p["action"], "proposed_quotes": list(p["quotes"]),
            "proposed_confidence": float(p["confidence"])}


class LLM:
    """Identity of the proposing model; it goes into the trace (type, id, fingerprint)."""
    model_id = "scripted-stand-in-llm"

    def fingerprint(self):
        return "scripted-v1"


cat = Catalog()


@cat.fn
def within_30_days(days_since_delivery: int) -> bool:
    return days_since_delivery <= 30


@cat.check(hard=True, then={"action": "reject"})
def return_window_open(within_30_days: bool) -> bool:
    """Returns after 30 days are rejected, whatever anyone proposes."""
    return within_30_days


@cat.check(hard=True, then={"action": "human_review"})
def under_auto_limit(order_total: float) -> bool:
    """Orders over 500 always go to a person."""
    return order_total <= 500


@cat.rule("action", model=LLM())
def action(email: str, proposed_action: str, proposed_quotes: list, proposed_confidence: float) \
        -> Literal["refund", "replace", "reject", "human_review"]:
    # the proposal, with its quotes as evidence: each quote is checked to be literally in `email`
    return Claim(proposed_action, evidence=list(proposed_quotes), confidence=proposed_confidence, source="email")


QUESTIONS = [Question("action", "What to do with the return request?", require_evidence=True,
                      checkpoints=["return_window_open", "under_auto_limit"])]
