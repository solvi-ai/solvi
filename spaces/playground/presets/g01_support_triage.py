"""Support triage: a customer message -> intent, urgent?, refund requested?, priority, route.

Evidence phrases are found with patterns and kept as Quotes (character offsets in the message); a phrase in a negated clause
("I don't want a refund", "not urgent") counts as evidence for "no". Intent is a readable rule list learned in milliseconds from
200 synthetic labeled tickets (at import, so it also happens in the browser). Hard checks: a legal threat, a chargeback threat
or a VIP past the 4-hour SLA force priority "high" whatever the rule computes.
Try: add "or I will dispute the charge with my bank" to the message, or set "tier" to "vip"."""
import random
import re
from datetime import datetime

from solvi import Answer, Catalog, Question, Quote
from solvi.rules import RuleList

cat = Catalog()
INTENTS = ["refund", "technical_help", "billing_question", "information", "cancellation", "other"]
NEGATORS = {"not", "no", "never", "don't", "dont", "doesn't", "didn't", "isn't", "won't", "without"}


def prepare(state):
    for k in ("received_at", "now"):
        if isinstance(state.get(k), str):
            state[k] = datetime.fromisoformat(state[k])
    return state


def _negator(text, start):
    """Offset of a negator in the same clause, at most 4 words before `start` (or None)."""
    clause = re.split(r"[.,;:!?]|\bbut\b|\bhowever\b", text[:start])[-1]
    words = list(re.finditer(r"[\w']+", clause))[-4:]
    for w in words:
        if w.group().lower() in NEGATORS:
            return start - len(clause) + w.start()
    return None


def _evidence(text, patterns, exclude=None):
    """First un-negated match -> Quote(True); else the first negated one -> Quote(False) citing the negation; else no evidence."""
    negated = None
    for m in sorted((m for p in patterns for m in re.finditer(p, text, re.I)), key=lambda m: m.start()):
        if exclude and re.match(exclude, text[m.end():], re.I):
            continue
        n = _negator(text, m.start())
        if n is None:
            return Quote(True, m.start(), m.end(), source="message")
        negated = negated or Quote(False, n, m.end(), source="message")
    return negated or Quote(False, 0, 0, source="message")


# ---------- evidence from the text (each value keeps its quote)
@cat.extract
def urgency(message):
    "time pressure or a blocked business"
    return _evidence(message, [r"\burgent(?:ly)?\b", r"\basap\b", r"\bas soon as possible\b", r"\bimmediately\b",
                               r"\bright (?:now|away)\b", r"\b(?:fixed|done|resolved|sorted) today\b", r"\bdeadline\b",
                               r"\b(?:by|before) (?:tonight|tomorrow|end of (?:the )?day|monday|tuesday|wednesday|thursday|friday)\b",
                               r"\b(?:production|site|checkout|store|shop|service) (?:is|went) down\b",
                               r"\b(?:team is|we are|we're|i am|i'm) (?:completely )?blocked\b", r"\bcan(?:'t|not) (?:work|sell|take orders)\b",
                               r"\blosing (?:money|sales|orders|customers)\b"])


@cat.extract
def refund_ask(message):
    "asks for money back (not a question about the refund policy)"
    return _evidence(message, [r"\brefund(?:ed)?\b", r"\bmoney back\b", r"\breimburse(?:d|ment)?\b", r"\bpay me back\b",
                               r"\breverse (?:the |this |that )?(?:duplicate |extra |second )?(?:charge|payment)\b"],
                     exclude=r"\s+(?:policy|policies|terms|window|rules)\b")


@cat.extract
def legal_threat(message):
    "threatens lawyers, a lawsuit or a regulator"
    return _evidence(message, [r"\b(?:lawyer|attorney|solicitor)s?\b", r"\b(?:sue|suing)\b", r"\blegal action\b",
                               r"\bsmall claims\b", r"\bcourt\b", r"\breport (?:you|this|your company) to\b",
                               r"\b(?:consumer protection|ombudsman|trading standards)\b"])


@cat.extract
def chargeback_threat(message):
    "threatens a chargeback / bank dispute"
    return _evidence(message, [r"\bcharge ?back\b", r"\bdispute (?:the |this |these |that )?(?:charge|payment|transaction)s?\b",
                               r"\b(?:call|contact|tell) my bank\b", r"\b(?:through|with|via) my bank\b"])


@cat.extract
def anger(message):
    "strong frustration"
    return _evidence(message, [r"\bunacceptable\b", r"\bridiculous\b", r"\bworst\b", r"\bfurious\b", r"\bscam\b", r"!{2,}",
                               r"\b(?:third|fourth|fifth) time\b", r"\bfed up\b", r"\bdisgusted\b"])


# ---------- words for the learned intent rules: stop words dropped, crude stemming, words in a negated clause get the prefix "not"
STOP = set("a an the i im i'm me my we our us you your it its is are was were be been to of for and or in on at with this that "
           "have has had do does did can could would will should please hi hello hey thanks thank regards team just so how what "
           "why when who where which there want like need some any all from about into after before also get got very much more "
           "they them their".split())


def _stem(w):
    for suf in ("ation", "ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


@cat.fn
def cue_words(message):
    out, neg = [], 0
    for tok in re.findall(r"[a-z']+|[.,;:!?]", message.lower()):
        if tok in ".,;:!?" or tok in ("but", "however"):
            neg = 0
        elif tok in NEGATORS:
            neg = 3
        elif tok not in STOP and len(tok) > 2:
            out.append(("not" if neg else "") + _stem(tok))
            neg = max(0, neg - 1)
    return " ".join(dict.fromkeys(out))


@cat.fn
def intent_class(cue_words):
    return INTENT_RULES.predict({"cue_words": cue_words})[0]


@cat.fn
def intent_rule_line(cue_words):
    """which line of the learned list fired"""
    _, r = INTENT_RULES.predict({"cue_words": cue_words})
    return f"if {r['if']} -> {r['then']} ({r['support']}/{r['covered']})" if r else f"no line matched -> default {INTENT_RULES.default}"


@cat.fn
def waiting_hours(received_at, now):
    return round((now - received_at).total_seconds() / 3600, 1)


# ---------- hard checks: False forces the answer, nothing below can change it
@cat.check(hard=True, then={"priority": "high", "route": "legal"})
def no_legal_threat(legal_threat):
    return not legal_threat


@cat.check(hard=True, then={"priority": "high"})
def no_chargeback_threat(chargeback_threat):
    return not chargeback_threat


@cat.check(hard=True, then={"priority": "high"})
def vip_sla_ok(tier, waiting_hours):
    """a VIP ticket waiting 4 hours or more is high priority"""
    return tier != "vip" or waiting_hours < 4


# ---------- answer rules
@cat.rule("intent")
def intent(intent_class, intent_rule_line):
    return intent_class


@cat.rule("urgent")
def urgent(urgency):
    return urgency


@cat.rule("refund_requested")
def refund_requested(refund_ask):
    return refund_ask


@cat.rule("priority")
def priority(intent_class, urgency, anger, tier):
    base = {"cancellation": 2, "information": 0, "other": 0}.get(intent_class, 1)
    score = base + 2 * urgency + anger + (tier == "vip")
    return "high" if score >= 3 else "normal" if score >= 1 else "low"


@cat.rule("route")
def route(intent_class):
    return {"refund": "billing", "billing_question": "billing", "technical_help": "tech_support",
            "cancellation": "retention"}.get(intent_class, "general")


QUESTIONS = [
    Question("intent", "What does the customer want?", Answer.choice(INTENTS)),
    Question("urgent", "Is there time pressure?", Answer.yes_no()),
    Question("refund_requested", "Does the customer ask for money back?", Answer.yes_no()),
    Question("priority", "Priority", Answer.choice(["low", "normal", "high"]),
             checkpoints=["no_legal_threat", "no_chargeback_threat", "vip_sla_ok"]),
    Question("route", "Which queue?", Answer.choice(["billing", "tech_support", "retention", "legal", "general"]),
             checkpoints=["no_legal_threat"]),
]


# ---------- 200 synthetic labeled tickets -> the intent rule list (learned at import: a few ms)
ASK = {       # the intent-bearing phrase; everything around it is shared by all intents
    "refund": ["please refund {obj}", "I want my money back {obj}", "please reimburse me {obj}", "reverse the duplicate charge {obj}",
               "you charged me twice {obj}, I want it returned", "I am requesting a refund {obj}", "the trial converted and took money {obj}, refund it"],
    "technical_help": ["the app crashes {obj}", "sync is broken {obj}", "I get an error {obj}", "the dashboard will not load {obj}",
                       "exports fail {obj}", "login stopped working {obj}", "the integration is broken {obj}",
                       "I don't want a refund, just fix the bug {obj}"],
    "billing_question": ["why is the invoice higher {obj}?", "how do I update the card {obj}?", "can I get a VAT invoice {obj}?",
                         "what is this charge on my statement {obj}?", "can we switch to annual billing {obj}?",
                         "the billing address on the receipt is wrong {obj}", "can we pay the invoice by bank transfer {obj}?"],
    "information": ["do you offer a nonprofit discount {obj}?", "is there an on-premise version {obj}?", "where is the API documentation {obj}?",
                    "what is your refund policy {obj}?", "which regions host the data {obj}?", "does the plan include SSO {obj}?",
                    "what are your support hours {obj}?", "how do I invite teammates {obj}?"],
    "cancellation": ["please cancel the subscription {obj}", "I want to downgrade to free {obj}", "close the account {obj}",
                     "please don't renew {obj}", "we are switching to another tool, stop the renewal {obj}", "terminate the contract {obj}",
                     "cancel and refund the unused months {obj}"],
    "other": ["thanks for the great support {obj}", "I have feedback about the webinar", "who handles partnerships?",
              "I'm a journalist writing an article about you", "can I speak at your conference?", "just a note that my email changed"],
}
OBJ = ["", "", "for the {plan} plan", "on our company account", "for last month", "for order {order}", "for our workspace",
       "since {month}"]
CONTEXT = ["", "", "", "We have used the product for {n} years.", "I wrote last week as well.", "Following up on my earlier message.",
           "Account: {email}.", "I checked my card statement this morning."]
OPEN = ["", "Hi,", "Hello team,", "Hey there,", "Good morning,"]
TAIL = ["", "", "Thanks.", "Regards, Sam", "Cheers", "No rush.", "This is urgent.", "Our whole team is blocked right now.",
        "Really disappointed.", "Please advise."]


def make_ticket(rng, intent):
    f = dict(plan=rng.choice(["Pro", "Team", "Business"]), order=f"#{rng.randint(10000, 99999)}",
             month=rng.choice(["March", "July", "October"]), n=rng.randint(2, 6),
             email=rng.choice(["ops@kline.io", "sam@hollow.dev", "finance@orbit.co"]))
    ask = rng.choice(ASK[intent]).format(obj=rng.choice(OBJ).format(**f)).replace(" ?", "?").replace("  ", " ").strip()
    ask = ask[0].upper() + ask[1:] + ("" if ask[-1] in "?." else ".")
    return " ".join(x for x in (rng.choice(OPEN), rng.choice(CONTEXT).format(**f), ask, rng.choice(TAIL)) if x)


def make_tickets(n, seed):
    rng = random.Random(seed)
    return [({"message": make_ticket(rng, i)}, i) for i in (rng.choice(INTENTS) for _ in range(n))]


TRAIN = make_tickets(200, seed=7)
# the list System(cat, QUESTIONS).learn_rule("intent", TRAIN, ["cue_words"]) would learn, fitted on the one fact it reads
# (learn_rule first computes every fact of every example, which is slower in the browser)
INTENT_RULES = RuleList(["cue_words"], min_support=3, min_precision=0.8).fit(
    [{"cue_words": cue_words(s["message"])} for s, _ in TRAIN], [y for _, y in TRAIN])
