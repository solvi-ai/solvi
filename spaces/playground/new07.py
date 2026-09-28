"""The "New in 0.7" tab: escalation with a guarantee (act_guard), a vote of two model families, text in (a message → the
question it asks and its fields, each with a quote) and reports for people.

NO MODEL RUNS HERE. Every decider is a keyword stand-in with the decider's contract (one logit per option); the numbers
show the mechanics, not the quality of any model. With a real checkpoint, `DecideModel.load("<folder or HF id>")` takes
the stand-in's place and nothing else changes.

The Space installs solvi from PyPI, so the version in the browser may be older than the features shown here. Every
feature is detected before use; a demo that needs a newer solvi says so instead of failing. Pure Python, no gradio."""
from __future__ import annotations

import datetime as dt
import html
import inspect
import random
import zlib
from typing import Literal

import numpy as np

from solvi import Catalog, Question, System
from solvi.decide import DecideModel


def solvi_version():
    try:
        from importlib.metadata import version
        return version("solvi")
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------------------------------------- feature detection
def has_act_guard():
    from solvi.decide import DecisionPart
    return hasattr(DecisionPart, "act_guard")


def has_groups():
    from solvi.decide import DecisionPart
    return has_act_guard() and "groups" in inspect.signature(DecisionPart.act_guard).parameters


def has_vote():
    try:
        from solvi.multi import Vote  # noqa: F401
        return True
    except ImportError:
        return False


def has_textin():
    try:
        from solvi.textin import CueExtractor, TextIn  # noqa: F401
        return hasattr(System, "ask_text")
    except ImportError:
        return False


def has_report():
    from solvi import Response
    return hasattr(Response, "report")


def needs(feature, version):
    return (f"**This demo needs solvi {version} or newer ({feature}); this Space loaded solvi {solvi_version()}.** "
            "The Space installs the newest solvi from PyPI when the page loads; reload after the release, or run the same "
            "code locally with `pip install -U solvi`.")


# ---------------------------------------------------------------------------------------------- stand-in deciders
TASK = "Which team should handle this support email?"
TEAMS = {"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
         "shipping": "delivery, tracking, parcels"}
CORE = {"billing": ["I was charged twice for order {n}, please refund one payment.", "My invoice {n} shows the wrong amount.",
                    "Why did my card get billed again? I cancelled.", "The payment failed but the money left my account."],
        "technical": ["The app crashes when I open the settings.", "I get error {n} when I upload a photo.",
                      "The page freezes after the last update.", "The search button does nothing, I see an error."],
        "shipping": ["Where is my parcel {n}? The tracking has not moved.", "My order {n} was delivered damaged.",
                     "The courier says delivered but no package arrived.", "My delivery is three days late."]}
MIXED = [("The app crashed while I paid, and now I was charged twice for order {n}.", "billing"),
         ("My parcel {n} never arrived and the refund page shows an error.", "shipping"),
         ("The tracking page crashes every time; where is my delivery?", "shipping"),
         ("I was billed for order {n} but the courier lost the package.", "shipping")]


def dataset(seed, n):
    """Labelled support emails: most name their team plainly, one in four mixes two teams' words (the hard ones)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 4 == 3:
            text, team = rng.choice(MIXED)
        else:
            team = list(TEAMS)[i % 3]
            text = rng.choice(CORE[team])
        out.append((text.format(n=rng.randint(1000, 9999)), team))
    return out


class KeywordFamily:
    """A keyword stand-in for a decider: a logit per option from its keywords, a label bias and deterministic noise of its
    own. Two instances with different keywords and noise play two model families with different mistakes."""
    KEYWORDS = {"billing": ["charged", "refund", "invoice", "payment", "billed", "paid", "money"],
                "technical": ["crash", "error", "freez", "button", "update", "app"],
                "shipping": ["parcel", "tracking", "deliver", "courier", "package", "arrive"]}

    def __init__(self, name, weight, noise, bias=None, favour=None):
        self.model_id, self.weight, self.noise, self.bias = name, weight, noise, dict(bias or {})
        self.favour = favour or {}

    def fingerprint(self):
        return f"{self.model_id}-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = []
            for o in it.options:
                hits = sum(low.count(k) for k in self.KEYWORDS.get(o, [])) + sum(low.count(k) for k in self.favour.get(o, []))
                noise = (zlib.crc32((self.model_id + o + it.text).encode()) % 1000) / 1000 - 0.5
                z.append(self.weight * hits + self.bias.get(o, 0.0) + self.noise * noise)
            out.append(np.array(z))
        return out


def stand_in(name, **kw):
    return DecideModel(KeywordFamily(f"stand-in/{name}", **kw), meta={"format": "stand-in", "temperature": 1.0})


FAMILY_A = dict(weight=2.4, noise=2.0)
SMALL = dict(weight=1.6, noise=4.0, bias={"technical": 0.6})     # a weaker stand-in: often unsure on the mixed emails
FAMILY_B = dict(weight=2.0, noise=3.0, favour={"shipping": ["lost", "late"], "billing": ["cancel"]})


def _pct(x):
    return f"{100 * float(x):.1f}%"


def _on_new(part, test):
    """Answered share, error among the answered and P(answered alone and wrong) on new labelled emails."""
    ds = part.decide([x for x, _ in test]) if hasattr(part, "decide") and not hasattr(part, "model") else \
        [part(x) for x, _ in test]
    auto = np.array([d.escalate is None for d in ds])
    wrong = np.array([d.value != y for d, (_, y) in zip(ds, test)])
    return auto.mean(), (wrong[auto].mean() if auto.any() else 0.0), (auto & wrong).mean()


def _report_parts(res, question):
    """(markdown, html) of res.report when this solvi has it, else a note."""
    if not has_report():
        return needs("res.report", "0.7"), ""
    md = res.report(format="md", question=question)
    page = res.report(format="html", question=question)
    frame = (f'<iframe title="decision report" style="width:100%;height:640px;border:1px solid #cbd5e1;border-radius:8px" '
             f'srcdoc="{html.escape(page, quote=True)}"></iframe>')
    return md, frame


# ---------------------------------------------------------------------------------------------- 1. act_guard
def demo_guard(email, risk=0.10):
    """Calibrate one stand-in decider on 300 labelled emails with act_guard, check the promise on 1,200 new ones, then ask
    one email in a catalog: the audit's guarantee line says what the threshold promises."""
    if not has_act_guard():
        return needs("act_guard", "0.5.1"), "", "", ""
    risk = float(risk)
    model = stand_in("small", **SMALL)
    part = model.decision("team", TASK, "email", TEAMS)
    calib, test = dataset(1, 300), dataset(2, 1200)
    info = part.act_guard(calib, risk=risk)
    a, e, r = _on_new(part, test)
    lines = [f"**Escalation with a guarantee** — `part.act_guard(examples, risk={risk:g})` on {info['n']} labelled emails "
             f"(synthetic; one in four mixes two teams' words). The promise: P(answered alone **and** wrong) ≤ {risk:g}, as a "
             "share of all emails, for emails like the examples.",
             "",
             "| | answered alone | error among them | answered and wrong (the risk) |",
             "|---|---|---|---|",
             f"| the 300 labelled emails | {_pct(info['answered'])} | {_pct(info['error'])} | {_pct(info['risk'])} |",
             f"| 1,200 new emails | {_pct(a)} | {_pct(e)} | {_pct(r)} |",
             "",
             f"Threshold on the model's {info.get('signal', 'signal')}: **{float(info['threshold']):.3f}**. "
             f"`must_escalate_at_least` = {_pct(info.get('must_escalate_at_least', 0.0))}: when the model is wrong on a "
             "share of the examples above the risk, any threshold has to hand at least this share to a person."]
    cat = Catalog()
    q = part.question(cat, "team", "Which team handles the email?")
    res = System(cat, [q]).ask({"email": email})
    t = res["team"]
    lines += ["", f"**This email:** team = **{t.answer if t.status != 'abstain' else '—'}** ({t.status}"
                  + (f", {t.guard}" if t.guard else "") + f") — {t.why}"]
    if not has_groups():
        lines += ["", "_Thresholds per group (`act_guard(..., groups=...)`) need solvi 0.7; this Space loaded "
                      f"{solvi_version()}._"]
    lines += ["", "_Stand-in decider: keyword logits, not a model. The promise is on average over calibration sets and "
                  "holds for inputs like the calibration examples: one set of new emails can land a little above it, and a "
                  "new domain needs new labels._"]
    md, frame = _report_parts(res, "team")
    return "\n".join(lines), str(res.audit("team")), md, frame


# ---------------------------------------------------------------------------------------------- 2. Vote of two families
def demo_vote(email, risk=0.10):
    """Two stand-in "families" answer the same question; Vote(rule="all") answers only when both agree and both are sure,
    else it escalates with both proposals. One act_guard for the vote as a whole."""
    if not has_vote():
        return needs("solvi.multi.Vote", "0.6"), "", "", ""
    from solvi.multi import Vote
    risk = float(risk)
    a_part = stand_in("family-a", **FAMILY_A).decision("team", TASK, "email", TEAMS)
    b_part = stand_in("family-b", **FAMILY_B).decision("team", TASK, "email", TEAMS)
    vote = Vote([a_part, b_part], rule="all")
    calib, test = dataset(1, 300), dataset(2, 600)
    lines = ["**A vote of two model families** — `Vote([a, b], rule=\"all\")`: an answer only when both propose the same "
             "team and each is sure; otherwise the email escalates with both proposals listed.", ""]
    rows = []
    for name, p in (("family A alone", a_part), ("family B alone", b_part)):
        one = Vote([p], rule="all")
        if hasattr(one, "act_guard"):
            one.act_guard(calib, risk=risk)
        rows.append((name, *_on_new(one, test)))
    if hasattr(vote, "act_guard"):
        vote.act_guard(calib, risk=risk)
    rows.append(("vote A + B", *_on_new(vote, test)))
    lines += [f"Each calibrated with `act_guard(risk={risk:g})` on 300 labelled emails, measured on 600 new ones:", "",
              "| | answered alone | error among them | answered and wrong |", "|---|---|---|---|"]
    lines += [f"| {n} | {_pct(a)} | {_pct(e)} | {_pct(r)} |" for n, a, e, r in rows]
    d = vote.decide(email)
    props = []
    for v in (d.extra or {}).get("votes", []):
        props.append(f"{v.get('model', v.get('part', '?'))}: `{v.get('value')}`"
                     + (f" (escalated: {v['escalate']})" if v.get("escalate") else ""))
    lines += ["", "**This email:** " + (f"**{d.value}** (both agree)" if d.escalate is None else f"escalated — {d.escalate}")]
    if props:
        lines += ["", "Proposals: " + "; ".join(props)]
    lines += ["", "_Two keyword stand-ins with different keywords and noise play two families; their mistakes differ, "
                  "which is what makes a vote useful. Two models where one is the other's student make the same mistakes, "
                  "and a vote of them helps little._"]
    cat = Catalog()
    q = vote.question(cat, "team", "Which team handles the email?")
    res = System(cat, [q]).ask({"email": email})
    md, frame = _report_parts(res, "team")
    return "\n".join(lines), str(res.audit("team")), md, frame


# ---------------------------------------------------------------------------------------------- 3. Text in
TODAY = dt.date(2026, 9, 28)
ROUTES = {"request_refund": ["refund", "money back", "charged"], "change_delivery_address": ["address", "deliver"],
          "cancel_order": ["cancel"]}


class RouteScorer:
    """A keyword stand-in that picks the entry point (3 per keyword of that request found in the text)."""
    model_id = "stand-in/route-scorer"

    def fingerprint(self):
        return "route-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = np.array([3.0 * sum(low.count(k) for k in ROUTES.get(o, [])) for o in it.options])
            out.append(np.stack([z, z - 1.0], 1))
        return out


def shop():
    """A shop desk with three requests; each one's input fields come from the types its functions read."""
    cat = Catalog()

    @cat.check(hard=True, then={"request_refund": "reject", "cancel_order": "keep", "change_delivery_address": "refused"})
    def order_exists(order_id: str) -> bool:
        """Orders are A-<digits>."""
        return order_id.startswith("A-")

    @cat.fn
    def days_since_purchase(purchase_date: dt.date) -> int:
        return (TODAY - purchase_date).days

    @cat.fn
    def amount_eur(amount: float, currency: Literal["EUR", "USD", "RUB"]) -> float:
        return amount * {"EUR": 1.0, "USD": 0.9, "RUB": 0.01}[currency]

    @cat.rule("request_refund")
    def refund(order_exists: bool, days_since_purchase: int, amount_eur: float) -> Literal["approve", "review", "reject"]:
        if days_since_purchase > 30:
            return "reject"
        return "review" if amount_eur > 1000 else "approve"

    @cat.rule("change_delivery_address")
    def change(order_exists: bool, new_address: str) -> Literal["changed", "refused"]:
        return "changed" if len(new_address) > 5 else "refused"

    @cat.rule("cancel_order")
    def cancel(order_exists: bool, urgent: bool) -> Literal["cancelled", "keep"]:
        return "cancelled" if urgent else "keep"

    qs = [Question("request_refund", "Refund an order"), Question("change_delivery_address", "Change where an order is delivered"),
          Question("cancel_order", "Cancel an order")]
    return cat, System(cat, qs)


def demo_textin(message, risk=None):
    """A message → the question it asks (a stand-in decider picks the entry point) and each input field read with a quote
    (CueExtractor: deterministic candidates of the field's type after a cue word) → ask_text answers it in one trace."""
    if not has_textin():
        return needs("solvi.textin.TextIn and system.ask_text", "0.7"), "", "", ""
    from solvi.textin import CueExtractor, TextIn
    _, system = shop()
    decider = DecideModel(RouteScorer(), meta={"format": "stand-in", "temperature": 1.0})
    tin = TextIn(system, decider, CueExtractor(), today=TODAY, patterns={"order_id": r"[A-Z]-\d+"},
                 synonyms={"currency": {"EUR": ["euro", "euros", "€"], "RUB": ["rubles", "руб", "₽"],
                                        "USD": ["dollars", "$"]}})
    read = tin.read(message)
    lines = ["**Text in** — `TextIn(system, decider).read(message)`: the decider picks which request the message makes, "
             "each field is read with a quote and a deterministic parser per type, and `system.ask_text(read)` answers it.",
             "", f"Entry point: **{read.question or '— (escalated: no request chosen)'}**"]
    route = read.route or {}
    if isinstance(route, dict) and route.get("probs"):
        lines[-1] += " · " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(route["probs"].items(), key=lambda kv: -kv[1]))
    if read.question:
        lines += ["", "| field | state | value | quote |", "|---|---|---|---|"]
        for name, f in read.fields.items():
            q = f.quote
            quote = f"`{message[q.start:q.end]}` [{q.start}:{q.end}]" if q is not None and q.end > q.start else "—"
            val = f"`{f.value}`" if f.ok else (f.why or "")
            lines.append(f"| {name}{' (required)' if f.required else ''} | {f.status} | {val} | {quote} |")
    if read.missing:
        lines += ["", f"Missing: {', '.join(read.missing)} → the clarifying question: _{read.clarify()}_"]
    res = system.ask_text(read)
    lines += ["", "**Answers:**"]
    for q, r in res.results.items():
        if r.status == "abstain" and read.question and q != read.question:
            continue
        lines.append(f"- {q} = **{r.answer if r.status != 'abstain' else '—'}** ({r.status}) — {r.why}")
    lines += ["", "_The routing decider is a keyword stand-in; the fields are read by plain code (CueExtractor), still "
                  "recorded as read by a model: which number is \"the amount\" is a guess, so it is not counted as given. "
                  "Nothing in the message is executed: it can only pick one of the questions and fill typed fields._"]
    question = read.question or next(iter(res.results))
    md, frame = _report_parts(res, question)
    return "\n".join(lines), str(res.audit(question)), md, frame


# ---------------------------------------------------------------------------------------------- 4. a report
def demo_report(email, risk=0.10):
    """The report of one decision for an auditor or a customer: the guarantee demo's decision as a page."""
    if not has_report():
        return needs("res.report", "0.7"), "", "", ""
    _, audit, md, frame = demo_guard(email, risk)
    lead = ("**A report for people** — `res.report(format=\"md\" | \"html\" | \"data\")`: each answer, what it rests on, "
            "the safeguards that fired, the guarantee line, the models with their fingerprints, the trace's hashes and "
            "the replay status (no model is called). Below: the Markdown and the self-contained HTML page of the "
            "decision from the guarantee demo.")
    return lead, audit, md, frame


DEMOS = {
    "Escalation with a guarantee (act_guard)": (demo_guard, "I was charged twice for order 5521, please refund one payment."),
    "Vote of two model families": (demo_vote, "My parcel 7710 never arrived and the refund page shows an error."),
    "Text in: a message → question + fields with quotes": (
        demo_textin, "Hi, please refund order A-10457: I paid 1.5 million rubles on 12 September and it arrived broken."),
    "Report for people (res.report)": (demo_report, "The app crashed while I paid, and now I was charged twice for order 8812."),
}


def run(name, text, risk=0.10):
    fn, default = DEMOS[name]
    try:
        return fn(text or default, risk)
    except Exception as e:  # noqa: BLE001 — a demo error is shown, not raised into the UI
        return (f"**This demo failed on solvi {solvi_version()}:** `{type(e).__name__}: {e}`", "", "", "")
