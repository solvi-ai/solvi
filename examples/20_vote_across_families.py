"""A vote of two model families: one guarantee, more answers given alone than either model gives by itself.

Two deciders of different families — say solvi-large and a System One model trained elsewhere — make different
mistakes. Where they agree, they are right far more often than either alone, so under the same guarantee a vote of the
two can answer more questions alone than the better model by itself (two models of one family — a teacher and its
student — agree on their mistakes too, and a vote of them gains nothing).

  1. two System One servers (stand-ins started in this process, one per family) become decision parts;
  2. each alone and their vote are calibrated with act_guard: P(answered alone and wrong) ≤ 10%;
  3. on new tickets: how much each answers alone, its error, the risk;
  4. the vote as a question of a catalog: a hard check, the audit (both proposals, the guarantee), the replay — and a
     sure mistake of one family that the other does not share: the vote escalates instead of answering.

Measured with real models (typed-decisions, 300 calibration questions, 200 random splits, risk 10%): a vote of solvi-large
and Julia 1 answered 50% of the questions alone, against 31% for solvi-large and 40% for Julia 1 each alone, at the same
risk (Julia's number there is in-distribution: it was trained on data like that set). The models agreed on 54% of the
questions and were right on 80% of those.

The servers here are keyword stand-ins that speak the System One HTTP API, so the example runs anywhere without a
network; point `systemone(URL, MODEL)` at real servers (or use `solvi.llm.llm(...)` for an LLM) and nothing else changes.

Run:  uv run python examples/20_vote_across_families.py"""
from __future__ import annotations

import json
import random
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from solvi import Catalog, System
from solvi.multi import Cascade, Vote
from solvi.systemone import systemone

TASK = "Which team should handle this ticket?"
TEAMS = {"billing": "payments, invoices, refunds", "technical": "bugs, errors, crashes",
         "shipping": "delivery, tracking, parcels", "account": "login, password, profile"}
PLAIN = {"billing": ["I was charged twice for order {n}.", "Invoice {n} shows the wrong amount.",
                     "Please refund the payment for order {n}."],
         "technical": ["The app crashes when I open settings.", "I get error {n} on upload.",
                       "The page freezes after the update."],
         "shipping": ["Where is parcel {n}? Tracking has not moved.", "Order {n} was delivered damaged.",
                      "The courier never came with order {n}."],
         "account": ["I cannot log in since yesterday.", "Reset my password please, account {n}.",
                     "My profile shows someone else's name."]}
HARD = [("The app charged me twice when it crashed on order {n}.", "billing"),
        ("After the update I cannot log in to track parcel {n}.", "account"),
        ("The refund page throws error {n}.", "technical"),
        ("My parcel {n} was sent to the old address in my profile.", "account"),
        ("The courier asked me to pay again for order {n}.", "billing"),
        ("The tracking link in my account crashes the app.", "technical"),
        ("I paid for express delivery but parcel {n} came late.", "shipping"),
        ("I changed my password and now the invoice page is blank.", "technical")]


def tickets(seed, n):
    """Support tickets: most name their team plainly, two in five mix two teams' words (only one of them is right)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if rng.random() < 0.4:
            t, team = rng.choice(HARD)
        else:
            team = list(TEAMS)[i % 4]
            t = rng.choice(PLAIN[team])
        out.append((t.format(n=rng.randint(1000, 9999)), team))
    return out


def _u(*key):
    """A deterministic number in [0, 1) per key."""
    return (zlib.crc32("|".join(map(str, key)).encode()) % 10007) / 10007


class Family:
    """How one model family reads a ticket: its keywords per team, noise of its own, and slips — on a share of tickets
    it is sure of a wrong team (as real models are: a sure mistake looks like a sure answer). Each family slips on
    different tickets: that is what a vote uses. Deterministic per text."""

    def __init__(self, name, keywords, noise, slips):
        self.name, self.keywords, self.noise, self.slips = name, keywords, noise, slips

    def probabilities(self, text, options):
        low = text.lower()
        z = []
        for o in options:
            hits = sum(low.count(k) for k in self.keywords.get(o, []))
            z.append(1.6 * hits + self.noise * (_u(self.name, o, text) - 0.5))
        if _u(self.name, "slip", text) < self.slips:
            best = int(np.argmax(z))
            wrong = [i for i in range(len(options)) if i != best]
            k = wrong[int(_u(self.name, "to", text) * len(wrong))]
            z[best], z[k] = z[k], z[best]            # as sure of the wrong team as it would have been of the right one
        e = np.exp(np.array(z) - max(z))
        return dict(zip(options, (e / e.sum()).tolist()))


FAMILY_A = Family("family-a", {"billing": ["charged", "invoice", "refund", "pay"], "technical": ["crash", "error", "freez",
                                                                                               "blank", "update"],
                               "shipping": ["parcel", "tracking", "courier", "deliver"],
                               "account": ["log in", "password", "profile", "account"]}, noise=1.5, slips=0.18)
FAMILY_B = Family("family-b", {"billing": ["charged", "invoice", "refund", "paid", "pay again"],
                               "technical": ["crash", "error", "freez", "throws", "blank"],
                               "shipping": ["parcel", "courier", "late", "express", "deliver"],
                               "account": ["log in", "password", "profile", "address"]}, noise=1.8, slips=0.12)


def serve(family):
    """A stand-in System One server for one family, on a free local port → its URL (it stops with the process)."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            answers = {}
            for name, q in body["questions"].items():
                p = family.probabilities(body["state"], list(q["criteria"]))
                answers[name] = {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
            out = json.dumps({"model": body["model"], "answers": answers}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def on_new(comb, test):
    """On new tickets: answered alone, error among them, P(answered alone and wrong)."""
    ds = comb.decide([x for x, _ in test])
    auto = np.array([d.escalate is None for d in ds])
    wrong = np.array([d.value != y for d, (_, y) in zip(ds, test)])
    return auto.mean(), (wrong[auto].mean() if auto.any() else 0.0), (auto & wrong).mean()


if __name__ == "__main__":
    a = systemone(serve(FAMILY_A), "family-a").decision("team", TASK, "ticket", TEAMS)     # api_key=... for a hosted one
    b = systemone(serve(FAMILY_B), "family-b").decision("team", TASK, "ticket", TEAMS)
    calib, test = tickets(1, 300), tickets(2, 600)

    print("=== each model alone and the vote, calibrated to P(answered alone and wrong) ≤ 10% ===")
    rows = [("family A alone", Cascade([a], name="team")), ("family B alone", Cascade([b], name="team")),
            ("vote A + B (both agree)", Vote([a, b], rule="all", name="team"))]
    for label, comb in rows:
        info = comb.act_guard(calib, max_risk=0.10)
        got, err, risk = on_new(comb, test)
        print(f"  {label:25s} answered alone {got:4.0%}, error among them {err:5.1%}, risk {risk:5.1%}   "
              f"(threshold {info['threshold']:.2f})")
    vote = rows[2][1]
    agree = np.mean([a.decide(x).value == b.decide(x).value for x, _ in test])
    right = np.mean([a.decide(x).value == y for x, y in test if a.decide(x).value == b.decide(x).value])
    print(f"  the families agree on {agree:.0%} of new tickets and are right on {right:.0%} of those")

    print("\n=== the vote in a catalog: a hard check first, then the audit and the replay ===")
    cat = Catalog()

    @cat.check(hard=True, then={"team": "account"})
    def not_locked(ticket: str) -> bool:
        """A ticket about a locked account goes to the account team, whatever the models say."""
        return "locked" not in ticket.lower()
    q = vote.question(cat, "team", "Which team handles the ticket?", requires=["not_locked"])
    system = System(cat, [q])
    for ticket in ("I was charged twice for order 4409, please refund the payment.",     # both sure, both billing
                   "I was charged twice for order 4411, please refund the payment.",     # family B slips, sure of it
                   "My account is locked and the app crashes."):                         # the hard check decides
        res = system.ask({"ticket": ticket})
        r = res["team"]
        print(f"\n{ticket!r}\n  → {r.answer} ({r.status})")
        print(res.audit("team"))
        print(f"  replay: {res.trace.replay(cat)['ok']}")
