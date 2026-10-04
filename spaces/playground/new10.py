"""The "New in 1.0" tab of the playground: live demos of what solvi 1.0 added, each a few hundred milliseconds in the
browser and none with a model.

- build: `solvi.build` from labelled examples (the five-line path) and `explain()`;
- checks: `res.checks` as a table (name, result, reason, hard, then) and `then=` as a function of facts;
- quotes: a quote typed otherwise (no-break spaces, curly quotes, "…") accepted, the source's own text kept; a made-up
  quote refused; the same with `Catalog(quotes="literal")`, 0.9's rule;
- journal: bytes per decision in a full and a compact journal, both verified and replayed;
- levels: what `from solvi import ...` and `from solvi.core import ...` give.

Each demo returns Markdown (and a table where there is one); an error is shown, not raised into the UI."""
from __future__ import annotations

import os
import random
import tempfile
import time
import warnings

import pandas as pd


def solvi_version():
    try:
        from importlib.metadata import version
        return version("solvi")
    except Exception:  # noqa: BLE001
        return "?"


def _needs_1_0():
    """A note when the solvi this page loaded predates 1.0 (the demos use 1.0's API), else None."""
    import solvi
    if not hasattr(solvi, "Agent") or not hasattr(solvi, "build"):
        return f"These demos need solvi 1.0 or newer; this page loaded solvi {solvi_version()}."
    return None


def _guard(fn):
    """A demo error is shown in the tab (with the solvi version), never raised into the UI."""
    def run(*args):
        note = _needs_1_0()
        if note:
            return _empty(fn, f"**{note}**")
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            return _empty(fn, f"**This demo failed on solvi {solvi_version()}:** `{type(e).__name__}: {e}`")
    run.__name__ = fn.__name__
    return run


def _empty(fn, md):
    n = {"build_demo": 3, "checks_demo": 3, "quotes_demo": 2, "journal_demo": 2, "levels_demo": 1}[fn.__name__]
    return md if n == 1 else (md,) + (pd.DataFrame(),) * (n - 1)


def _ms(t0):
    return f"{(time.perf_counter() - t0) * 1000:.0f} ms"


# ====================================================================================================== build
BUILD_CODE = '''import solvi
from solvi import Answer, Question

s = solvi.build(Question("refund", "Refund without asking a person?", Answer.yes_no()),
                examples, catalog=cat, max_risk=0.02, slow=notes_reader)   # examples: [(state, answer)]
print(s.explain())
res = s.ask(state)          # res.answer, res.by ("s1", "s2" or "human"), res.reasons'''


def _refund_catalog():
    from solvi import Catalog
    cat = Catalog()

    @cat.fn
    def refund_share(amount, order_total) -> float:          # how much of the order is asked back
        return amount / order_total

    @cat.fn
    def delivered_late(days_late) -> bool:
        return days_late > 2
    return cat


def notes_reader(state):
    """The slow path (an LLM in practice): it also reads the agent's free-text notes."""
    share = state["amount"] / state["order_total"]
    return "yes" if state["days_late"] > 2 or share < 0.4 or (share < 0.7 and "loyal" in state["notes"]) else "no"


def _request(rng, n):
    total = round(rng.uniform(20, 400), 2)
    state = {"amount": round(total * rng.uniform(0.05, 1.0), 2), "order_total": total,
             "days_late": rng.choice([0, 0, 0, 1, 3, 6]),
             "notes": f"order {n}: customer since {rng.randint(2012, 2026)}, " + rng.choice(["loyal", "new", "returning"])}
    return state, notes_reader(state)


@_guard
def build_demo(n_examples=1200, max_risk=0.02):
    """→ (markdown with explain(), the first decisions as a table, who answered 200 more)."""
    import solvi
    from solvi import Answer, Question
    n_examples, max_risk = int(n_examples), float(max_risk)
    rng = random.Random(7)
    examples = [_request(rng, n) for n in range(n_examples)]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = solvi.build(Question("refund", "Refund without asking a person?", Answer.yes_no()), examples,
                        catalog=_refund_catalog(), max_risk=max_risk, slow=notes_reader)
    built = _ms(t0)
    rows = []
    for state, truth in [_request(rng, n) for n in range(5000, 5008)]:
        res = s.ask(state)
        rows.append({"amount / order": f"{state['amount']:.0f} / {state['order_total']:.0f}",
                     "days late": state["days_late"], "notes": state["notes"].split(", ")[-1],
                     "answer": res.answer, "by": res.by, "right": "yes" if res.answer == truth else "no",
                     "reason": res.reasons[-1] if res.reasons else ""})
    t1 = time.perf_counter()
    more = [(s.ask(st), truth) for st, truth in [_request(rng, n) for n in range(6000, 6200)]]
    asked = _ms(t1)
    by = {k: sum(r.by == k for r, _ in more) for k in ("s1", "s2", "human")}
    alone_wrong = sum(r.by == "s1" and r.answer != t for r, t in more)
    md = (f"**solvi.build** on {n_examples} labelled examples, `max_risk={max_risk}`: built in {built} (solvi "
          f"{solvi_version()}, in this tab). Then 200 new requests in {asked}: **System 1 alone {by['s1']}**, the slow "
          f"path {by['s2']}, a person {by['human']}; System 1 alone and wrong: {alone_wrong} of 200 (the promise: "
          f"at most {max_risk:.0%} of inputs like the calibration examples).\n\n"
          f"```python\n{BUILD_CODE}\n```\n\n**`s.explain()`** — what it chose and why:\n\n```text\n{s.explain()}\n```")
    return md, pd.DataFrame(rows), pd.DataFrame([{"answered by": k, "of 200": v} for k, v in by.items()])


# ====================================================================================================== checks
CHECKS_CODE = '''def review_or_part(amount: float, limit: float) -> str:     # then= as a function of facts
    return "partial" if amount <= 2 * limit else "person"

@cat.check(hard=True, then={"refund": review_or_part})
def within_limit(amount: float, limit: float) -> bool:
    return Fail(f"{amount:g} is over the desk's limit of {limit:g}") if amount > limit else True

@cat.check(hard=True, then={"refund": "no"})
def known_customer(customer: str) -> bool:
    return Fail(f"{customer} is on the blocked list") if customer in BLOCKED else True

res.checks   # [CheckResult(name, questions, status, hard, reason, then), ...] in flow order'''


def _desk():
    from solvi import Answer, Catalog, Fail, Question, System
    cat = Catalog()
    blocked = {"mallory"}

    @cat.fn
    def days(purchase_day: int, today: int) -> int:
        return today - purchase_day

    @cat.check
    def recent(days: int) -> bool:
        "bought in the last 30 days"
        return days <= 30

    def review_or_part(amount: float, limit: float) -> str:
        return "partial" if amount <= 2 * limit else "person"

    @cat.check(hard=True, then={"refund": review_or_part})
    def within_limit(amount: float, limit: float) -> bool:
        return Fail(f"{amount:g} is over the desk's limit of {limit:g}") if amount > limit else True

    @cat.check(hard=True, then={"refund": "no"})
    def known_customer(customer: str) -> bool:
        return Fail(f"{customer} is on the blocked list") if customer in blocked else True

    @cat.check(hard=True)
    def has_receipt(receipt: str) -> bool:
        "a receipt number starting with R"
        return receipt.startswith("R")

    @cat.rule("refund")
    def refund(recent: bool, has_receipt: bool) -> str:
        return "full" if recent else "no"
    return System(cat, [Question("refund", "Refund?", Answer.choice(["full", "partial", "no", "person"]))])


@_guard
def checks_demo(amount=80.0, limit=100.0, days_ago=12, customer="ann", receipt="R-1042"):
    """→ (markdown, res.checks as a table, the answer row)."""
    s = _desk()
    state = {"amount": float(amount), "limit": float(limit), "purchase_day": 100 - int(days_ago), "today": 100,
             "customer": str(customer or ""), "receipt": str(receipt or "")}
    t0 = time.perf_counter()
    res = s.ask(state)
    took = _ms(t0)
    rows = [{"check": c.name, "result": c.status, "reason": c.reason or "", "hard": "hard" if c.hard else "soft",
             "then": ", ".join(f"{q} = {v}" for q, v in (c.then or {}).items())} for c in res.checks]
    rep = res.trace.replay(s, res.flow)
    a = res["refund"]
    then_rec = [r for r in res.trace.records if r.kind == "then"]
    md = (f"**refund = `{a.answer}`** ({a.status}) — {a.why}\n\n"
          f"Decided in {took}; the trace replays: **{'OK' if rep['ok'] else 'MISMATCH'}**"
          + (f"; the `then` function's value is a record of the trace (`{then_rec[0].name}`) and replay re-ran it."
             if then_rec else ".")
          + f"\n\n```python\n{CHECKS_CODE}\n```")
    return md, pd.DataFrame(rows), pd.DataFrame([{"question": "refund", "answer": a.answer, "status": a.status,
                                                   "why": a.why}])


CHECK_CASES = {
    "all checks pass": (80, 100, 12, "ann", "R-1042"),
    "over the limit: then computes 'partial'": (150, 100, 12, "ann", "R-1042"),
    "far over the limit: then computes 'person'": (450, 100, 12, "ann", "R-1042"),
    "blocked customer: then = 'no'": (80, 100, 12, "mallory", "R-1042"),
    "no receipt: a hard check without then": (80, 100, 12, "ann", "none"),
    "bought 45 days ago: the soft check fails": (80, 100, 45, "ann", "R-1042"),
}


# ====================================================================================================== quotes
NOTES = ("Call log, 14 March: the customer said the parcel was “left at the door” and the box was crushed. "
         "Refund promised within 10 days — pending the photo… Agent: B. Ortiz.")
QUOTE_CASES = {
    "typed otherwise (no-break spaces, straight quotes, '...', a plain hyphen)":
        "the parcel was \"left at the door\" and the box was crushed",
    "with '...' for '…' and '-' for '—'": "Refund promised within 10 days - pending the photo...",
    "exact (copied from the notes)": "left at the door",
    "made up (not in the notes)": "the customer asked for a full refund",
}
QUOTES_CODE = '''@cat.check
def quoted(notes: str) -> bool:                  # the evidence a model wrote, checked against the notes
    return Claim(True, evidence=[model_quote])

Catalog()                     # 1.0: matched on a normalized view; the source's own text is what is kept
Catalog(quotes="literal")     # 0.9's rule'''


def _quote_system(written, quotes):
    from solvi import Answer, Catalog, Claim, Question, System
    cat = Catalog(quotes=quotes)

    @cat.check
    def quoted(notes: str) -> bool:
        "the model's quote is in the notes"
        return Claim(True, evidence=[written])

    @cat.rule("grounded")
    def grounded(quoted: bool) -> bool:
        return quoted
    return System(cat, [Question("grounded", "Is the model's claim backed by the notes?", Answer.yes_no())])


def _show(s):
    """A string with its invisible characters made visible."""
    return (s.replace(" ", "⍽").replace(" ", "⍽").replace("‑", "‑"))


@_guard
def quotes_demo(written, notes=NOTES):
    """→ (markdown, a table: one row per quote rule)."""
    from solvi.core import find_quote
    written, notes = str(written or ""), str(notes or "")
    rows = []
    for mode in ("normalized", "literal"):
        s = _quote_system(written, mode)
        res = s.ask({"notes": notes})
        rec = res.trace.records[0]
        ev = [e for e in (rec.extra or {}).get("evidence") or [] if e[0] >= 0]     # (-1, -1): not found
        qm = (rec.extra or {}).get("quote_match")
        a = res["grounded"]
        rows.append({"Catalog(quotes=…)": f'"{mode}"' + (" (1.0 default)" if mode == "normalized" else " (0.9 rule)"),
                     "answer": a.answer if a.status != "abstain" else "abstain",
                     "kept quote (the notes' own text)": ev[0][3] if ev else "",
                     "offsets": f"{ev[0][0]}–{ev[0][1]}" if ev else "",
                     "record says": f"needed the normalized view ({qm['form']})" if qm else (
                         "literal match" if ev else "refused: " + (a.why or "not grounded")[:140]),
                     "replay": "OK" if res.trace.replay(s, res.flow)["ok"] else "MISMATCH"})
    q = find_quote(written, {"notes": notes})
    md = (f"The model wrote: `{_show(written)}` (⍽ marks a no-break space).\n\n"
          + (f"`find_quote` finds it in **{q.source}** at {q.start}–{q.end}: the notes say “{q.value}” — that, not what "
             "the model typed, is what a decision stores." if q else
             "`find_quote` finds it in no source: **a made-up quote is refused** under both rules.")
          + f"\n\n```python\n{QUOTES_CODE}\n```")
    return md, pd.DataFrame(rows)


# ====================================================================================================== journal
@_guard
def journal_demo(n=100):
    """Ask the refund desk n times into a full and a compact JSON-lines journal → (markdown, a table)."""
    from solvi.core.store import JSONLStorage
    n = int(n)
    rng = random.Random(3)
    states = [{"amount": float(rng.choice([40, 80, 120, 260, 520])), "limit": 100.0,
               "purchase_day": 100 - rng.choice([3, 12, 25, 40]), "today": 100,
               "customer": rng.choice(["ann", "bob", "mallory"]), "receipt": rng.choice(["R-1", "R-77", "none"])}
              for _ in range(n)]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for record in ("full", "compact"):
            s = _desk()
            path = os.path.join(tmp, f"{record}.jsonl")
            s.storage = JSONLStorage(path, record=record)
            s.storage.catalog = s
            t0 = time.perf_counter()
            for st in states:                                  # a System with a storage stores every decision

                s.ask(st)
            took = (time.perf_counter() - t0) * 1000
            size = os.path.getsize(path)
            v = s.storage.verify()
            t1 = time.perf_counter()
            bad = s.storage.replay_all(s)
            rtook = (time.perf_counter() - t1) * 1000
            unverified = sum(set(b.get("kinds") or {}) == {"not_kept"} for b in bad)
            rows.append({"record=": f'"{record}"', "decisions": v["count"], "bytes per decision": round(size / n),
                         "verify()": "OK" if v["ok"] else f"FAIL {v['problems'][:1]}",
                         "replayed OK": n - len(bad), "not verified": unverified, "mismatch": len(bad) - unverified,
                         "ask ms / decision": round(took / n, 2), "replay ms / decision": round(rtook / n, 2)})
    full, comp = rows[0]["bytes per decision"], rows[1]["bytes per decision"]
    nv = rows[1]["not verified"]
    md = (f"{n} decisions of the refund desk (the Checks demo's catalog), stored twice. **Compact: {comp:,} bytes a "
          f"decision vs {full:,} full ({full / comp:.1f}× smaller).** A compact record keeps the input, the answers, "
          "each check's result and reason, and every step's hash — not the computed values or the timings; "
          "`replay_all` re-runs it from its input and compares every step's hash, and says *not verified* where it "
          "cannot check, instead of passing"
          + (f" — here {nv} compact decisions where the `then=` function fired (over the limit): its step is recorded "
             "after the flow and a compact record of solvi 1.0.0 does not keep it, so replay reports them as not "
             "verified rather than as passed. solvi 1.0.1 keeps that step in compact records and re-runs the `then=` "
             "function on replay, so there they replay OK; on 1.0.0 keep such decisions full (`record=\"sample:N\"` "
             "or a full store) when you must re-check them" if nv else "") + ".\n\n"
          "```python\nstore = JSONLStorage(\"desk.jsonl\", record=\"compact\")   # or \"full\" (default), \"sample:N\"\n"
          "store.verify(); store.replay_all(system); store.rederive(id, system)\n```")
    return md, pd.DataFrame(rows)


# ====================================================================================================== levels
@_guard
def levels_demo():
    import solvi
    import solvi.core as core
    exp = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import solvi.experimental as ex
            exp = list(ex.STATUS)
    except Exception:  # noqa: BLE001 — an older solvi without solvi.experimental: the line is left out
        exp = []
    entry = [n for n in solvi.__all__ if n in ("build", "Agent", "Guard", "Knowledge")]
    vocab = [n for n in solvi.__all__ if n not in entry]
    protos = [n for n in core.__all__ if n in ("Scorer", "Decider", "Adapter", "Head", "Extractor", "Strategist",
                                                 "Monitor", "Proposer", "Space", "Environment", "TraceStorage",
                                                 "SlowPath", "ActionModel", "WriteGate", "RiskPolicy")]
    return (f"**Two levels** (computed from the solvi {solvi_version()} this tab loaded).\n\n"
            f"**`from solvi import …`** — the high level, {len(solvi.__all__)} names: the ready systems "
            f"{', '.join(f'`{n}`' for n in entry)}, and the shared vocabulary {', '.join(f'`{n}`' for n in vocab)}.\n\n"
            f"**`from solvi.core import …`** — the building blocks they are made of, {len(core.__all__)} names; the "
            f"places to put a part of your own: {', '.join(f'`{n}`' for n in protos)} (each with a conformance check "
            "in `solvi.testing.conformance`).\n\n"
            + (f"**`solvi.experimental`** — works, no measured gain yet; warns on import, marked in every decision "
               f"that uses it, graduates or goes in 1.2: {', '.join(f'`{n}`' for n in exp)}.\n\n" if exp else "")
            + "Coming from 0.9: every old import path still works in 1.0.x with a warning; `solvi migrate PATH` "
              "rewrites your code (this Space was moved with it).")
