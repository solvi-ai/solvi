"""Hack the trace: solvi makes a loan decision and publishes its hash-chained trace; the player edits the trace to flip the
answer, and the verifier shows which defence catches the forgery.

Every solvi answer comes with `resp.trace`: records (step, kind, name, input hashes, value, prev, hash) chained by hashes
from the hash of the input (`init_hash`). `trace.replay(catalog)` recomputes every step from its recorded inputs and checks
each record's hash and the chain. Replay proves that a trace is internally consistent. It cannot prove that it is the SAME
trace that was produced at decision time: a forger who changes the input and honestly re-runs everything gets a consistent
trace of a different application. That is what the published receipt is for (the input hash and the final hash, handed to
the applicant and written to an append-only log when the decision is made).

Pure logic, no UI: decide, dumps/loads, verify, rehash_step, rechain, rerun, attack."""
from __future__ import annotations

import json

from solvi import Answer, Catalog, Question, System
from solvi.runtime import MISSING, Record, Trace, vhash

cat = Catalog()


@cat.fn
def monthly_payment(loan_amount, term_months):
    """annuity payment for the loan at 12% a year (1% a month)"""
    r = 0.01
    return round(loan_amount * r / (1 - (1 + r) ** -term_months), 2)


@cat.fn
def debt_to_income(monthly_debts, monthly_payment, monthly_income):
    """share of the income that goes to debts, the new loan included"""
    return round((monthly_debts + monthly_payment) / monthly_income, 3)


@cat.check(hard=True, then={"decision": "reject"})
def affordable(debt_to_income):
    """hard: after the loan, debts take at most 60% of the income"""
    return debt_to_income <= 0.6


@cat.fn
def risk_points(debt_to_income, late_payments_12m, years_employed):
    """1 point per % of debt-to-income, +15 per late payment this year, −3 per year in the job (up to 5)"""
    return round(debt_to_income * 100) + 15 * late_payments_12m - 3 * min(years_employed, 5)


@cat.rule("decision")
def decision(risk_points):
    return "approve" if risk_points <= 35 else "review" if risk_points <= 50 else "reject"


QUESTIONS = [Question("decision", "Approve the loan?", Answer.choice(["approve", "review", "reject"]),
                      requires=["affordable"])]
system = System(cat, QUESTIONS)

APPLICANTS = {
    "Alex: 15,000 over 36 months, 2 late payments": {
        "applicant": "Alex", "monthly_income": 4200, "monthly_debts": 900, "loan_amount": 15000, "term_months": 36,
        "late_payments_12m": 2, "years_employed": 1},
    "Sam: 20,000 over 48 months, high debts": {
        "applicant": "Sam", "monthly_income": 3000, "monthly_debts": 1150, "loan_amount": 20000, "term_months": 48,
        "late_payments_12m": 1, "years_employed": 4},
    "Kim: 8,000 over 24 months, new job": {
        "applicant": "Kim", "monthly_income": 2600, "monthly_debts": 700, "loan_amount": 8000, "term_months": 24,
        "late_payments_12m": 1, "years_employed": 0},
}

LEVELS = {
    "1 · Edit a value": (
        "Change the decision in the last record from `reject` to `approve`. Nothing else.",
        "**Record hash.** Each record's `hash` covers its own contents (step, name, input hashes, the value's hash, prev). "
        "Change the value and the stored hash no longer matches: *record modified after execution*. The step is also "
        "recomputed from its inputs, which gives `reject` again."),
    "2 · Edit a value and fix its hash": (
        "Lower `risk_points` so the rule would approve, set the decision to `approve`, and press **Rehash step N** for "
        "each record you changed, so every record matches its own hash.",
        "**Hash chain.** Every record stores the hash of the record before it (`prev`). Rehash step 4 and step 5 still "
        "points at the old hash of step 4: *hash chain broken*. And the recompute still says `risk_points` is 60."),
    "3 · Rewrite the whole chain": (
        "Same edits, then press **Recompute the whole hash chain**: every input hash, prev and hash is rebuilt, so all "
        "hashes are consistent.",
        "**Replay.** Replay does not trust the values: it recomputes every step from its recorded inputs. "
        "`risk_points` from the recorded debt-to-income and late payments is 60, not what you wrote: "
        "*value ≠ recomputed*. You cannot fix that without changing an input."),
    "4 · Change an input too": (
        "Change an input in `init` (say, `late_payments_12m` to 0 or a higher income), press **Re-run the steps** "
        "(an honest recomputation from the edited input) and **Recompute the whole hash chain**.",
        "**The published receipt.** Now the trace is an honest, consistent record, of a *different* application: "
        "replay passes. What catches it is the receipt published at decision time: the input hash and the final hash "
        "no longer match it. A hash chain proves integrity only against an anchor kept outside the trace."),
    "Bonus · Remove the evidence": (
        "Delete the `risk_points` record, set the decision to `approve`, recompute the hash chain.",
        "**The plan.** solvi's replay does not recompute a step whose inputs are missing from the trace, so this one "
        "passes replay. The verifier re-plans the flow from the catalog and sees a step is missing (and the receipt "
        "still does not match)."),
}


def decide(applicant_key):
    """→ (resp, trace dict, receipt)"""
    init = dict(APPLICANTS[applicant_key])
    resp = system.ask(init, ["decision"])
    d = trace_dict(resp.trace)
    receipt = {"answer": resp["decision"].answer, "init_hash": d["init_hash"], "head": d["records"][-1]["hash"],
               "steps": len(d["records"])}
    return resp, d, receipt


def trace_dict(trace):
    recs = []
    for r in trace.records:
        rec = {"step": r.step, "kind": r.kind, "name": r.name, "value": None if r.value is MISSING else r.value,
               "inputs": dict(r.inputs), "prev": r.prev, "hash": r.hash}
        if r.error:
            rec["error"] = r.error
        recs.append(rec)
    return {"init": dict(trace.init), "init_hash": trace.init_hash, "records": recs}


def dumps(d):
    """JSON with one record per two lines (readable in a code editor, still valid JSON)."""
    j = json.dumps
    lines = ["{", f'  "init": {j(d["init"], ensure_ascii=False)},', f'  "init_hash": {j(d["init_hash"])},', '  "records": [']
    recs = d["records"]
    for i, r in enumerate(recs):
        extra = "".join(f', {j(k)}: {j(v, ensure_ascii=False)}' for k, v in r.items()
                        if k not in ("step", "kind", "name", "value", "inputs", "prev", "hash"))
        lines.append(f'    {{"step": {j(r.get("step"))}, "kind": {j(r.get("kind"))}, "name": {j(r.get("name"))}, '
                     f'"value": {j(r.get("value"), ensure_ascii=False)},')
        lines.append(f'     "inputs": {j(r.get("inputs"))}, "prev": {j(r.get("prev"))}, "hash": {j(r.get("hash"))}{extra}}}'
                     + ("," if i < len(recs) - 1 else ""))
    lines += ["  ]", "}"]
    return "\n".join(lines)


class TraceFormatError(ValueError):
    pass


def loads(text):
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise TraceFormatError(f"the JSON does not parse: {e.msg} (line {e.lineno}, column {e.colno})") from None
    if not isinstance(d, dict) or not isinstance(d.get("init"), dict) or not isinstance(d.get("records"), list):
        raise TraceFormatError('expected {"init": {...}, "init_hash": "...", "records": [...]}')
    for i, r in enumerate(d["records"]):
        if not isinstance(r, dict) or not isinstance(r.get("inputs", {}), dict):
            raise TraceFormatError(f"record #{i + 1} must be an object with an \"inputs\" object")
    return d


def to_trace(d):
    recs = []
    for r in d["records"]:
        rec = Record(step=r.get("step"), kind=r.get("kind"), name=r.get("name"), inputs=dict(r.get("inputs") or {}),
                     value=r.get("value"), error=r.get("error"), prev=r.get("prev", ""), hash=r.get("hash", ""))
        recs.append(rec)
    return Trace(str(d.get("init_hash", "")), recs, dict(d["init"]))


def answer_of(d):
    """The answer this trace claims: a failed hard check forces 'reject', else the rule's recorded value."""
    for r in d["records"]:
        if r.get("name") == "affordable" and r.get("value") is False:
            return "reject"
    for r in reversed(d["records"]):
        if r.get("name") == "answer:decision":
            return r.get("value")
    return None


def _body(r):
    return {"step": r.get("step"), "kind": r.get("kind"), "name": r.get("name"), "inputs": r.get("inputs") or {},
            "value": vhash(r.get("value")), "quote": None, "error": r.get("error"), "prev": r.get("prev", "")}


def _part(name):
    if isinstance(name, str) and name.startswith("answer:"):
        return cat.rules.get(name[7:])
    return cat.parts.get(name)


def planned_names(init):
    from solvi.strategist import plan
    flow = plan(cat, QUESTIONS, init.keys())
    return [s.part.name for s in flow.steps]


CHECKS = [
    ("record", "Record hashes", "each record matches its own hash"),
    ("chain", "Hash chain", "each record points to the hash of the one before it"),
    ("inputs", "Recorded inputs", "the input hashes match the values they came from"),
    ("recompute", "Replay", "every step recomputed from its recorded inputs gives the recorded value"),
    ("plan", "Complete flow", "every step the catalog plans is there, in order, none marked failed"),
    ("anchor", "Input anchor", "init_hash is the hash of init"),
    ("receipt", "Published receipt", "same input hash and same final hash as published at decision time"),
]


def verify(text, original, receipt):
    """→ dict(ok, parse_error, answer, changed, checks={key: [messages]}, bad_steps={step: [messages]}, first)"""
    try:
        d = loads(text)
    except TraceFormatError as e:
        return {"ok": False, "parse_error": str(e)}
    checks = {k: [] for k, _, _ in CHECKS}
    bad_steps: dict = {}

    def flag(key, step, msg):
        checks[key].append(msg)
        bad_steps.setdefault(step, []).append(msg)

    unknown = [r for r in d["records"] if _part(r.get("name")) is None]
    for r in unknown:
        flag("plan", r.get("step"), f"step {r.get('step')}: no part named {r.get('name')!r} in the catalog")
    if not unknown:
        try:
            rep = to_trace(d).replay(cat)
            for step, nm, msg in rep["mismatches"]:
                key = ("record" if "modified" in msg else "chain" if "chain" in msg else "inputs" if msg.startswith("input ")
                       else "recompute")
                text_ = {"record": f"step {step} ({nm}): record modified after execution — its hash does not match its "
                                   f"contents",
                         "chain": f"hash chain broken at step {step} ({nm}): its prev is not the hash of the record before",
                         "inputs": f"step {step} ({nm}): {msg}",
                         "recompute": f"step {step} ({nm}): {msg}"}[key]
                flag(key, step, text_)
        except Exception as e:  # noqa: BLE001 - any malformed edit is reported, never crashes the page
            flag("recompute", None, f"replay could not run: {type(e).__name__}: {e}")
    want = planned_names(d["init"])
    got = [r.get("name") for r in d["records"]]
    if got != want:
        missing = [n for n in want if n not in got]
        extra = [n for n in got if n not in want]
        msg = "the flow is not the planned one: " + "; ".join(
            x for x in (f"missing {', '.join(missing)}" if missing else "", f"unexpected {', '.join(map(str, extra))}"
                        if extra else "", "steps out of order" if not missing and not extra else "") if x)
        flag("plan", None, msg)
    if [r.get("step") for r in d["records"]] != list(range(1, len(d["records"]) + 1)):
        flag("plan", None, "step numbers are not 1, 2, 3, … (a step was removed or added)")
    for r in d["records"]:
        if r.get("error"):
            flag("plan", r.get("step"), f"step {r.get('step')} ({r.get('name')}) claims it failed ({r.get('error')!r}) — "
                                        f"replay does not recompute failed steps, but the original run had no failures")
    if vhash(d["init"]) != d.get("init_hash"):
        flag("anchor", None, f"init_hash {str(d.get('init_hash'))[:10]}… is not the hash of init ({vhash(d['init'])[:10]}…)")
    head = d["records"][-1].get("hash") if d["records"] else None
    if d.get("init_hash") != receipt["init_hash"]:
        flag("receipt", None, f"the input hash {str(d.get('init_hash'))[:10]}… ≠ the published {receipt['init_hash'][:10]}… "
                              f"(a different application)")
    if head != receipt["head"]:
        flag("receipt", None, f"the final hash {str(head)[:10]}… ≠ the published {receipt['head'][:10]}…")
    ans = answer_of(d)
    changed = ans != receipt["answer"]
    tampered = json.dumps(d, sort_keys=True) != json.dumps(original, sort_keys=True)
    ok = not any(checks.values())
    first = next(((label, checks[k][0]) for k, label, _ in CHECKS if checks[k]), None)
    return {"ok": ok, "parse_error": None, "answer": ans, "changed": changed, "tampered": tampered, "checks": checks,
            "bad_steps": bad_steps, "first": first, "doc": d, "replay_ok": not any(checks[k] for k in
                                                                               ("record", "chain", "inputs", "recompute"))}


# --------------------------------------------------------------------------------------------------------------------
# The forger's helpers
# --------------------------------------------------------------------------------------------------------------------

def rehash_step(text, step):
    """Recompute one record's own hash (keeps its prev)."""
    d = loads(text)
    for r in d["records"]:
        if r.get("step") == step:
            r["hash"] = vhash(_body(r))
            return dumps(d)
    raise TraceFormatError(f"there is no step {step}")


def rechain(text):
    """Rebuild every hash: init_hash from init, each record's input hashes from the values it read, prev and hash."""
    d = loads(text)
    d["init_hash"] = vhash(d["init"])
    vals = dict(d["init"])
    prev = d["init_hash"]
    for r in d["records"]:
        r["inputs"] = {x: vhash(vals[x]) for x in (r.get("inputs") or {}) if x in vals}
        r["prev"] = prev
        r["hash"] = vhash(_body(r))
        prev = r["hash"]
        vals[r.get("name")] = r.get("value")
    return dumps(d)


def rerun(text):
    """Recompute every value from init, in the trace's order (an honest re-execution of the edited input)."""
    d = loads(text)
    vals = dict(d["init"])
    for r in d["records"]:
        part = _part(r.get("name"))
        if part is None:
            continue
        args = {x: vals.get(x, MISSING) for x in part.inputs}
        if any(v is MISSING for v in args.values()):
            vals[r["name"]] = r.get("value")
            continue
        try:
            r["value"] = part.func(**args)
        except Exception as e:  # noqa: BLE001
            raise TraceFormatError(f"step {r.get('step')} ({r.get('name')}) fails on these inputs: {type(e).__name__}: {e}")
        vals[r["name"]] = r["value"]
    return dumps(d)


def _set(d, name, value):
    for r in d["records"]:
        if r.get("name") == name:
            r["value"] = value


def attack(level, original):
    """The level's attack applied to the original trace → JSON text."""
    d = json.loads(json.dumps(original))
    n = len(d["records"])
    if level.startswith("1"):
        _set(d, "answer:decision", "approve")
        return dumps(d)
    if level.startswith("Bonus"):
        d["records"] = [r for r in d["records"] if r["name"] != "risk_points"]
        _set(d, "answer:decision", "approve")
        return rechain(dumps(d))
    if level.startswith("4"):
        d["init"]["late_payments_12m"] = 0
        d["init"]["monthly_income"] = int(d["init"]["monthly_income"] * 1.6)
        return rechain(rerun(dumps(d)))
    _set(d, "risk_points", 20)
    _set(d, "answer:decision", "approve")
    t = dumps(d)
    if level.startswith("2"):
        for step in (n - 1, n):
            t = rehash_step(t, step)
        return t
    return rechain(t)
