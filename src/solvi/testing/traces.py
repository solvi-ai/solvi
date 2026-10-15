"""Regression cases from what a System has decided: `cases_from_store` turns recorded decisions into cases (their
inputs and the answers given), `boundary` makes inputs at the edges of what the cases show — per field the values seen,
then empty / single / long text, 0, -1, one past the smallest and largest number, an empty or one-element list, one
level into dicts and lists — and, given the System, pins its current answers on them as the expected ones. Both give
case dicts in the cases.json format (see solvi.testing), so `solvi test` and `run_case` run them."""
from __future__ import annotations

import copy
import json
import warnings

from .honesty import _plain

LONG = 1000             # a "long" text or list has this many characters / elements
BIG = 2 ** 63           # a number past a 64-bit integer


def _key(x):
    return json.dumps(x, sort_keys=True, default=str)


def _outcome(answers, status):
    return _key([answers, status])


def cases_from_store(storage, n=None):
    """Regression cases from the decisions a store holds → [case dict]: per decision its recorded input as "state",
    the answers it gave as "expected" (an abstention as null), each question's status as "status" and the questions
    it asked as "ask"; the name is "stored <id>". The same input recorded twice is one case (the first). With n: at most
    n cases — one per distinct outcome (answers and statuses) first, in store order, then the rest in store order.
    Full and compact records alike (both keep the input and the answers); redacted ones are passed over. The state is
    the input as it was given to ask: a task whose cases go through prepare(state) should not prepare it again."""
    out, seen = [], set()
    for s in storage.iter():
        state = s.input
        if not isinstance(state, dict):
            continue
        k = _key(state)
        if k in seen:
            continue
        seen.add(k)
        rec = s.data.get("answers") or {}
        answers = s.answers
        status = {q: a[2] for q, a in rec.items() if len(a) > 2}
        expected = {q: None if status.get(q) == "abstain" else a for q, a in answers.items()}
        out.append({"name": f"stored {s.id}", "state": state, "expected": expected, "status": status,
                    "ask": list(answers)})
    if n is None or len(out) <= n:
        return out
    first, rest, outs = [], [], set()
    for c in out:
        o = _outcome(c["expected"], c["status"])
        (rest if o in outs else first).append(c)
        outs.add(o)
    keep = {id(c) for c in (first + rest)[:n]}
    return [c for c in out if id(c) in keep]


# --- boundary inputs
class _Profile:
    """What the cases show about one field: the values seen (by kind), numbers' range, texts' and lists' lengths, and
    — one level deep — a profile per key of a dict value and of a list's elements."""

    def __init__(self):
        self.values = []
        self.nums = []
        self.keys = {}
        self.elem = None

    def add(self, v, depth=0):
        if len(self.values) < 8 and v not in self.values:
            self.values.append(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.nums.append(v)
        if depth < 1 and isinstance(v, dict):
            for k, x in v.items():
                self.keys.setdefault(k, _Profile()).add(x, depth + 1)
        if depth < 1 and isinstance(v, list):
            for x in v:
                self.elem = self.elem or _Profile()
                self.elem.add(x, depth + 1)


def _text(s):
    base = s or "abc"
    return [("empty text", ""), ("one character", base[:1]), ("long text", (base * (LONG // len(base) + 1))[:LONG]),
            ("non-ASCII text", base + " é Привет 😀"), ("padded text", f"  {base}  "), ("upper case", base.upper())]


def _number(v, p):
    out = [("zero", 0 if isinstance(v, int) else 0.0), ("minus one", -1 if isinstance(v, int) else -1.0)]
    if p is not None and p.nums:
        lo, hi = min(p.nums), max(p.nums)
        step = 1 if all(isinstance(x, int) for x in p.nums) else max(abs(hi - lo) * 0.01, 1e-9)
        out += [("one past the largest seen", hi + step), ("one below the smallest seen", lo - step),
                ("the largest seen", hi), ("the smallest seen", lo)]
    out += [("one above", v + 1), ("one below", v - 1)]
    out.append(("huge", BIG if isinstance(v, int) else 1e300))
    return out


def _variants(v, p, depth=0):
    """Boundary replacements of one field's value → [(what, value)]."""
    if isinstance(v, bool):
        return [("flipped", not v)]
    if isinstance(v, (int, float)):
        return _number(v, p)
    if isinstance(v, str):
        return _text(v) + [("another value seen", x) for x in (p.values if p else ()) if isinstance(x, str) and x != v]
    if isinstance(v, list):
        out = [("empty list", []), ("one element", v[:1])] if v else []
        if v:
            out.append(("long list", [v[i % len(v)] for i in range(LONG)]))
            if depth < 1:
                for what, x in _variants(v[0], p.elem if p else None, depth + 1):
                    out.append((f"first element: {what}", [x] + v[1:]))
        return out
    if isinstance(v, dict) and depth < 1:
        out = [("empty dict", {})]
        for k, x in v.items():
            for what, y in _variants(x, (p.keys.get(k) if p else None), depth + 1):
                out.append((f"{k}: {what}", {**v, k: y}))
        return out
    return []


def _store_kw(system):
    """store=False for System.ask (an object with a plain ask(state) gets nothing): a boundary input is not a decision."""
    import inspect
    try:
        return {"store": False} if "store" in inspect.signature(system.ask).parameters else {}
    except (TypeError, ValueError):
        return {}


def _answers(res):
    expected = {q: None if r.status == "abstain" else _plain(r.answer) for q, r in res.results.items()}
    return expected, {q: r.status for q, r in res.results.items()}


def boundary(cases, n=50, system=None):
    """Inputs at the edges of what the cases show → up to n new case dicts (name, state, note "boundary: <field>:
    <what>"): each case's state with one field changed — to a value just past the range the cases show, 0, -1, a huge
    number; empty, one-character, long, non-ASCII, padded text; an empty, one-element or long list; one level into
    dicts and lists. Deduplicated, without the cases' own states, taken in turn from each kind of change, the same for
    the same cases (no randomness).
    system: the current answers are pinned — "expected" and "status" from asking each input (never stored), so the
    cases fail when a later change moves an answer at an edge. An input whose ask raises is left out (fuzz reports
    those), so is one whose answers differ when asked again or whose trace does not replay (nondeterministic); a
    UserWarning says how many. Without a system the cases have no expectations: fill them in before running."""
    states = [c["state"] for c in cases if isinstance(c.get("state"), dict)]
    prof = {}
    for st in states:
        for k, v in st.items():
            prof.setdefault(k, _Profile()).add(v)
    seen = {_key(st) for st in states}
    by_kind = {}
    for i, st in enumerate(states):
        for k, v in st.items():
            for what, x in _variants(v, prof.get(k)):
                new = {**copy.deepcopy(st), k: x}
                key = _key(new)
                if key in seen:
                    continue
                seen.add(key)
                kind = what.split(": ")[-1]
                by_kind.setdefault(kind, []).append({"name": f"boundary {len(seen)}", "state": new,
                                                     "note": f"boundary: {k}: {what}"})
    queues = list(by_kind.values())                   # in turn from each kind of change
    out = []
    while queues and len(out) < (n if system is None else 4 * n):
        for q in list(queues):
            if q:
                out.append(q.pop(0))
            else:
                queues.remove(q)
    for i, c in enumerate(out):
        c["name"] = f"boundary {i + 1}"
    if system is None:
        return out[:n]
    kept, crashed, unstable = [], 0, 0
    for c in out:
        if len(kept) >= n:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = system.ask(copy.deepcopy(c["state"]), **_store_kw(system))
                b = system.ask(copy.deepcopy(c["state"]), **_store_kw(system))
                rep = a.trace.replay(system)
        except Exception:  # noqa: BLE001
            crashed += 1
            continue
        if _answers(a) != _answers(b) or not rep["ok"]:
            unstable += 1
            continue
        c["expected"], c["status"] = _answers(a)
        kept.append(c)
    for i, c in enumerate(kept):
        c["name"] = f"boundary {i + 1}"
    if crashed or unstable:
        warnings.warn(f"boundary: {crashed} input(s) left out because ask raised (run fuzz for them), {unstable} because "
                      "the answers were nondeterministic (another answer when asked again, or a trace that does not "
                      "replay)", UserWarning, stacklevel=2)
    return kept
