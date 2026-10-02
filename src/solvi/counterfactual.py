"""Counterfactual explanations: the smallest change of a decision's given inputs that changes its answer — "approve if
amount ≤ 1000 (now 1200)", "refund if purchase_date ≥ 2026-08-20 (now 2026-08-10)" — for adverse-action reasons in
lending and clear answers in support.

How: the recorded decision is re-run on changed inputs through the deterministic flow only. The flow is the recorded one
(same keys, so the same plan); every model-backed part — an extractor, a model decision, a learned answer head, a part
whose provenance is decided / learned / proposed — is replaced by the proposal it recorded in this trace, so no model is
ever called: the answer is "what the code would decide if the models said what they said". A model part that did not run
in the recorded decision (skipped after a hard check failed) has no proposal: on inputs that need it the question abstains,
and such inputs are not counted as changing the answer. Learned rule lists (learn_rule) are code and are re-run.

The search, per input:
  numbers (int, float) and dates — probe outward from the current value in both directions with doubling steps, then
     bisect between the last unchanged and the first changed value: the nearest threshold crossing for inputs the answer
     is monotone in (a non-monotone input may hide a nearer crossing between two probes). Non-negative inputs stay
     non-negative unless a domain says otherwise; float bounds are shown at the shortest decimal that still holds;
  booleans, Enums and Literal-typed inputs (System(inputs=...)) — every other value;
  anything else — only with `domains={fact: [values]}`.
Two inputs together only when no single one changes the answer: one input's candidates (its values, or probe points) with
a search over the other, then each bound tightened with the other change made. Changes are ranked by count, then by size
(relative change of a number, 1 for an enumerated value)."""
from __future__ import annotations

import copy
import dataclasses
import datetime
import enum
import math
from dataclasses import dataclass, field

from .provenance import FUZZY


# ---------------------------------------------------------------- results
@dataclass
class Change:
    """One change of a given input in a counterfactual: the fact, its value now, the value tried (`op` "=" or a
    boundary "≤", "<", "≥", ">" for a number) and its cost."""
    fact: str
    now: object
    to: object                     # the value tried (the boundary for a number)
    op: str = "="                  # "=", "≤", "<", "≥", ">"
    cost: float = 1.0

    def __str__(self):
        return f"{self.fact} {self.op} {_val(self.to)} (now {_val(self.now)})"


@dataclass
class Counterfactual:
    """The answer after a set of Changes (str: "<answer> if <changes>"), its status and reason, and `cost` (the sum
    of the changes' costs); `to_dict()` for JSON."""
    answer: object                 # the answer after the change
    changes: list
    status: str = "ok"
    why: str = ""
    kind: str | None = None

    @property
    def cost(self):
        return sum(c.cost for c in self.changes)

    def __str__(self):
        return f"{_ans(self.answer, self.kind)} if " + " and ".join(str(c) for c in self.changes)

    def to_dict(self):
        return {"answer": _plain(self.answer), "status": self.status, "why": self.why, "text": str(self),
                "changes": [{"fact": c.fact, "now": _plain(c.now), "to": _plain(c.to), "op": c.op, "cost": c.cost}
                            for c in self.changes]}


@dataclass
class Counterfactuals:
    question: str
    answer: object                 # the answer now
    status: str
    found: list = field(default_factory=list)       # [Counterfactual], best first
    searched: list = field(default_factory=list)    # the inputs searched
    not_searched: dict = field(default_factory=dict)   # input → why not
    held: list = field(default_factory=list)        # model-backed parts held at their recorded proposals
    unavailable: list = field(default_factory=list)  # model-backed parts without a recorded proposal (did not run)
    evals: int = 0
    exhausted: bool = False                         # the search stopped at max_evals
    kind: str | None = None
    inconclusive: str | None = None                 # why nothing was searched (a stored response whose input was lost)

    @property
    def best(self):
        return self.found[0] if self.found else None

    def __iter__(self):
        return iter(self.found)

    def __len__(self):
        return len(self.found)

    def __bool__(self):
        return bool(self.found)

    def __str__(self):
        now = "abstains" if self.status == "abstain" else f"= {_ans(self.answer, self.kind)} [{self.status}]"
        lines = [f"{self.question} {now}"]
        if self.found:
            lines += [f"  {c}" for c in self.found]
        elif self.inconclusive:
            lines.append(f"  no conclusion: {self.inconclusive}")
        else:
            lines.append("  no change of " + (", ".join(self.searched) or "any input")
                         + " (one at a time" + (" or two together" if self._two else "") + ") changes the answer")
        if self.held:
            lines.append("  held at their recorded proposals (no model called): " + ", ".join(self.held))
        if self.unavailable:
            lines.append("  without a recorded proposal (did not run): " + ", ".join(self.unavailable))
        if self.not_searched:
            lines.append("  not searched: " + "; ".join(f"{k} ({w})" for k, w in self.not_searched.items()))
        if self.exhausted:
            lines.append(f"  search stopped after {self.evals} evaluations (max_evals)")
        return "\n".join(lines)

    _two = False

    def to_dict(self):
        return {"question": self.question, "answer": _plain(self.answer), "status": self.status,
                "found": [c.to_dict() for c in self.found], "searched": list(self.searched),
                "not_searched": dict(self.not_searched), "held": list(self.held), "unavailable": list(self.unavailable),
                "evals": self.evals, "exhausted": self.exhausted, "inconclusive": self.inconclusive}


def _val(v):
    if isinstance(v, enum.Enum):
        return str(v.value)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    if isinstance(v, float):
        return f"{v:g}" if abs(v) < 1e15 else repr(v)
    return repr(v) if isinstance(v, str) else str(v)


def _ans(a, kind=None):
    if isinstance(a, str):
        return a
    from .primitives import fmt
    return fmt(a, kind)


def _plain(v):
    from .storage import plain
    return plain(v)


# ---------------------------------------------------------------- re-running the deterministic flow
class _FixedHead:
    """An answer head held at its recorded answer (its probabilities from the trace); without a record it cannot answer."""
    model_id = "held"

    def __init__(self, rec):
        self.rec = rec
        self.features = [] if rec is not None else ["<no recorded answer>"]

    def predict(self, vals):
        return dict(self.rec.probs or {self.rec.value: 1.0})

    def contributions(self, vals):
        return {}

    def fingerprint(self):
        return self.rec.model.get("fp", "held") if self.rec is not None and self.rec.model else "held"


def _fuzzy(p):
    """Is a part model-backed (held at its recorded proposal)? A learned rule list is code: it is re-run."""
    if p.model is not None and type(p.model).__name__ == "RuleList":
        return False
    return p.model is not None or p.provenance in FUZZY or getattr(p.func, "__solvi_decision__", None) is not None


def _pinned(p, rec):
    """A part that returns the value its model proposed in the recorded trace (or fails as it failed), calling nothing."""
    from .runtime import MISSING
    if rec is None:
        why = f"{p.name}: a model-backed part that did not run in the recorded decision (no proposal to hold)"

        def f(**_):
            raise LookupError(why)
    elif rec.value is MISSING:
        err = rec.error or "no value"

        def f(**_):
            raise LookupError(err)
    else:
        value = rec.value

        def f(**_):
            return value
    f.__name__ = p.func.__name__ if p.func is not None else p.name
    return dataclasses.replace(p, func=f, kind="fn" if p.kind == "extract" else p.kind, model=None, provenance=None,
                               options=None, validate=None, min_confidence=None, alternatives=None, features=None,
                               exact=None, source=None, types=None, returns=None, tin=None, tout=None, timeout=None,
                               blocking=False)


class Rerun:
    """The recorded decision as a function of its given inputs: `rerun({fact: value})` → the question's Result, through the
    recorded flow with every model-backed part held at its recorded proposal."""

    def __init__(self, res, system, question):
        from .strategist import Flow, Step
        self.res, self.question = res, question
        tr = res.trace
        by = {}
        for r in tr.records:
            by.setdefault(r.name, r)
        cat = copy.copy(system.catalog)
        cat.parts, cat.rules = dict(cat.parts), dict(cat.rules)
        self.held, self.unavailable = [], []
        steps = []
        for st in res.flow.steps:
            p = system.catalog.rules.get(st.part.name[7:]) if st.part.kind == "rule" else system.catalog.parts.get(st.part.name)
            if p is None:
                raise ValueError(f"the system's catalog has no part {st.part.name!r} of this decision's flow")
            rec = by.get(p.name)
            q = p
            if p.alternatives is not None:
                alts = [a for a in p.alternatives if not _fuzzy(a)]
                used = next((a for a in p.alternatives if rec is not None and a.name == rec.producer), None)
                if len(alts) < len(p.alternatives):
                    if used is not None and _fuzzy(used) or not alts:
                        q = _pinned(p, rec)
                        self._note(p.name, rec)
                    else:
                        from .core import _group_func
                        q = dataclasses.replace(p, alternatives=alts)
                        q.func = _group_func(q)
                        self.held += [f"{p.name} ({a.name}: left out)" for a in p.alternatives if _fuzzy(a)]
            elif _fuzzy(p):
                q = _pinned(p, rec)
                self._note(p.name, rec)
            if q is not p:
                (cat.rules if p.kind == "rule" else cat.parts)[p.name[7:] if p.kind == "rule" else p.name] = q
            steps.append(Step(q, list(st.reasons)))
        f = res.flow
        self.flow = Flow(steps, f.per_question, f.skipped, f.unresolved, f.types, [])
        s = copy.copy(system)
        s.catalog, s.storage, s.learn, s.cost_policy = cat, None, False, None
        s.heads = {}
        for qn in system.heads:
            rec = next((r for r in tr.records if r.kind == "head" and r.name == "answer:" + qn), None)
            s.heads[qn] = _FixedHead(rec)
            if qn == question and qn not in system.catalog.rules:
                (self.held if rec is not None else self.unavailable).append(f"answer:{qn} (answer head)")
        self.system = s
        self.qs = [system.questions[q] for q in res.results if q in system.questions]
        self.init = dict(tr.init)
        self.evals = 0

    def _note(self, name, rec):
        (self.held if rec is not None else self.unavailable).append(name)

    def __call__(self, changes):
        from .runtime import execute
        self.evals += 1
        state = dict(self.init)
        state.update(changes)
        trace, vals = execute(self.system.catalog, self.flow, state)
        results, _, _ = self.system._results(self.qs, self.flow, trace, vals)
        return results[self.question]


# ---------------------------------------------------------------- the search
class _Stop(Exception):
    pass


def _literal_values(system, fact):
    import typing
    m = getattr(system, "inputs", None)
    fs = getattr(m, "model_fields", None) or {}
    ann = fs[fact].annotation if fact in fs else None
    if ann is not None and typing.get_origin(ann) is typing.Literal:
        return list(typing.get_args(ann))
    return None


def _num(v):
    """A searchable number → (as float, to value, integer?) or None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return float(v), (lambda x: int(round(x))), True
    if isinstance(v, float) and math.isfinite(v):
        return v, float, False
    if isinstance(v, datetime.date) and not isinstance(v, datetime.datetime):
        return float(v.toordinal()), (lambda x: datetime.date.fromordinal(int(round(x)))), True
    return None


def _bounds(v, domain):
    """The search range of a number: the domain's, else dates within the calendar and non-negative numbers kept
    non-negative."""
    if domain is not None:
        return domain
    if isinstance(v, datetime.date):
        return 1.0, float(datetime.date.max.toordinal())
    return (0.0 if v >= 0 else -math.inf), math.inf


def _nice(a, b):
    """The number with the fewest decimals in [min(a, b), max(a, b)]."""
    lo, hi = min(a, b), max(a, b)
    for d in range(-6, 13):
        for x in (round(hi, d), round(lo, d), round((lo + hi) / 2, d)):
            if lo <= x <= hi:
                return float(x)
    return float(b)


class _Search:
    def __init__(self, rerun, res, question, target, max_evals):
        self.rerun, self.max_evals = rerun, max_evals
        a = res.results[question]
        self.now, self.now_status = a.answer, a.status
        self.target = target
        self.hits = {}

    def changed(self, changes):
        """Does the answer change (to the target) with these inputs? → the Result, or None."""
        key = tuple(sorted((k, repr(v)) for k, v in changes.items()))
        if key in self.hits:
            return self.hits[key]
        if self.rerun.evals >= self.max_evals:
            raise _Stop
        r = self.rerun(changes)
        ok = r.status != "abstain" and (r.answer != self.now or self.now_status == "abstain")
        if self.target is not None:
            ok = ok and r.answer == self.target
        self.hits[key] = r if ok else None
        return self.hits[key]

    def numeric(self, fact, v0, base, bound=None, quick=False):
        """The nearest crossing of `fact` in each direction with `base` changes held → [(Change, Result)]. quick: first
        try the far end of each direction and skip the direction when the answer does not change there (monotone inputs;
        used inside the search over two inputs)."""
        x0, to, integer = _num(v0)
        lo_b, hi_b = _bounds(v0, bound)
        scale = max(abs(x0), 1.0)
        step0 = 1.0 if integer else scale / 1024
        out = []
        for sign in (-1, 1):
            if quick:
                far = min(max(x0 + sign * (scale * 1e6 + 1e6), lo_b), hi_b)
                if far == x0 or self.changed({**base, fact: to(far)}) is None:
                    continue
            last, k = x0, 0
            while True:
                x = x0 + sign * step0 * 2 ** k
                k += 1
                if x < lo_b or x > hi_b:
                    x = lo_b if sign < 0 else hi_b
                    if not math.isfinite(x) or x == last:
                        break
                if abs(x - x0) > scale * 1e6 + 1e6:
                    break
                r = self.changed({**base, fact: to(x)})
                if r is not None:
                    out.append(self._bisect(fact, v0, x0, last, x, to, integer, base, sign))
                    break
                if x in (lo_b, hi_b):
                    break
                last = x
        return [c for c in out if c is not None]

    def _bisect(self, fact, v0, x0, good, bad, to, integer, base, sign):
        """good: unchanged, bad: changed; → (Change at the boundary, Result there)."""
        r = self.changed({**base, fact: to(bad)})
        for _ in range(200):
            if integer and abs(bad - good) <= 1:
                break
            if not integer and abs(bad - good) <= 1e-9 * max(1.0, abs(bad)):
                break
            mid = (good + bad) / 2
            if integer:
                mid = math.floor(mid) if sign > 0 else math.ceil(mid)
                if mid in (good, bad):
                    break
            got = self.changed({**base, fact: to(mid)})
            if got is not None:
                bad, r = mid, got
            else:
                good = mid
        strict = False
        if not integer:
            nice = _nice(good, bad)
            got = self.changed({**base, fact: to(nice)})
            if got is not None:
                bad, r = nice, got
            else:
                bad, strict = nice, True
        op = (">" if strict else "≥") if sign > 0 else ("<" if strict else "≤")
        return Change(fact, v0, to(bad), op, abs(bad - x0) / max(abs(x0), 1.0)), r

    def values(self, fact, v0, vals, base):
        out = []
        for v in vals:
            if v == v0 and type(v) is type(v0):
                continue
            r = self.changed({**base, fact: v})
            if r is not None:
                out.append((Change(fact, v0, v, "=", 1.0), r))
        return out


def search(res, question, max_changes=2, over=None, target=None, domains=None, system=None, max_evals=5000):
    """See Response.counterfactual."""
    system = system if system is not None else getattr(res, "_system", None)
    if system is None or not hasattr(system, "questions"):
        raise ValueError("counterfactual needs the System that answered: pass system= (a response loaded from data "
                         "without its System does not know its questions)")
    if question not in res.results:
        raise KeyError(f"question {question!r} was not asked in this response")
    if max_changes not in (1, 2):
        raise ValueError("max_changes must be 1 or 2")
    domains = dict(domains or {})
    rerun = Rerun(res, system, question)
    a = res.results[question]
    out = Counterfactuals(question, a.answer, a.status, held=sorted(set(rerun.held)), unavailable=sorted(set(rerun.unavailable)),
                          kind=a.kind)
    out._two = max_changes == 2
    init = res.trace.init
    if over is None:
        from .audit import build
        over = [g["name"] for g in build(res, question)[question].given]
    lost = {f: w for f, w in (getattr(res.trace, "unrestored", None) or {}).items() if f in init}
    if lost:                                          # a stored response whose input did not come back as it was: when
        from .runtime import vhash                    # the re-run of the unchanged input no longer gives the recorded
        base = rerun({})                              # answer (text where the decision read a date), every re-run is
        if (vhash(base.answer), base.status) != (vhash(a.answer), a.status):   # off, and "no change changes it" is wrong
            for f in over:
                out.not_searched[f] = (f"came back from storage as JSON gave it — {lost[f]}" if f in lost
                                       else "the decision's input was not restored")
            out.inconclusive = ("the stored input was not restored (" + ", ".join(sorted(lost)) + "): re-run as it came "
                                "back, the decision does not give its recorded answer — declare the types "
                                "(System(inputs=...) or annotations) and load it again")
            return out
    kinds = {}
    for f in over:
        if f in lost:
            out.not_searched[f] = f"came back from storage as JSON gave it — {lost[f]}"
            continue
        if f not in init:
            out.not_searched[f] = "not a given input of this decision"
            continue
        v, d = init[f], domains.get(f)
        if isinstance(d, (list, set, frozenset)) or (isinstance(d, tuple) and _num(v) is None):
            kinds[f] = ("values", list(d))
        elif _num(v) is not None:
            kinds[f] = ("number", tuple(float(_num(x)[0]) for x in d) if isinstance(d, tuple) else None)
        elif isinstance(v, bool):
            kinds[f] = ("values", [not v])
        elif isinstance(v, enum.Enum):
            kinds[f] = ("values", list(type(v)))
        elif _literal_values(system, f) is not None:
            kinds[f] = ("values", _literal_values(system, f))
        else:
            out.not_searched[f] = f"{type(v).__name__}: no domain (pass domains={{{f!r}: [...]}})"
    out.searched = list(kinds)
    s = _Search(rerun, res, question, target, max_evals)

    def one(f, base, quick=False):
        k, d = kinds[f]
        return s.numeric(f, init[f], base, d, quick) if k == "number" else s.values(f, init[f], d, base)
    found = []
    try:
        for f in kinds:
            found += [Counterfactual(r.answer, [c], r.status, r.why, a.kind) for c, r in one(f, {})]
        if not found and max_changes == 2:
            found = _pairs(s, kinds, init, one)
    except _Stop:
        out.exhausted = True
    found.sort(key=lambda c: (len(c.changes), c.cost))
    seen, uniq = set(), []
    for c in found:
        k = frozenset((x.fact, x.op, repr(x.to)) for x in c.changes)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    out.found, out.evals = uniq, rerun.evals
    return out


def _candidates(f, kind, v0, domain):
    """Values of `f` to hold while searching another input: its enumerated values, or probe points (nearest first)."""
    if kind == "values":
        return [(x, 1.0) for x in domain if not (x == v0 and type(x) is type(v0))]
    x0, to, integer = _num(v0)
    lo_b, hi_b = _bounds(v0, domain)
    scale = max(abs(x0), 1.0)
    step0 = 1.0 if integer else scale / 16
    out = []
    for k in range(0, 17):
        for sign in (-1, 1):
            x = x0 + sign * step0 * 2 ** k
            if lo_b <= x <= hi_b:
                out.append((to(x), abs(x - x0) / scale))
    return sorted(out, key=lambda t: t[1])


def _pairs(s, kinds, init, one):
    """Two inputs together: for each candidate of one input (its values, or probe points nearest first), a search over the
    other; then a number's change is tightened to its boundary with the other change made."""
    found, best = [], math.inf
    names = list(kinds)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            fa, fb = (y, x) if kinds[y][0] == "values" and kinds[x][0] == "number" else (x, y)
            ka, da = kinds[fa]
            for va, ca in _candidates(fa, ka, init[fa], da):
                if ca >= best:
                    break
                hits = one(fb, {fa: va}, True)
                for cb, r in hits:
                    ch_a = Change(fa, init[fa], va, "=", ca)
                    if ka == "number":                # tighten fa's change with fb's change made
                        tight = [c for c in s.numeric(fa, init[fa], {fb: cb.to}, da)
                                 if (_num(c[0].to)[0] >= _num(init[fa])[0]) == (_num(va)[0] >= _num(init[fa])[0])]
                        if tight:
                            ch_a, r = tight[0]
                    cf = Counterfactual(r.answer, [ch_a, cb], r.status, r.why)
                    found.append(cf)
                    best = min(best, cf.cost)
                if hits:
                    break
    return found
