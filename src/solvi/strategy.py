"""The code strategist: dead ends dropped, the cheapest verified plan by declared (or measured) costs — no model.

The deterministic strategist (solvi.strategist) walks back from each question's targets by exact names and, for a fact with
alternative producers (`provides=`), needs the inputs of ALL of them (a fallback chain). This module plans differently:

1. **Points.** A search (an exact 0/1 program; branch and bound as a fallback) picks ONE producer per needed fact so that
   the plan is valid and cheapest (declared `cost=`; a part without a declared cost counts as `unit`). A producer whose
   inputs cannot be computed from the given facts (a dead end) is never chosen. Hard checks that govern a question (its
   `then` names the question, it has no `then`, or the question requires it) and that the deterministic flow over the
   usable producers contains are **mandatory milestones**: every plan must keep them.
2. **Verification.** The whole plan is checked (inputs bound, acyclic, types, every mandatory hard check kept); a plan that
   fails the check is used with the failure recorded (`on_failure="code"`), or falls back to the deterministic strategist
   (`"deterministic"`), or every question abstains (`"abstain"`).

The plan is recorded in the trace as one hashed record (kind "plan"); `trace.replay` re-verifies it against the catalog.

    from solvi.strategy import CostStrategist
    system = System(cat, questions, strategist=CostStrategist())                          # dead ends dropped
    system = System(cat, questions, strategist=CostStrategist(producers="equivalent"))    # the cheapest verified plan

(The model strategist, `ModelStrategist`, that proposed producers where declared costs did not settle the choice, was
removed in 1.0: it gained nothing over declaring `cost=`.)"""
from __future__ import annotations

import dataclasses
import time

from . import _deprecate
from .strategist import Flow, PlanError, plan as det_plan
from .core import Catalog, _group_func
from .runtime import narrowed, replay_plan             # defined there (the runtime replays plan records); re-exported

UNIT = 1.0
MAX_EXPAND = 20000               # branch and bound (search(method="bnb"), or no scipy milp): nodes before it stops


# ---------------------------------------------------------------------------------------------------------------- catalog
def alternatives(p):
    """The producers of a part's fact: its alternatives, or the part itself."""
    return list(p.alternatives) if p.alternatives is not None else [p]


def reachable(catalog, init_keys):
    """Facts computable from init_keys when a fact needs only ONE of its producers (the deterministic strategist needs the
    inputs of every alternative)."""
    have = set(init_keys)
    changed = True
    while changed:
        changed = False
        for name, p in catalog.parts.items():
            if name not in have and any(all(x in have for x in a.inputs) for a in alternatives(p)):
                have.add(name)
                changed = True
    return have


def usable(catalog, fact, reach):
    return [a for a in alternatives(catalog.parts[fact]) if all(x in reach for x in a.inputs)]


def cost_fn(costs=None, unit=UNIT):
    """costs: None — declared `cost=` (unit when not declared); a dict {part name: cost} overrides it."""
    def c(a):
        if costs is not None and a.name in costs:
            return float(costs[a.name])
        return float(a.cost) if a.cost is not None else unit
    return c


def type_ok(catalog, fact, reader_type):
    """Does the fact's type fit a reader's declared type? (untyped on either side: yes)"""
    from .typed import compatible
    t = catalog.types.get(fact)
    if t is None or reader_type is None:
        return True
    try:
        return bool(compatible(t, reader_type))
    except Exception:  # noqa: BLE001
        return False


def targets_of(catalog, q, reach, init_keys, heads=None):
    heads = heads or {}
    rule = catalog.rules.get(q.name)
    if rule is not None:
        return list(rule.inputs)
    if q.name in heads:
        return list(heads[q.name].features)
    if q.uses:
        return list(q.uses)
    return sorted(f for f in reach - set(init_keys) if catalog.parts[f].kind != "rule")


def governs(part, q):
    return not part.then or q.name in part.then or part.name in q.requires


# ---------------------------------------------------------------------------------------------------------------- search
@dataclasses.dataclass
class Selection:
    """One producer per needed fact (names), the facts the plan needs, its cost and whether the search proved it cheapest."""
    choice: dict                      # fact → producer name (an alternative's name, or the part's own name)
    needed: set
    cost: float
    proven: bool = True
    feasible: bool = True
    why: str = ""


def search(catalog, questions, init_keys, costs=None, heads=None, fixed=None, extra=(), unit=UNIT,
           method="auto"):
    """Cheapest valid selection for the questions' targets, their required parts and `extra` parts (e.g. mandatory checks).
    fixed: {fact: producer name} choices that must be kept.
    method: "milp" (exact: a 0/1 program solved by scipy's HiGHS), "bnb" (branch and bound, capped at MAX_EXPAND nodes)
    or "auto" (milp when scipy has it). Ties go to the producer declared first.
    → Selection (feasible=False when a target cannot be computed; `why` says which)."""
    if method in ("auto", "milp"):
        try:
            from scipy.optimize import milp  # noqa: F401
            return _search_milp(catalog, questions, init_keys, costs, heads, fixed, extra, unit)
        except ImportError:
            if method == "milp":
                raise
    return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, unit)


def _start(catalog, questions, init, reach, heads, extra):
    start, missing = [], []
    for q in questions:
        for f in targets_of(catalog, q, reach, init, heads) + list(q.requires):
            (start if f in reach else missing).append(f)
    for f in extra:
        (start if f in reach else missing).append(f)
    return list(dict.fromkeys(start)), missing


def _search_milp(catalog, questions, init_keys, costs, heads, fixed, extra, unit):
    """min Σ c_a·x_a  s.t.  Σ_{a produces f} x_a = y_f (a needed fact has one producer), x_a ≤ y_i for every input i of a
    (a chosen producer's inputs are needed), y_t = 1 for the targets; x, y ∈ {0, 1}. Ties: +1e-6 × declaration position."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    init = set(init_keys)
    reach = reachable(catalog, init)
    c = cost_fn(costs, unit)
    fixed = dict(fixed or {})
    start, missing = _start(catalog, questions, init, reach, heads, extra)
    if not start:
        return Selection({}, set(), float("inf") if missing else 0.0, True, not missing,
                         ("cannot compute: " + ", ".join(sorted(set(missing)))) if missing else "")
    facts = [f for f in catalog.parts if f in reach and f not in init]
    fi = {f: j for j, f in enumerate(facts)}
    alts = []                                   # (fact, producer)
    for f in facts:
        us = usable(catalog, f, reach)
        if f in fixed:
            us = [a for a in us if a.name == fixed[f]] or us
        alts += [(f, a) for a in us]
    nf, na = len(facts), len(alts)
    obj = np.zeros(nf + na)
    for k, (f, a) in enumerate(alts):
        pos = [x.name for x in alternatives(catalog.parts[f])].index(a.name)
        obj[nf + k] = c(a) + 1e-6 * (1 + pos)
    rows = []
    A = lil_matrix((nf + sum(len([x for x in a.inputs if x not in init]) for _, a in alts), nf + na))
    r = 0
    by_fact = {}
    for k, (f, a) in enumerate(alts):
        by_fact.setdefault(f, []).append(k)
    for f in facts:                             # Σ x_a − y_f = 0
        for k in by_fact.get(f, ()):
            A[r, nf + k] = 1
        A[r, fi[f]] = -1
        rows.append((0, 0))
        r += 1
    for k, (f, a) in enumerate(alts):           # y_i − x_a ≥ 0
        for x in a.inputs:
            if x in init:
                continue
            A[r, fi[x]] = 1
            A[r, nf + k] = -1
            rows.append((0, np.inf))
            r += 1
    lo = np.zeros(nf + na)
    for f in start:
        if f in fi:
            lo[fi[f]] = 1
    res = milp(obj, integrality=np.ones(nf + na), bounds=Bounds(lo, np.ones(nf + na)),
               constraints=LinearConstraint(A.tocsr(), [a for a, _ in rows], [b for _, b in rows]),
               options={"time_limit": 10.0})
    if res.x is None and res.status == 1:           # time limit without a solution: branch and bound instead
        return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, unit)
    if res.x is None:
        return Selection({}, set(), float("inf"), False, False, "no valid plan: " + str(res.message))
    x = res.x
    choice = {f: a.name for k, (f, a) in enumerate(alts) if x[nf + k] > 0.5}
    cost = sum(c(a) for k, (f, a) in enumerate(alts) if x[nf + k] > 0.5)
    if not _acyclic(catalog, choice, init):
        return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, unit)
    sel = Selection(choice, set(choice) | (init & _inputs_closure(catalog, choice, init)), cost, res.status == 0)
    if missing:
        sel.why = "not computable (left to the questions to abstain): " + ", ".join(sorted(set(missing)))
    return sel


def _acyclic(catalog, choice, init):
    color = {}

    def dfs(f):
        color[f] = 1
        for x in producer(catalog, f, choice[f]).inputs:
            if x in init or x not in choice:
                continue
            k = color.get(x, 0)
            if k == 1 or (k == 0 and not dfs(x)):
                return False
        color[f] = 2
        return True
    return all(color.get(f, 0) == 2 or dfs(f) for f in choice)


def _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, unit):
    init = set(init_keys)
    reach = reachable(catalog, init)
    c = cost_fn(costs, unit)
    fixed = dict(fixed or {})
    start = []
    missing = []
    for q in questions:
        for f in targets_of(catalog, q, reach, init, heads) + list(q.requires):
            (start if f in reach else missing).append(f)
    for f in extra:
        (start if f in reach else missing).append(f)
    alts_of, lb_of = {}, {}

    def alts(f):
        if f not in alts_of:
            us = usable(catalog, f, reach)
            if f in fixed:
                us = [a for a in us if a.name == fixed[f]] or us
            order = {id(a): i for i, a in enumerate(alternatives(catalog.parts[f]))}
            alts_of[f] = sorted(us, key=lambda a: (c(a), order[id(a)]))
            lb_of[f] = c(alts_of[f][0]) if alts_of[f] else 0.0
        return alts_of[f]
    best = {"cost": float("inf"), "choice": None}
    stats = {"n": 0, "cut": False}

    def acyclic(choice):
        color = {}

        def dfs(f):
            color[f] = 1
            for x in catalog.parts[f].inputs if f not in choice else next(
                    a for a in alternatives(catalog.parts[f]) if a.name == choice[f]).inputs:
                if x in init or x not in choice:
                    continue
                k = color.get(x, 0)
                if k == 1 or (k == 0 and not dfs(x)):
                    return False
            color[f] = 2
            return True
        return all(color.get(f, 0) == 2 or dfs(f) for f in choice)

    def rec(agenda, chosen, cost):
        stats["n"] += 1
        pend = {f for f in agenda if f not in init and f not in chosen}
        lb = 0.0
        for f in pend:
            alts(f)
            lb += lb_of[f]
        if cost + lb >= best["cost"] - 1e-12:
            return
        agenda = list(agenda)
        while agenda:
            f = agenda.pop()
            if f in init or f in chosen:
                continue
            a_s = alts(f)
            if not a_s:
                return
            if len(a_s) == 1:
                chosen[f] = a_s[0].name
                cost += c(a_s[0])
                agenda.extend(a_s[0].inputs)
                continue
            for a in a_s:
                if stats["n"] >= MAX_EXPAND:
                    stats["cut"] = True
                    if best["choice"] is not None:
                        return
                rec(tuple(agenda) + tuple(a.inputs), {**chosen, f: a.name}, cost + c(a))
            return
        if cost < best["cost"] - 1e-12 and acyclic(chosen):
            best["cost"], best["choice"] = cost, dict(chosen)
    for f in start:
        alts(f)
    rec(tuple(dict.fromkeys(start)), {}, 0.0)
    if best["choice"] is None:
        return Selection({}, set(), float("inf"), False, False,
                         "cannot compute: " + ", ".join(sorted(set(missing) or set(start))))
    ch = best["choice"]
    sel = Selection(ch, set(ch) | (init & _inputs_closure(catalog, ch, init)), best["cost"], not stats["cut"])
    if missing:
        sel.why = "not computable (left to the questions to abstain): " + ", ".join(sorted(set(missing)))
    return sel


def _inputs_closure(catalog, choice, init):
    out = set()
    for f, n in choice.items():
        for a in alternatives(catalog.parts[f]):
            if a.name == n:
                out |= set(a.inputs)
    return out


def producer(catalog, fact, name):
    for a in alternatives(catalog.parts[fact]):
        if a.name == name:
            return a
    raise KeyError(f"{name} is not a producer of {fact}")


# ---------------------------------------------------------------------------------------------------------------- flows
def _reads(kept, fact, inputs):
    """Does any of `inputs` depend on `fact` through the producers kept so far (kept: fact → its producers)?"""
    todo, seen = list(inputs), set()
    while todo:
        x = todo.pop()
        if x == fact:
            return True
        if x in seen:
            continue
        seen.add(x)
        for a in kept.get(x, ()):
            todo.extend(a.inputs)
    return False


def _widen(catalog, kept, ok):
    """Add to each fact's kept producers its other producers that pass `ok` and do not make the fact depend on itself
    (facts derivable from each other — net from gross, gross from net: only one direction can stay) → kept, each fact's
    producers in declaration order after those it already had."""
    for f, alts in kept.items():
        g = catalog.parts[f]
        for a in g.alternatives or ():
            if not any(a is k for k in alts) and ok(a) and not _reads(kept, f, a.inputs):
                alts.append(a)
    return kept


def _narrow(g, alts):
    n = dataclasses.replace(g, alternatives=alts, inputs=list(dict.fromkeys(x for a in alts for x in a.inputs)))
    n.func = _group_func(n)
    return n


def view_usable(catalog, reach):
    """The catalog with every fact's producers narrowed to the usable ones (dead ends dropped), all kept as fallbacks —
    except a usable producer that would make its fact depend on itself (facts derivable from each other): of such a
    ring, the producers that make each fact computable from the given ones stay, in declaration order."""
    v = Catalog.__new__(Catalog)
    v.__dict__.update(catalog.__dict__)
    v.parts = dict(catalog.parts)
    have = {x for x in reach if x not in catalog.parts}          # the given facts
    kept, changed = {}, True
    while changed:                                    # each fact's first producer whose inputs are already computable:
        changed = False                               # an acyclic skeleton, then widened with the rest
        for f, g in catalog.parts.items():
            if f in kept or f not in reach:
                continue
            a = next((a for a in alternatives(g) if all(x in have or x in kept for x in a.inputs)), None)
            if a is not None:
                kept[f] = [a]
                changed = True
    _widen(catalog, kept, lambda a: all(x in reach for x in a.inputs))
    for f, alts in kept.items():
        g = catalog.parts[f]
        if g.alternatives is None or len(alts) == len(g.alternatives):
            continue
        v.parts[f] = _narrow(g, [a for a in g.alternatives if any(a is k for k in alts)])
    return v


def mandatory_checks(catalog, questions, init_keys, heads=None):
    """Mandatory milestones: the hard checks that govern each question in the flow the deterministic strategist would build
    if every fact needed only its usable producers (dead ends dropped) — independent of which producer a plan chooses, so a
    shortcut that skips the fact such a check reads cannot drop the check. → {question: [check]}"""
    reach = reachable(catalog, init_keys)
    flow = det_plan(view_usable(catalog, reach), list(questions), init_keys, heads)
    return governing(catalog, flow, questions)


def view(catalog, choice, fallbacks=True, reach=None):
    """A shallow copy of the catalog in which every fact with alternative producers keeps only the chosen one (first) and,
    with fallbacks, the other producers whose inputs the plan already computes (after it, in declaration order) — unless
    such a producer reads, through any fact, the fact it would produce (then the plan could not run)."""
    from .core import Catalog
    v = Catalog.__new__(Catalog)
    v.__dict__.update(catalog.__dict__)
    v.parts = dict(catalog.parts)
    have = set(choice) | set(reach or ())
    kept = {f: [producer(catalog, f, name)] for f, name in choice.items()}
    if fallbacks:
        _widen(catalog, kept, lambda a: all(x in have for x in a.inputs))
    for f, alts in kept.items():
        g = catalog.parts[f]
        if g.alternatives is not None:
            v.parts[f] = _narrow(g, alts)
    return v


def with_checkpoints(questions, gov):
    out = []
    for q in questions:
        extra = [c for c in gov.get(q.name, ()) if c not in q.requires]
        out.append(dataclasses.replace(q, requires=list(q.requires) + extra) if extra else q)
    return out


def governing(catalog, flow, questions):
    """Hard checks in a flow that govern each question (they decide its answer when false) → {question: [check]}."""
    names = {s.part.name for s in flow.steps}
    out = {}
    for q in questions:
        fs = set(flow.per_question.get(q.name, ()))
        out[q.name] = [n for n in catalog.parts if n in names and n in fs and catalog.parts[n].kind == "check"
                       and catalog.parts[n].hard and governs(catalog.parts[n], q)]
    return out


def validate(catalog, flow, questions, init_keys, mandatory=None):
    """The plan checks → [problem]: every step's inputs are given or made by an earlier step (for a fact with alternative
    producers: every producer kept in the flow), the order is acyclic, producer and reader types fit, no question is left
    unresolved, and every mandatory hard check of a question is in its flow."""
    bad = []
    have = set(init_keys)
    seen = set()
    for i, st in enumerate(flow.steps):
        p = st.part
        if p.kind == "rule":
            ins = p.inputs
        else:
            ins = [x for a in alternatives(p) for x in a.inputs]
        for x in dict.fromkeys(ins):
            if x not in have:
                bad.append(f"step {i + 1} {p.name}: input {x} is not bound (not given, not made earlier)")
        for a in alternatives(p) if p.kind != "rule" else [p]:
            for x, t in (a.types or {}).items():
                if x in have and not type_ok(catalog, x, t):
                    bad.append(f"step {i + 1} {a.name}: {x} does not fit its type")
        if p.name in seen and p.kind != "rule":
            bad.append(f"{p.name} appears twice")
        seen.add(p.name)
        if p.kind != "rule":
            have.add(p.name)
    for q in questions:
        if flow.unresolved.get(q.name):
            bad.append(f"{q.name}: cannot compute " + ", ".join(flow.unresolved[q.name]))
        fs = set(flow.per_question.get(q.name, ()))
        for c in (mandatory or {}).get(q.name, ()):
            if c not in fs:
                bad.append(f"{q.name}: mandatory hard check {c} is not in the flow")
    return bad


def build(catalog, questions, init_keys, sel, heads=None, fallbacks=True, gov=None):
    """Selection → Flow (the deterministic strategist on the narrowed catalog, with mandatory checks as required parts)."""
    qs = with_checkpoints(questions, gov or {})
    return det_plan(view(catalog, sel.choice, fallbacks, reachable(catalog, init_keys) & (sel.needed | set(init_keys))),
                    qs, init_keys, heads)


# ---------------------------------------------------------------------------------------------------------------- strategists
class CostStrategist:
    """The code strategist for System(..., strategist=...): dead ends dropped, the cheapest verified plan by declared (or
    measured) costs — no model.

    producers: how to treat the alternative producers of a fact (`provides=`):
      "declared" (default) — as the deterministic strategist does: a fallback chain in declaration order (the first is the
        preferred one), except that producers whose inputs cannot be computed (dead ends) are dropped instead of making the
        whole fact unreachable. Answers are those of the deterministic strategist wherever it can answer.
      "equivalent" — the producers of a fact are interchangeable (any accepted output is the same fact): one is chosen as
        the primary, the cheapest valid plan by declared `cost=` (unit when undeclared, or measured run times with
        System(cost_policy="measured")); the others stay as run-time fallbacks when their inputs are already computed
        (keep_alternatives=True).
    on_failure: when the verified plan cannot be built — "code" (the code plan), "deterministic" (solvi.strategist.plan)
    or "abstain" (every question abstains). keep_alternatives: keep the other producers of a fact as run-time fallbacks.
    record: write the plan record into the trace. (0.7 names, removed in 0.9: fallback= for on_failure=, fallbacks= for
    keep_alternatives=.)"""

    @_deprecate.removed_kwargs(fallback="on_failure", fallbacks="keep_alternatives")
    def __init__(self, producers="declared", on_failure="code", costs=None, keep_alternatives=True, record=True):
        if on_failure not in ("code", "deterministic", "abstain"):
            raise ValueError('on_failure must be "code", "deterministic" or "abstain"')
        if producers not in ("declared", "equivalent"):
            raise ValueError('producers must be "declared" or "equivalent"')
        self.producers, self.on_failure, self.costs = producers, on_failure, costs
        self.keep_alternatives, self.record = keep_alternatives, record
        self.last = None

    fallback = _deprecate.removed_attr("fallback", "on_failure", "CostStrategist")
    fallbacks = _deprecate.removed_attr("fallbacks", "keep_alternatives", "CostStrategist")

    def plan(self, catalog, questions, init_keys, heads=None, costs=None):
        """→ Flow. costs: {producer: cost} from the caller (System(cost_policy="measured") passes measured run times), under the
        strategist's own `costs=` (which wins where both name a producer)."""
        t0 = time.perf_counter()
        init = set(init_keys)
        qs = list(questions)
        costs = self.costs if costs is None else {**costs, **(self.costs or {})}
        return self._plan(catalog, qs, init, heads, t0, costs)

    def _plan(self, catalog, qs, init, heads, t0, costs):
        if self.producers == "declared":
            reach = reachable(catalog, init)
            flow = det_plan(view_usable(catalog, reach), qs, init, heads)
            choice = {}
            for st in flow.steps:
                p = st.part
                if p.kind == "rule":
                    continue
                choice[p.name] = (p.alternatives[0].name if p.alternatives else None) if p.alternatives is not None else p.name
            report = {"strategist": "declared", "segments": [], "fallback": None, "mandatory": governing(catalog, flow, qs),
                      "code_cost": None, "proven": True, "cost": None}
            sel = Selection(choice, set(), float("nan"))
            return self._done(flow, report, t0, sel)
        gov = mandatory_checks(catalog, qs, init, heads)
        must = [c for cs in gov.values() for c in cs]
        code = search(catalog, qs, init, costs, heads, extra=must)
        report = {"strategist": "code", "segments": [], "fallback": None,
                  "code_cost": code.cost, "proven": code.proven, "mandatory": gov}
        if not code.feasible:
            flow = det_plan(catalog, qs, init, heads)
            report["fallback"] = "no feasible selection: " + code.why
            return self._done(flow, report, t0, None)
        sel = code
        flow = build(catalog, qs, init, code, heads, self.keep_alternatives, gov)
        bad = validate(catalog, flow, qs, init, gov)
        if bad:
            report["fallback"] = "code plan rejected: " + "; ".join(bad[:3])
            if self.on_failure == "deterministic":
                flow, sel = det_plan(catalog, qs, init, heads), None
            elif self.on_failure == "abstain":
                flow, sel = _abstain(qs, "strategist: " + report["fallback"]), None
        report["cost"] = sel.cost if sel is not None else None
        return self._done(flow, report, t0, sel)

    def _done(self, flow, report, t0, sel):
        report["ms"] = (time.perf_counter() - t0) * 1000
        report["choice"] = dict(sel.choice) if sel is not None else None
        report["model"] = None                        # the model strategist's place in the report (removed in 1.0)
        flow.strategy = report
        self.last = report
        return flow



def _abstain(questions, why):
    return Flow([], {q.name: [] for q in questions}, {}, {q.name: [why] for q in questions})


# ---------------------------------------------------------------------------------------------------------------- trace
def plan_record(flow):
    """The hashed trace record of a planned flow (kind "plan"): the chosen producers and the mandatory checks."""
    from .runtime import Record
    s = getattr(flow, "strategy", None)
    if s is None:
        return None
    value = {"choice": s.get("choice") or {}, "mandatory": s.get("mandatory") or {}}
    extra = {"strategist": s["strategist"], "segments": []}      # "segments": the model strategist's (removed in 1.0)
    if s.get("fallback"):
        extra["fallback"] = s["fallback"]
    if s.get("costs"):                                # costs from measurements (System(cost_policy="measured")): why each choice
        extra["costs"] = s["costs"]
    return Record(step=0, kind="plan", name="plan:strategy", inputs={}, value=value, provenance="computed", model=None,
                  extra=extra)


__all__ = ["CostStrategist", "Selection", "search", "validate", "build", "view", "reachable",
           "plan_record", "replay_plan", "PlanError", "alternatives", "usable", "narrowed", "mandatory_checks", "governing",
           "producer"]
