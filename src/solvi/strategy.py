"""Model strategist (experimental, exps_v2 L18): the typed decomposer of the L3–L6 research series as a solvi strategist.

The deterministic strategist (solvi.strategist) walks back from each question's targets by exact names and, for a fact with
alternative producers (`provides=`), needs the inputs of ALL of them (a fallback chain). This module plans differently:

1. **Points — code.** A branch-and-bound search picks ONE producer per needed fact so that the plan is valid and cheapest
   (declared `cost=`; a part without a declared cost counts as `unit`). A producer whose inputs cannot be computed from the
   given facts (a dead end) is never chosen. Hard checks that govern a question (its `then` names the question, it has no
   `then`, or the question names it as a checkpoint) and that the cheapest plan contains are **mandatory milestones**:
   every later plan must keep them.
2. **Segments — the model.** Where the choice is not settled by declared costs (a fact with several usable producers, not
   all of them with a declared cost), the model gets a short task — "produce this fact from these available facts; these are
   the candidate parts (narrowed by code)" — and proposes 1–4 parts, in order. All segments of a plan go through the model in
   one batch.
3. **Verification — code.** Each segment is checked (every part is a candidate, the last one provides the fact, every input is
   available or made earlier in the segment, types fit); an accepted segment fixes those choices and the search completes
   the rest; the whole plan is checked again (inputs bound, acyclic, types, every mandatory hard check kept). A rejected
   segment falls back to the code choice for that fact; a plan that fails the final check falls back to the code plan
   (`fallback="code"`), to the deterministic strategist (`"deterministic"`) or abstains (`"abstain"`).

So a model error costs cost or coverage, never a silent wrong wiring: every candidate of a segment provides the same fact
(that is what `provides=` declares) and runs its own validator at run time, names bind exactly, and only verified plans run.
The plan is recorded in the trace as one hashed record (kind "plan"; provenance "proposed" when the model chose any part,
with the model's fingerprint and, per segment, what was proposed and why it was accepted or rejected); `trace.replay`
re-verifies it against the catalog.

    from solvi.strategy import ModelStrategist
    system = System(cat, questions, strategist=ModelStrategist.load("path/to/strategist-checkpoint"))
    system = System(cat, questions, strategist=ModelStrategist())      # code only: cheapest verified plan, no model

Experimental: see docs/strategist.md for what was measured (L18) and when the model helps at all."""
from __future__ import annotations

import dataclasses
import time

from .strategist import Flow, PlanError, plan as det_plan

UNIT = 1.0
MAX_NODES = 4


# ---------------------------------------------------------------------------------------------------------------- catalog
def alternatives(p):
    """The producers of a part's fact: its alternatives, or the part itself."""
    return list(p.alternatives) if p.alternatives is not None else [p]


def fact_of(a):
    """The fact a producer sets (its `provides`, else its name)."""
    return a.provides or a.name


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
    return not part.then or q.name in part.then or part.name in q.checkpoints


# ---------------------------------------------------------------------------------------------------------------- search
@dataclasses.dataclass
class Selection:
    """One producer per needed fact (names), the facts the plan needs, its cost and whether the search proved it cheapest."""
    choice: dict                      # fact → producer name (an alternative's name, or the part's own name)
    needed: set
    cost: float
    proven: bool = True
    expanded: int = 0
    feasible: bool = True
    why: str = ""


def search(catalog, questions, init_keys, costs=None, heads=None, fixed=None, extra=(), max_expand=20000, unit=UNIT,
           method="auto"):
    """Cheapest valid selection for the questions' targets, their checkpoints and `extra` parts (e.g. mandatory checks).
    fixed: {fact: producer name} choices that must be kept (a model's accepted segments).
    method: "milp" (exact: a 0/1 program solved by scipy's HiGHS), "bnb" (branch and bound, capped at max_expand nodes)
    or "auto" (milp when scipy has it). Ties go to the producer declared first.
    → Selection (feasible=False when a target cannot be computed; `why` says which)."""
    if method in ("auto", "milp"):
        try:
            from scipy.optimize import milp  # noqa: F401
            return _search_milp(catalog, questions, init_keys, costs, heads, fixed, extra, unit)
        except ImportError:
            if method == "milp":
                raise
    return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, max_expand, unit)


def _start(catalog, questions, init, reach, heads, extra):
    start, missing = [], []
    for q in questions:
        for f in targets_of(catalog, q, reach, init, heads) + list(q.checkpoints):
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
        return Selection({}, set(), float("inf") if missing else 0.0, True, 0, not missing,
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
        return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, 20000, unit)
    if res.x is None:
        return Selection({}, set(), float("inf"), False, 0, False, "no valid plan: " + str(res.message))
    x = res.x
    choice = {f: a.name for k, (f, a) in enumerate(alts) if x[nf + k] > 0.5}
    cost = sum(c(a) for k, (f, a) in enumerate(alts) if x[nf + k] > 0.5)
    if not _acyclic(catalog, choice, init):
        return _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, 20000, unit)
    sel = Selection(choice, set(choice) | (init & _inputs_closure(catalog, choice, init)), cost, res.status == 0, 0)
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


def _search_bnb(catalog, questions, init_keys, costs, heads, fixed, extra, max_expand, unit):
    init = set(init_keys)
    reach = reachable(catalog, init)
    c = cost_fn(costs, unit)
    fixed = dict(fixed or {})
    start = []
    missing = []
    for q in questions:
        for f in targets_of(catalog, q, reach, init, heads) + list(q.checkpoints):
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
                if stats["n"] >= max_expand:
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
        return Selection({}, set(), float("inf"), False, stats["n"], False,
                         "cannot compute: " + ", ".join(sorted(set(missing) or set(start))))
    ch = best["choice"]
    sel = Selection(ch, set(ch) | (init & _inputs_closure(catalog, ch, init)), best["cost"], not stats["cut"], stats["n"])
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
def view_usable(catalog, reach):
    """The catalog with every fact's producers narrowed to the usable ones (dead ends dropped), all kept as fallbacks."""
    from .core import Catalog, _group_func
    v = Catalog.__new__(Catalog)
    v.__dict__.update(catalog.__dict__)
    v.parts = dict(catalog.parts)
    for f, g in catalog.parts.items():
        if g.alternatives is None or f not in reach:
            continue
        alts = [a for a in g.alternatives if all(x in reach for x in a.inputs)]
        if len(alts) == len(g.alternatives):
            continue
        n = dataclasses.replace(g, alternatives=alts, inputs=list(dict.fromkeys(x for a in alts for x in a.inputs)))
        n.func = _group_func(n)
        v.parts[f] = n
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
    with fallbacks, the other producers whose inputs the plan already computes (after it, in declaration order)."""
    from .core import Catalog, _group_func
    v = Catalog.__new__(Catalog)
    v.__dict__.update(catalog.__dict__)
    v.parts = dict(catalog.parts)
    have = set(choice) | set(reach or ())
    for f, name in choice.items():
        g = catalog.parts[f]
        if g.alternatives is None:
            continue
        first = producer(catalog, f, name)
        rest = [a for a in g.alternatives if a is not first and all(x in have for x in a.inputs)] if fallbacks else []
        alts = [first] + rest
        n = dataclasses.replace(g, alternatives=alts, inputs=list(dict.fromkeys(x for a in alts for x in a.inputs)))
        n.func = _group_func(n)
        v.parts[f] = n
    return v


def with_checkpoints(questions, gov):
    out = []
    for q in questions:
        extra = [c for c in gov.get(q.name, ()) if c not in q.checkpoints]
        out.append(dataclasses.replace(q, checkpoints=list(q.checkpoints) + extra) if extra else q)
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
    """Selection → Flow (the deterministic strategist on the narrowed catalog, with mandatory checks as checkpoints)."""
    qs = with_checkpoints(questions, gov or {})
    return det_plan(view(catalog, sel.choice, fallbacks, reachable(catalog, init_keys) & (sel.needed | set(init_keys))),
                    qs, init_keys, heads)


# ---------------------------------------------------------------------------------------------------------------- segments
def segments(catalog, questions, init_keys, sel, costs=None, depth=3, all_facts=False):
    """The facts whose producer the model should choose, each as a short task (narrowed by code):
    {"fact", "type", "question", "available": [(fact, type, how)], "candidates": [part info], "code": [code's parts]}.
    A fact is a segment when it has ≥ 2 usable producers and not all of them declare a cost (else code decides);
    all_facts=True: every fact with ≥ 2 producers (training / evaluation)."""
    from .typed import type_name
    init = set(init_keys)
    reach = reachable(catalog, init)
    ch = sel.choice
    # what depends on what in the code plan (to keep a segment's available facts upstream of its fact)
    ins = {f: set(producer(catalog, f, n).inputs) for f, n in ch.items()}
    down = {}

    def downstream(f):
        if f not in down:
            down[f] = {g for g, xs in ins.items() if f in xs}
            for g in list(down[f]):
                down[f] |= downstream(g)
        return down[f]
    qtext = "; ".join(q.text for q in questions)
    out = []
    for f in sorted(ch, key=lambda f: list(catalog.parts).index(f)):
        g = catalog.parts[f]
        if g.alternatives is None:
            continue
        us = usable(catalog, f, reach)
        declared = all(a.cost is not None or (costs and a.name in costs) for a in us)
        if not all_facts and (len(us) < 2 or declared):
            continue
        avail = (init | set(ch)) - {f} - downstream(f)
        cand, seen_f = [], set()
        frontier, d = [f], 0
        while frontier and d < depth:
            nxt = []
            for x in frontier:
                if x in seen_f or x not in catalog.parts:
                    continue
                seen_f.add(x)
                for a in alternatives(catalog.parts[x]):
                    cand.append(_info(catalog, a, x, d + 1, costs))
                    nxt += [y for y in a.inputs if y not in avail and y not in init]
            frontier, d = nxt, d + 1
        how = {x: ("given" if x in init else "computed") for x in avail}
        av = sorted((x, type_name(catalog.types[x]) if x in catalog.types else _given_type(catalog, x), how[x])
                    for x in avail if x in init or x in catalog.parts)
        code_nodes = _code_nodes(catalog, f, ch, avail)
        out.append({"fact": f, "type": type_name(catalog.types[f]) if f in catalog.types else "any", "question": qtext,
                    "available": av, "candidates": cand, "code": code_nodes})
    return out


def _given_type(catalog, x):
    from .typed import type_name
    rs = catalog.readers.get(x)
    return type_name(next(iter(rs.values()))) if rs else "any"


def _info(catalog, a, fact, depth, costs):
    from .typed import type_name
    c = costs.get(a.name) if costs and a.name in costs else a.cost
    return {"name": a.name, "fact": fact, "kind": a.kind, "provides": a.provides,
            "params": [(x, type_name((a.types or {}).get(x)) if (a.types or {}).get(x) is not None else "any") for x in a.inputs],
            "returns": type_name(a.returns) if a.returns is not None else "any", "doc": a.doc or "", "cost": c, "depth": depth}


def _code_nodes(catalog, f, choice, avail):
    """The code plan's parts for this segment: the chosen producer of f and, before it, those of its inputs not available."""
    out, seen = [], set()

    def visit(x):
        if x in seen or x in avail or x not in choice:
            return
        seen.add(x)
        a = producer(catalog, x, choice[x])
        for y in a.inputs:
            visit(y)
        out.append(a.name)
    visit(f)
    return out


def check_segment(catalog, seg, nodes):
    """A proposed segment (part names, in order) → None if acceptable, else why not."""
    if not nodes:
        return "empty segment"
    if len(nodes) > MAX_NODES:
        return f"{len(nodes)} parts (at most {MAX_NODES})"
    by = {c["name"]: c for c in seg["candidates"]}
    if len(set(nodes)) != len(nodes):
        return "a part appears twice"
    have = {x for x, _, _ in seg["available"]}
    made = []
    for n in nodes:
        if n not in by:
            return f"{n} is not a candidate of this segment"
        c = by[n]
        a = producer(catalog, c["fact"], n)
        for x in a.inputs:
            if x not in have:
                return f"{n}: input {x} is not available"
            t = (a.types or {}).get(x)
            if t is not None and not type_ok(catalog, x, t):
                return f"{n}: {x} does not fit its type"
        if c["fact"] in have and c["fact"] != seg["fact"]:
            return f"{n}: {c['fact']} is already available"
        have.add(c["fact"])
        made.append(c["fact"])
    if made[-1] != seg["fact"]:
        return f"the last part does not provide {seg['fact']}"
    if seg["fact"] in made[:-1]:
        return f"{seg['fact']} is produced twice"
    need = set()
    for n in nodes[1:]:
        need |= set(producer(catalog, by[n]["fact"], n).inputs)
    idle = [m for m in made[:-1] if m not in need]
    if idle:
        return "unused parts: " + ", ".join(idle)
    return None


# ---------------------------------------------------------------------------------------------------------------- strategists
class ModelStrategist:
    """A strategist for System(..., strategist=...).

    producers: how to treat the alternative producers of a fact (`provides=`):
      "declared" (default) — as the deterministic strategist does: a fallback chain in declaration order (the first is the
        preferred one), except that producers whose inputs cannot be computed (dead ends) are dropped instead of making the
        whole fact unreachable. Answers are those of the deterministic strategist wherever it can answer; nothing is left
        to choose, so a model is never asked.
      "equivalent" — the producers of a fact are interchangeable (any accepted output is the same fact): one is chosen as
        the primary, the cheapest valid plan by declared `cost=` (unit when undeclared); where declared costs do not settle
        it, the model (if any) proposes the segment; the others stay as run-time fallbacks when their inputs are already
        computed (fallbacks=True).
    fallback: when the verified plan cannot be built — "code" (the code plan), "deterministic" (solvi.strategist.plan) or
    "abstain" (every question abstains). record: write the plan record into the trace."""

    def __init__(self, model=None, producers="declared", fallback="code", costs=None, fallbacks=True, record=True,
                 max_expand=20000):
        if fallback not in ("code", "deterministic", "abstain"):
            raise ValueError('fallback must be "code", "deterministic" or "abstain"')
        if producers not in ("declared", "equivalent"):
            raise ValueError('producers must be "declared" or "equivalent"')
        self.model, self.producers, self.fallback, self.costs = model, producers, fallback, costs
        self.fallbacks, self.record, self.max_expand = fallbacks, record, max_expand
        self.last = None

    @classmethod
    def load(cls, path, backend="auto", producers="equivalent", threads=None, quantized=False, **kw):
        """A trained segment model (a directory or a Hugging Face id; docs/strategist.md) behind this strategist. The model
        chooses among interchangeable producers, so `producers` defaults to "equivalent" here."""
        from .strategy_model import SegmentModel
        return cls(SegmentModel.load(path, backend=backend, threads=threads, quantized=quantized), producers=producers, **kw)

    @property
    def fingerprint(self):
        return None if self.model is None else self.model.fingerprint

    def info(self):
        """{"type", "id", "fp"} of the model (as in trace records), or None."""
        return None if self.model is None else self.model.info()

    def plan(self, catalog, questions, init_keys, heads=None, costs=None):
        """→ Flow. costs: {producer: cost} from the caller (System(costs="measured") passes measured run times), under the
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
        code = search(catalog, qs, init, costs, heads, extra=must, max_expand=self.max_expand)
        report = {"strategist": "model" if self.model is not None else "code", "segments": [], "fallback": None,
                  "code_cost": code.cost, "proven": code.proven, "mandatory": gov}
        if not code.feasible:
            flow = det_plan(catalog, qs, init, heads)
            report["fallback"] = "no feasible selection: " + code.why
            return self._done(flow, report, t0, None)
        sel, code_flow = code, build(catalog, qs, init, code, heads, self.fallbacks, gov)
        segs = segments(catalog, qs, init, code, costs) if self.model is not None else []
        fixed = {}
        if segs:
            props = self.model.propose(segs)
            for seg, prop in zip(segs, props):
                row = {"fact": seg["fact"], "code": seg["code"], "proposed": list(prop["nodes"]),
                       "score": prop.get("score"), "candidates": len(seg["candidates"])}
                why = check_segment(catalog, seg, prop["nodes"])
                if why is not None and prop.get("second"):
                    why2 = check_segment(catalog, seg, prop["second"])
                    row["first_rejected"] = why
                    if why2 is None:
                        row["proposed"], why = list(prop["second"]), None
                if why is None:
                    by = {c["name"]: c["fact"] for c in seg["candidates"]}
                    for n in row["proposed"]:
                        fixed[by[n]] = n
                    row["accepted"] = "verified: every input available, types fit, provides " + seg["fact"]
                    row["by"] = "model"
                else:
                    row["rejected"] = why
                    row["by"] = "code"
                report["segments"].append(row)
            if fixed:
                sel = search(catalog, qs, init, costs, heads, fixed=fixed, extra=must, max_expand=self.max_expand)
                ignored = [f for f, n in fixed.items() if sel.choice.get(f) not in (None, n)]
                if ignored:
                    report["overridden"] = ignored
        flow = build(catalog, qs, init, sel, heads, self.fallbacks, gov) if sel is not code else code_flow
        bad = validate(catalog, flow, qs, init, gov)
        if bad and sel is not code:
            report["fallback"] = "plan rejected: " + "; ".join(bad[:3])
            bad0 = validate(catalog, code_flow, qs, init, gov)
            if self.fallback == "code" and not bad0:
                flow, sel = code_flow, code
            elif self.fallback == "deterministic":
                flow, sel = det_plan(catalog, qs, init, heads), None
            else:
                flow, sel = _abstain(qs, "strategist: " + report["fallback"]), None
        elif bad:
            report["fallback"] = "code plan rejected: " + "; ".join(bad[:3])
            if self.fallback == "deterministic":
                flow, sel = det_plan(catalog, qs, init, heads), None
            elif self.fallback == "abstain":
                flow, sel = _abstain(qs, "strategist: " + report["fallback"]), None
        report["cost"] = sel.cost if sel is not None else None
        return self._done(flow, report, t0, sel)

    def _done(self, flow, report, t0, sel):
        report["ms"] = (time.perf_counter() - t0) * 1000
        report["choice"] = dict(sel.choice) if sel is not None else None
        report["model"] = self.info()
        flow.strategy = report
        self.last = report
        return flow


def _abstain(questions, why):
    return Flow([], {q.name: [] for q in questions}, {}, {q.name: [why] for q in questions})


# ---------------------------------------------------------------------------------------------------------------- trace
def plan_record(flow):
    """The hashed trace record of a planned flow (kind "plan"): the chosen producers, the mandatory checks and, per segment,
    what the model proposed and why it was accepted or rejected."""
    from .runtime import Record
    s = getattr(flow, "strategy", None)
    if s is None:
        return None
    by_model = any(r.get("by") == "model" for r in s.get("segments", ()))
    value = {"choice": s.get("choice") or {}, "mandatory": s.get("mandatory") or {}}
    extra = {"strategist": s["strategist"], "segments": [{k: r[k] for k in ("fact", "proposed", "code", "by", "accepted",
                                                                               "rejected", "first_rejected") if k in r}
                                                          for r in s.get("segments", ())]}
    if s.get("fallback"):
        extra["fallback"] = s["fallback"]
    if s.get("costs"):                                # costs from measurements (System(costs="measured")): why each choice
        extra["costs"] = s["costs"]
    return Record(step=0, kind="plan", name="plan:strategy", inputs={}, value=value,
                  provenance="proposed" if by_model else "computed", model=s.get("model") if by_model else None, extra=extra)


def replay_plan(r, catalog, init_keys):
    """Re-verify a plan record against the catalog: every chosen producer exists and provides its fact, its inputs are given
    or chosen facts, and the choice is acyclic → [(step, name, reason)]."""
    bad = []
    v = r.value if isinstance(r.value, dict) else {}
    ch = v.get("choice") or {}
    init = set(init_keys)
    for f, n in ch.items():
        if f not in catalog.parts:
            bad.append((r.step, r.name, f"plan chose a producer of {f}, which is not in the catalog"))
            continue
        try:
            a = producer(catalog, f, n)
        except KeyError:
            bad.append((r.step, r.name, f"{n} is not a producer of {f}"))
            continue
        lost = [x for x in a.inputs if x not in init and x not in ch]
        if lost:
            bad.append((r.step, r.name, f"{n}: inputs {', '.join(lost)} are neither given nor chosen"))
    return bad


def narrowed(group, tried):
    """A fact's producer group narrowed to the producers that ran (replay of a plan that dropped the others)."""
    from .core import _group_func
    names = [n for n, _ in tried]
    alts = [a for a in group.alternatives if a.name in names] or list(group.alternatives)
    g = dataclasses.replace(group, alternatives=alts, inputs=list(dict.fromkeys(x for a in alts for x in a.inputs)))
    g.func = _group_func(g)
    return g


__all__ = ["ModelStrategist", "Selection", "search", "segments", "check_segment", "validate", "build", "view", "reachable",
           "plan_record", "replay_plan", "PlanError"]
