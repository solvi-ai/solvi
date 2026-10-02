"""Decisions over a set: the answers of many items made consistent under set-level constraints — at most one / exactly
one "yes" per key (one counterpart per product, one owner per record), mutual exclusion, a capacity per key or per
answer (at most 3 tasks per shift) — from the items' own answers and probabilities.

    from solvi.sets import AtMostOne, Item, decide_set

    items = [Item.of(system.ask(s), "match", id=(s["abt"], s["buy"]), keys={"abt": s["abt"], "buy": s["buy"]})
             for s in states]
    out = decide_set(items, [AtMostOne("abt"), AtMostOne("buy")])
    out[("a1", "b7")].answer, out[("a1", "b7")].why   # "changed from 'yes' to 'no' to satisfy at_most_one(abt) on
                                                      #  abt=a1: ('a1', 'b3') holds 'yes' (0.99)"
    out.changed, out.feasible, out.method, out.components
    out.replay()                                      # the record re-checked and re-solved: {"ok", "mismatches", ...}
    SetDecision.from_dict(out.to_dict())              # JSON in, JSON out

What is chosen: the most probable combination of answers that satisfies every constraint — the product of the items'
probabilities, taken as independent (the objective of the constraints between answers inside one request). An item that
is not free — an answer without probabilities (a rule's, a hard check's, a forced one), an abstention, a multi-label,
span, ranking or estimate answer — never changes; its answer counts in the groups as given (an abstention counts
nowhere). A free item's answer may change to any answer it gives a probability above 0.

How. A constraint puts items in groups (by a key, by explicit lists of ids, or all items in one) and bounds how many of
each group hold a counted answer. Only groups that can bind are kept; items linked by them form connected components, and
each component is decided alone: a component whose answers as given already satisfy its groups keeps them (every item at
its most probable answer is the optimum), the others are solved.
- method="exact" (default): an integer program per component (HiGHS through scipy.optimize.milp, relative gap 0) —
  proven optimal unless `time_limit` (seconds per component) stops it; then the component's status says "time limit"
  and the best combination found is used, so the set is not exact and `exact` is False. A tie between equally probable
  combinations is broken by `Item.tie` (a second program: among the optimal combinations, the largest sum of the ties
  of the items kept at their given answer), then by the solver, deterministically for one scipy version.
- method="greedy": the stated approximation, linear in the items. From the surest item down (the probability of its
  given answer, then its tie, then input order), each free item takes its most probable answer whose groups still have
  room; then groups below their minimum take the item that loses least by moving into them. Not optimal: with a1–b1 at
  0.95 and a2–b1, a1–b2 at 0.9 under one counterpart per offer it keeps one pair where the most probable combination
  keeps two; it can also leave a group below its minimum that a search would fill.
A component that cannot satisfy its groups (fixed answers that conflict, a minimum no free item can meet, a greedy that
got stuck) keeps its answers as given: `feasible` is False, `violations` names each broken group, and every item of a
broken group says "not repaired" in its why — nothing is silently left broken.

Every changed item cites what changed it: the group whose maximum its given answer would break, with the items that hold
the group's counted answers ("kept"), or the group whose minimum needed it. The record holds each item's probabilities,
its given and final answer, the groups as evaluated (so replay needs neither the items' objects nor the key functions),
the method and every component's status. `replay()` re-checks the record — the final answers satisfy every group that is
not reported broken, fixed answers are unchanged, every change is cited, the objective is recomputed — and re-solves it:
under method="exact" no combination may be more probable than the recorded one (an equally probable different one is a
note, not a mismatch).

Not done here: constraints that are not counts over groups — transitivity of matches (a~b and b~c → a~c), "if a then b",
sums of weights; soft constraints with a cost; items whose errors are not independent (the objective treats them as
independent); learning anything — the probabilities are the items' own."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

MIN_P = 1e-12                  # an answer with a probability at or below this is never chosen


class _Each:
    """The answer of a capacity group is the answer itself: `Capacity(max=3, answer=EACH)` — at most 3 items per answer."""

    def __repr__(self):
        return "EACH"


EACH = _Each()


# ---------------------------------------------------------------- items
@dataclass
class Item:
    """One item of the set: its id (hashable, JSON-able: a str, an int or a tuple of them), the probabilities of the
    answers it may take (`probs`; None for a fixed item), its answer as given (None: the most probable; for a fixed
    item, the answer that never changes — None is an abstention, counted in no group), the keys constraints group it by,
    a tie score (see the module docs), the stored id of the response it came from and its own reason."""
    id: Any
    probs: dict | None = None
    answer: Any = None
    keys: dict = field(default_factory=dict)
    tie: float = 0.0
    source: Any = None
    why: str = ""

    @property
    def free(self):
        return bool(self.probs)

    @classmethod
    def of(cls, res, question, id=None, keys=None, tie=0.0):
        """An item from a Response's answer to `question`: free when the answer is a yes/no, choice or ordinal answer with
        probabilities and status "ok" (a learned head, a model decision, a rule returning a Decision); fixed otherwise —
        a forced or rule answer without probabilities as given, an abstention as None. The response's stored id is kept
        as the item's source."""
        r = res[question]
        free = r.status == "ok" and bool(r.probs) and r.kind in ("yes_no", "choice", "ordinal")
        return cls(id=id if id is not None else getattr(res, "stored_id", None),
                   probs=dict(r.probs) if free else None,
                   answer=None if r.status == "abstain" else r.answer,
                   keys=dict(keys or {}), tie=float(tie), source=getattr(res, "stored_id", None), why=r.why)


# ---------------------------------------------------------------- constraints
@dataclass
class Capacity:
    """At most `max` (and at least `min`) items of each group hold a counted answer.

    key: how items are grouped — the name of a key in `Item.keys` (an item without it is in no group), a function
    item → key (None: in no group), or None: all items in one group. groups: explicit lists of item ids instead of a key
    (mutual exclusion between listed items). answer: the counted answer, a list of them, or EACH (each answer value is a
    group of its own: a capacity per shift). name: the constraint's name in the record (default from the kind and key)."""
    key: Any = None
    max: int | None = None
    min: int = 0
    answer: Any = "yes"
    groups: list | None = None
    name: str | None = None

    def __post_init__(self):
        if self.max is None and not self.min:
            raise ValueError("a capacity needs max= or min=")
        if self.max is not None and (int(self.max) != self.max or self.max < 0):
            raise ValueError(f"max must be a non-negative integer, not {self.max!r}")
        if int(self.min) != self.min or self.min < 0:
            raise ValueError(f"min must be a non-negative integer, not {self.min!r}")
        if self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min} is above max {self.max}")
        if self.groups is not None and self.key is not None:
            raise ValueError("give key= or groups=, not both")
        if self.name is None:
            self.name = self._default_name()

    def _default_name(self):
        what = ("groups" if self.groups is not None else getattr(self.key, "__name__", None) if callable(self.key)
                else self.key if self.key is not None else "all")
        if self.min == self.max == 1:
            kind = "exactly_one"
        elif self.max == 1 and not self.min:
            kind = "at_most_one"
        else:
            kind = "capacity"
        return f"{kind}({what})"

    def counted(self, answer):
        """Is `answer` counted by this constraint?"""
        if answer is None:
            return False
        if self.answer is EACH:
            return True
        if isinstance(self.answer, (list, tuple, set, frozenset)):
            return answer in self.answer
        return answer == self.answer

    def members(self, items):
        """→ {group label: [(item index, counted answers among its candidates)]} — the groups as evaluated."""
        out = {}
        if self.groups is not None:
            index = {it.id: i for i, it in enumerate(items)}
            for n, g in enumerate(self.groups):
                lost = [x for x in g if x not in index]
                if lost:
                    raise KeyError(f"{self.name}: no item with id {lost[0]!r}")
                for x in g:
                    out.setdefault(f"group {n}", []).append(index[x])
        else:
            for i, it in enumerate(items):
                if self.key is None:
                    k = "all"
                elif callable(self.key):
                    k = self.key(it)
                else:
                    k = it.keys.get(self.key)
                if k is None:
                    continue
                label = k if self.key is None else (f"{self.key}={k}" if isinstance(self.key, str) else str(k))
                out.setdefault(label, []).append(i)
        groups = {}
        for label, idx in out.items():
            if self.answer is EACH:
                for a in sorted({a for i in idx for a in _answers(items[i])}, key=repr):
                    groups[f"{label}, answer={a}"] = [(i, [a]) for i in idx if a in _answers(items[i])]
            else:
                groups[str(label)] = [(i, [a for a in _answers(items[i]) if self.counted(a)]) for i in idx]
        return groups

    def to_dict(self):
        a = "EACH" if self.answer is EACH else (sorted(self.answer, key=repr) if isinstance(self.answer, (set, frozenset))
                                                 else list(self.answer) if isinstance(self.answer, (list, tuple)) else self.answer)
        return {"name": self.name, "min": self.min, "max": self.max, "answer": a,
                "key": self.key if isinstance(self.key, str) or self.key is None else f"function {getattr(self.key, '__name__', '?')}"}


def AtMostOne(key=None, answer="yes", groups=None, name=None):
    """At most one item of each group holds `answer` (one counterpart per product; with groups=, mutual exclusion)."""
    return Capacity(key, max=1, answer=answer, groups=groups, name=name)


def ExactlyOne(key=None, answer="yes", groups=None, name=None):
    """Exactly one item of each group holds `answer` (one owner per record)."""
    return Capacity(key, max=1, min=1, answer=answer, groups=groups, name=name)


def Exclusive(groups, answer="yes", name=None):
    """Mutual exclusion: of each listed group of item ids (pairs, or more), at most one holds `answer`."""
    return Capacity(None, max=1, answer=answer, groups=[list(g) for g in groups], name=name or "exclusive")


def _answers(it):
    """The answers an item may hold: a free item's answers with probability above MIN_P, a fixed item's own."""
    if it.probs:
        return [a for a, p in it.probs.items() if p > MIN_P]
    return [] if it.answer is None else [it.answer]


# ---------------------------------------------------------------- results
@dataclass
class Decided:
    """One item's final answer: `answer`, the answer as given (`was`), its probability (`confidence`), whether it is
    `fixed`, what changed it (`cited`: [{"constraint", "group", "kept": [ids]} or {"constraint", "group", "needed": n}])
    and the reason (`why`: the item's own, then the change)."""
    id: Any
    answer: Any
    was: Any
    confidence: float | None
    fixed: bool
    cited: list = field(default_factory=list)
    why: str = ""
    source: Any = None

    @property
    def changed(self):
        return self.answer != self.was

    def to_dict(self):
        return {"id": _plain(self.id), "answer": self.answer, "was": self.was, "confidence": self.confidence,
                "fixed": self.fixed, "cited": [{**c, **({"kept": [_plain(x) for x in c["kept"]]} if "kept" in c else {})}
                                               for c in self.cited],
                "why": self.why, "source": self.source}


@dataclass
class SetDecision:
    """The decision over a set: `items` (id → Decided, in input order; `out[id]`, iteration), `changed`, `feasible`,
    `violations` ([{"constraint", "group", "count", "min", "max"}]), `method`, `exact` (every component solved to a
    proven optimum), `components` (each solved component: its items, groups, status, objective, seconds), the
    constraints and the input as recorded. `to_dict` / `from_dict`; `replay()`."""
    items: dict
    constraints: list
    method: str
    components: list
    feasible: bool
    violations: list
    exact: bool
    record: dict = field(repr=False, default_factory=dict)

    def __getitem__(self, id):
        return self.items[_key(id)]

    def __iter__(self):
        return iter(self.items.values())

    def __len__(self):
        return len(self.items)

    @property
    def changed(self):
        return [d for d in self.items.values() if d.changed]

    @property
    def answers(self):
        return {k: d.answer for k, d in self.items.items()}

    def __str__(self):
        n = len(self.changed)
        lines = [f"{len(self.items)} items, {n} changed by {', '.join(c['name'] for c in self.constraints)} "
                 f"({self.method}{'' if self.exact or self.method == 'greedy' else ', not proven optimal'}); "
                 f"{len(self.components)} component(s) solved"
                 + ("" if self.feasible else f"; NOT FEASIBLE: {len(self.violations)} group(s) broken")]
        lines += [f"  {d.id!r}: {d.why[d.why.rfind('changed from'):]}" for d in self.changed[:20]]
        if n > 20:
            lines.append(f"  ... {n - 20} more")
        for v in self.violations[:10]:
            lines.append(f"  broken: {v['constraint']} on {v['group']} ({v['count']} counted, "
                         f"min {v['min']}, max {v['max']})")
        return "\n".join(lines)

    def to_dict(self):
        return {"items": [d.to_dict() for d in self.items.values()], "constraints": list(self.constraints),
                "method": self.method, "components": list(self.components), "feasible": self.feasible,
                "violations": list(self.violations), "exact": self.exact, "record": self.record}

    @classmethod
    def from_dict(cls, d):
        items = {}
        for x in d["items"]:
            dd = Decided(_key(x["id"]), x["answer"], x["was"], x["confidence"], x["fixed"],
                         [{**c, **({"kept": [_key(k) for k in c["kept"]]} if "kept" in c else {})} for c in x["cited"]],
                         x["why"], x.get("source"))
            items[dd.id] = dd
        return cls(items, d["constraints"], d["method"], d["components"], d["feasible"], d["violations"], d["exact"],
                   d.get("record", {}))

    def replay(self, time_limit=10.0):
        """Re-check the record and re-solve it → {"ok", "mismatches": [(item or group, what, why)], "notes": [...]}.

        Checked: every final answer is one the item may take (a fixed item's is its own); every group that is not
        reported broken holds between its min and max, and every reported one is broken; every changed item is free
        and cites a constraint; the recorded objective per component is that of the final answers. Then the record is
        solved again with its method: under "exact", a more probable combination than the recorded one is a mismatch, an
        equally probable different one a note; under "greedy", different answers are a mismatch."""
        rec = self.record
        items = [Item(_key(x["id"]), x["probs"], x["answer"], tie=x.get("tie", 0.0)) for x in rec["items"]]
        groups = [(g["constraint"], g["label"], g["min"], g["max"], [(i, a) for i, a in g["members"]]) for g in rec["groups"]]
        final = [self.items[it.id].answer for it in items]
        bad, notes = [], []
        for it, a in zip(items, final):
            d = self.items[it.id]
            if it.probs is None and a != it.answer:
                bad.append((_plain(it.id), "fixed", f"a fixed answer changed from {it.answer!r} to {a!r}"))
            if it.probs is not None and a not in _answers(it):
                bad.append((_plain(it.id), "answer", f"{a!r} is not an answer the item may take"))
            if d.changed and not d.cited:
                bad.append((_plain(it.id), "cited", "changed without a constraint cited"))
        broken = {(v["constraint"], v["group"]) for v in self.violations}
        for name, label, lo, hi, mem in groups:
            n = sum(final[i] in a for i, a in mem)
            ok = n >= lo and (hi is None or n <= hi)
            if ok == ((name, label) in broken):
                bad.append((f"{name} on {label}", "group", f"{n} counted (min {lo}, max {hi}); recorded "
                                                           + ("broken" if not ok else "kept") + " otherwise"))
        for c in self.components:
            got = sum(_logp(items[i], final[i]) for i in c["items"])
            if c.get("objective") is not None and abs(got - c["objective"]) > 1e-6 * (1 + abs(got)):
                bad.append((f"component {c['index']}", "objective", f"recorded {c['objective']:.6g}, the answers give {got:.6g}"))
        again = _solve(items, groups, self.method, time_limit)
        for c_new in again["components"]:
            mine = next((c for c in self.components if c["items"] == c_new["items"]), None)
            if mine is None:
                bad.append((f"component {c_new['index']}", "components", "the record solves different components"))
                continue
            same = all(again["final"][i] == final[i] for i in c_new["items"])
            if same:
                continue
            if self.method == "exact" and c_new["objective"] is not None and mine["objective"] is not None:
                if c_new["objective"] > mine["objective"] + 1e-7 * (1 + abs(mine["objective"])):
                    bad.append((f"component {mine['index']}", "optimum", f"a more probable combination exists "
                                                                         f"({c_new['objective']:.6g} > {mine['objective']:.6g})"))
                else:
                    notes.append(f"component {mine['index']}: re-solved to a different combination of the same probability")
            else:
                bad.append((f"component {mine['index']}", "answers", "re-solved to different answers"))
        return {"ok": not bad, "mismatches": bad, "notes": notes, "items": len(items), "groups": len(groups)}


def _plain(x):
    return list(_plain(y) for y in x) if isinstance(x, tuple) else x


def _key(x):
    return tuple(_key(y) for y in x) if isinstance(x, list) else x


def _logp(it, a):
    if not it.probs:
        return 0.0
    return math.log(max(float(it.probs.get(a, 0.0)), MIN_P))


# ---------------------------------------------------------------- the solver
def decide_set(items, constraints, method="exact", time_limit=10.0):
    """The answers of `items` made consistent under `constraints` (see the module docs) → a SetDecision.

    items: [Item] (Item.of(response, question, ...) for System answers), ids unique. constraints: [Capacity] (AtMostOne,
    ExactlyOne, Exclusive, Capacity). method: "exact" (default; an integer program per connected component) or
    "greedy" (the stated approximation). time_limit: seconds per component for "exact"."""
    if method not in ("exact", "greedy"):
        raise ValueError(f'method must be "exact" or "greedy", not {method!r}')
    items = list(items)
    if not items:
        raise ValueError("no items")
    for c in constraints:
        if not isinstance(c, Capacity):
            raise TypeError(f"a constraint is a Capacity (AtMostOne, ExactlyOne, Exclusive), not {type(c).__name__}")
    names = [c.name for c in constraints]
    if len(set(names)) != len(names):
        raise ValueError("two constraints share a name: " + ", ".join(sorted({n for n in names if names.count(n) > 1})))
    seen = set()
    norm = []
    for it in items:
        if not isinstance(it, Item):
            raise TypeError(f"an item is a solvi.sets.Item, not {type(it).__name__}")
        try:
            hash(it.id)
        except TypeError:
            raise TypeError(f"item id {it.id!r} is not hashable") from None
        if it.id in seen:
            raise ValueError(f"two items share the id {it.id!r}")
        seen.add(it.id)
        norm.append(_normalized(it))
    groups = []
    for c in constraints:
        for label, mem in c.members(norm).items():
            groups.append((c.name, label, c.min, c.max, mem))
    out = _solve(norm, groups, method, time_limit)
    return _result(norm, constraints, groups, method, out)


def _normalized(it):
    """A free item's probabilities checked (finite, non-negative, some above MIN_P) and its given answer settled."""
    if it.probs is None or not it.probs:
        return Item(it.id, None, it.answer, dict(it.keys), float(it.tie), it.source, it.why)
    probs = {}
    for a, p in it.probs.items():
        p = float(p)
        if not math.isfinite(p) or p < 0:
            raise ValueError(f"item {it.id!r}: probability of {a!r} is {p!r}")
        probs[a] = p
    if not any(p > MIN_P for p in probs.values()):
        raise ValueError(f"item {it.id!r}: no answer has a probability above 0")
    ans = it.answer
    if ans is None or probs.get(ans, 0.0) <= MIN_P:
        ans = max(probs, key=lambda a: probs[a])             # first of the most probable, in the probs' order
    return Item(it.id, probs, ans, dict(it.keys), float(it.tie), it.source, it.why)


def _binds(items, lo, hi, mem):
    """Can a group bind at all? (Its members' counted answers can exceed max, or it has a minimum.)"""
    return lo > 0 or (hi is not None and sum(1 for i, a in mem if a) > hi)


def _count(assign, mem):
    return sum(assign[i] in a for i, a in mem)


def _solve(items, groups, method, time_limit):
    """→ {"final": [answer per item], "components": [...]} (only components with a group broken as given are solved)."""
    n = len(items)
    final = [it.answer for it in items]
    live = [g for g in groups if _binds(items, g[2], g[3], g[4])]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, _, _, _, mem in live:
        idx = [i for i, a in mem]
        for i in idx[1:]:
            a, b = find(idx[0]), find(i)
            if a != b:
                parent[max(a, b)] = min(a, b)
    by = {}
    for gi, g in enumerate(live):
        if g[4]:
            by.setdefault(find(g[4][0][0]), []).append(gi)
    comps = []
    for root, gis in sorted(by.items()):
        broken = [gi for gi in gis if not _within(_count(final, live[gi][4]), live[gi][2], live[gi][3])]
        if not broken:
            continue
        idx = sorted({i for gi in gis for i, _ in live[gi][4]})
        gs = [live[gi] for gi in gis]
        t0 = time.perf_counter()
        if method == "exact":
            got, status, gap = _milp(items, idx, gs, time_limit)
        else:
            got, status, gap = _greedy(items, idx, gs), "greedy", None
        feasible = got is not None
        if feasible:
            ok = all(_within(_count({**dict(enumerate(final)), **got}, g[4]), g[2], g[3]) for g in gs)
            if not ok:                                        # the greedy got stuck: answers as given, reported broken
                feasible, status = False, "greedy could not satisfy every group"
        if feasible:
            for i, a in got.items():
                final[i] = a
        else:
            status = status if status != "optimal" else "infeasible"
        comps.append({"index": len(comps), "items": idx, "groups": [f"{g[0]} on {g[1]}" for g in gs],
                      "broken_as_given": len(broken), "status": status, "gap": gap,
                      "objective": sum(_logp(items[i], final[i]) for i in idx),
                      "seconds": round(time.perf_counter() - t0, 4)})
    return {"final": final, "components": comps, "live": live}


def _within(n, lo, hi):
    return n >= lo and (hi is None or n <= hi)


def _milp(items, idx, gs, time_limit):
    """The most probable answers of these items under these groups (an integer program) → ({item: answer}, status, gap);
    (None, status, None) when no combination satisfies them."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    var, cols = [], {}
    fixed = {}
    for i in idx:
        it = items[i]
        if it.probs:
            for a in _answers(it):
                cols[(i, a)] = len(var)
                var.append((i, a))
        else:
            fixed[i] = it.answer
    if not var:
        ok = all(_within(_count(fixed, g[4]), g[2], g[3]) for g in gs)
        return ({} if ok else None), ("optimal" if ok else "infeasible: fixed answers conflict"), None
    rows, cc, vals, lo, hi = [], [], [], [], []
    r = 0
    for i in idx:                                         # each free item holds exactly one answer
        if items[i].probs:
            for a in _answers(items[i]):
                rows.append(r)
                cc.append(cols[(i, a)])
                vals.append(1.0)
            lo.append(1.0)
            hi.append(1.0)
            r += 1
    for name, label, gmin, gmax, mem in gs:
        base = sum(1 for i, a in mem if i in fixed and fixed[i] in a)
        for i, a in mem:
            for x in a:
                if (i, x) in cols:
                    rows.append(r)
                    cc.append(cols[(i, x)])
                    vals.append(1.0)
        lo.append(gmin - base)
        hi.append(np.inf if gmax is None else gmax - base)
        r += 1
    A = coo_matrix((vals, (rows, cc)), shape=(r, len(var))).tocsr()
    w = np.array([_logp(items[i], a) for i, a in var])
    opts = {"time_limit": float(time_limit), "mip_rel_gap": 0.0}
    cons = [LinearConstraint(A, np.array(lo), np.array(hi))]
    res = milp(-w, constraints=cons, integrality=np.ones(len(var)), bounds=Bounds(0, 1), options=opts)
    if res.x is None:
        return None, ("infeasible" if res.status == 2 else f"no solution ({res.message})"), None
    status = "optimal" if res.status == 0 else "time limit"
    gap = getattr(res, "mip_gap", None)
    x = res.x
    ties = np.array([items[i].tie if a == items[i].answer else 0.0 for i, a in var])
    if res.status == 0 and np.any(ties != 0):              # among the optimal combinations: the largest sum of ties
        best = float(w @ np.round(x))
        tol = 1e-9 * (1 + abs(best))
        cons2 = cons + [LinearConstraint(w.reshape(1, -1), best - tol, np.inf)]
        res2 = milp(-ties, constraints=cons2, integrality=np.ones(len(var)), bounds=Bounds(0, 1), options=opts)
        if res2.x is not None and res2.status == 0:
            x = res2.x
    got = {}
    for (i, a), v in zip(var, x):
        if v > 0.5:
            got[i] = a
    return got, status, (None if gap is None else float(gap))


def _greedy(items, idx, gs):
    """The stated approximation (see the module docs) → {item: answer}."""
    member = {}                                           # item → [(group index, counted answers)]
    for k, (_, _, _, _, mem) in enumerate(gs):
        for i, a in mem:
            member.setdefault(i, []).append((k, a))
    count = [0] * len(gs)
    got = {}
    free = [i for i in idx if items[i].probs]
    for i in idx:
        if not items[i].probs:
            got[i] = items[i].answer
            for k, a in member.get(i, []):
                count[k] += got[i] in a

    def room(i, ans):
        return all(gs[k][3] is None or count[k] + 1 <= gs[k][3] for k, a in member.get(i, []) if ans in a)

    order = sorted(free, key=lambda i: (-items[i].probs[items[i].answer], -items[i].tie, i))
    for i in order:
        it = items[i]
        cand = sorted(_answers(it), key=lambda a: (a != it.answer, -it.probs[a]))
        pick = next((a for a in cand if room(i, a)), it.answer)
        got[i] = pick
        for k, a in member.get(i, []):
            count[k] += pick in a
    for k, (_, _, gmin, gmax, mem) in enumerate(gs):     # minimums: the item that loses least moves in
        while count[k] < gmin:
            best = None
            for i, a in mem:
                if not items[i].probs or got[i] in a:
                    continue
                for x in a:
                    if x not in _answers(items[i]) or not room(i, x):
                        continue
                    if any(got[i] in b and count[j] - 1 < gs[j][2] for j, b in member.get(i, [])):
                        continue
                    loss = _logp(items[i], got[i]) - _logp(items[i], x)
                    if best is None or loss < best[0]:
                        best = (loss, i, x)
            if best is None:
                break
            _, i, x = best
            for j, b in member.get(i, []):
                count[j] += (x in b) - (got[i] in b)
            got[i] = x
    return got


def _result(items, constraints, groups, method, out):
    final, comps = out["final"], out["components"]
    by_item = {}
    for g in groups:
        for i, a in g[4]:
            by_item.setdefault(i, []).append(g)
    violations = []
    for name, label, lo, hi, mem in groups:
        n = _count(final, mem)
        if not _within(n, lo, hi):
            violations.append({"constraint": name, "group": label, "count": n, "min": lo, "max": hi})
    broken = {(v["constraint"], v["group"]) for v in violations}
    decided = {}
    at = {it.id: i for i, it in enumerate(items)}

    def holds(k):
        j = at[k]
        return f"{k!r} holds {final[j]!r}" + (f" ({items[j].probs.get(final[j], 0.0):.2f})" if items[j].probs else "")
    for i, it in enumerate(items):
        a = final[i]
        cited, why = [], it.why
        if a != it.answer:
            for name, label, lo, hi, mem in by_item.get(i, []):
                cnt = dict(mem)
                if it.answer in cnt[i] and hi is not None and _count(final, mem) + 1 > hi:
                    kept = [items[j].id for j, b in mem if j != i and final[j] in b]
                    cited.append({"constraint": name, "group": str(label), "kept": kept})
                elif a in cnt[i] and it.answer not in cnt[i] and _count(final, mem) - 1 < lo:
                    cited.append({"constraint": name, "group": str(label), "needed": lo})
            if not cited:                                 # moved for the component's sake (a chain of exchanges)
                cited = [{"constraint": name, "group": str(label), "kept": [items[j].id for j, b in mem
                                                                             if j != i and final[j] in b]}
                         for name, label, lo, hi, mem in by_item.get(i, [])]
            why = (why + "; " if why else "") + f"changed from {it.answer!r} to {a!r} to satisfy " + "; ".join(
                f"{c['constraint']} on {c['group']}: " + (", ".join(holds(k) for k in c["kept"][:3]) or "nothing else holds it"
                                                          if "kept" in c else f"at least {c['needed']} needed")
                for c in cited[:3])
        else:
            nr = [f"{name} on {label}" for name, label, *_ in by_item.get(i, []) if (name, label) in broken]
            if nr:
                why = (why + "; " if why else "") + "not repaired: " + ", ".join(nr[:3]) + " broken"
        decided[it.id] = Decided(it.id, a, it.answer, (it.probs or {}).get(a) if it.probs else None,
                                 not it.probs, cited, why, it.source)
    exact = method == "exact" and all(c["status"] == "optimal" for c in comps if "infeasible" not in c["status"])
    record = {"items": [{"id": _plain(it.id), "probs": it.probs, "answer": it.answer, "tie": it.tie} for it in items],
              "groups": [{"constraint": name, "label": str(label), "min": lo, "max": hi,
                          "members": [[i, list(a)] for i, a in mem]} for name, label, lo, hi, mem in groups
                         if lo > 0 or any(a for _, a in mem)]}
    return SetDecision(decided, [c.to_dict() for c in constraints], method, comps, not violations, violations, exact, record)
