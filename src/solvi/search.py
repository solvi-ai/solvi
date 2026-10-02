"""Search over alternatives: candidates run through a System's checks, the best kept by an objective — a meeting slot, an
order of cities, a day of meetings, an assignment — with pruning of partial candidates by failed hard checks, a bound,
a budget of asks, and a record of what was searched.

    from solvi.search import Tree, search

    run = search(system, {"problem": text}, "accept",
                 space=lambda facts: Tree([], lambda order: [order + [c] for c in facts["trip"].cities if c not in order],
                                          complete=lambda order: len(order) == len(facts["trip"].cities)),
                 into="order", prune=["direct_flights", "events_kept"], keep=2)
    run.best, run.value, run.response            # the best accepted candidate, its objective, its Response (asked in full)
    run.exact, run.why_exact                      # whether the search can say it is the best, and why (or why not)
    run.asked, run.rejected, run.pruned           # what was searched: asks, rejections and cuts by check
    run.replay(system)                            # the winner's trace replays and is accepted; its objective recomputes

A candidate is given to the System under `into` (or, from a dict of domains, as several given facts) next to `state`,
and the System is asked `question`. It is accepted when `accept` says so — as in solvi.refine: "checks" (default: every
hard check that governs the question was evaluated and passed), an answer or a list of answers, or a function of the
Response — and the question did not abstain: a candidate the System could not decide is never chosen.

The space:
  a list, a tuple or any iterable (a generator) — complete candidates, in its order;
  a dict {fact: [values]} — every combination (the first fact outermost), each given as those facts (`into` = None);
  Tree(root, children, complete=None, bound=None) — a depth-first walk: `children(node)` → the nodes one step
      further, a node is a candidate when `complete(node)` (default: when it has no children); a partial node is asked
      only when `prune` is given (to cut below it);
  a function of the facts computed once from `state` (below) → one of the above: the space read from the problem.

What is cut. `prune`: hard checks that, false on a node, are false on every node below it (a city visited twice stays
visited twice, a meeting that cannot be reached stays out of reach) — a node on which one of them is false is not
expanded. A Tree's `bound(node)`: the best objective any candidate below the node can have (at least, when maximizing); a node
whose bound cannot beat what is kept is not expanded. Neither promise is checked: a prune check that is not monotone,
or a bound below a real candidate's value, can cut the best candidate; the record names the checks and the bound it
relied on. `budget`: asks at most.

What is kept: with an `objective` (a function of the candidate, or the name of a fact of its Response; `maximize=True`)
the `keep` best accepted candidates (the first found among equals); without one, the first `keep` accepted in the
space's order, and the search stops there (keep=2 says whether the first is the only one).

Exactness. `run.exact` is True when the search ended by itself — every candidate of the space was asked or cut by a
`prune` check or the `bound` — and then the best is the best of the space under those two promises (and THE only one
when keep ≥ 2 and one was found). It is False when the budget stopped it, and `why_exact` says so with the count.

Facts computed once. The facts the System computes from `state` alone (the problem read into typed facts, by rules or
by a model) are computed once and held for every candidate — only parts that read the candidate (directly or through
other facts) are re-run, so a model reading the problem is called once, not once per candidate. The winner is then
asked again in full with the System itself (stored when the System has storage): `run.response` is an ordinary decision
whose trace replays. If that full ask does not accept it — a held fact that should have been recomputed — the search
escalates and says so, instead of returning it.

Not done here: no proposals by a model (solvi.refine re-asks one; when `run.best` is None because the space was too big
for the budget, a refinement can take over), no search over numbers by bisection (`res.counterfactual` does that), no
parallel asks, and no proof of the `prune` / `bound` promises."""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tree:
    """A space walked depth-first: `root`, `children(node)` → the next nodes, `complete(node)` → is the node a candidate
    (default: when it has no children), `bound(node)` → the best objective any candidate below the node can reach (at
    least, when maximizing; at most, when minimizing) — a node whose bound cannot beat what is kept is not expanded."""
    root: Any
    children: Callable
    complete: Callable | None = None
    bound: Callable | None = None


@dataclass
class SearchRun:
    """The record of one search: what was found (`best`, `value`, `response`, `kept`), what was searched (`asked`,
    `accepted`, `rejected` by deciding check, `pruned` by check and "bound"), how it ended (`exact`, `why_exact`,
    `exhausted`), and why a person gets it when nothing is returned (`escalation`). `to_dict` / `from_dict`; `replay`."""
    question: str
    into: str | None
    space: str
    best: Any = None
    value: Any = None
    response: Any = None
    kept: list = field(default_factory=list)          # [(candidate, objective value)], best first
    asked: int = 0
    accepted: int = 0
    rejected: dict = field(default_factory=dict)      # deciding check (or "abstained: <cause>", "answer <x>") → count
    pruned: dict = field(default_factory=dict)        # prune check or "bound" → nodes not expanded
    exhausted: bool = False                           # the budget stopped the search
    budget: int = 0
    objective: str | None = None
    maximize: bool = True
    prune: list = field(default_factory=list)
    bound: str | None = None
    keep: int = 1
    held: list = field(default_factory=list)          # facts computed once from the state and held for every candidate
    escalation: str | None = None
    seconds: float = 0.0

    @property
    def exact(self):
        return not self.exhausted and (self.best is not None or self.accepted == 0)

    @property
    def why_exact(self):
        if self.exhausted:
            return (f"not exact: the budget of {self.budget} asks stopped the search; the best is the best of what was "
                    "asked")
        if not self.exact:
            return f"no result: {self.escalation}"
        cut = []
        if self.pruned:
            cut.append("pruned by " + ", ".join(f"{k} ({v})" for k, v in self.pruned.items()))
        relies = [f"the prune checks {', '.join(self.prune)} stay false below a node" if self.prune else "",
                  f"the bound {self.bound} is never below a candidate's objective" if self.bound else ""]
        relies = [r for r in relies if r]
        what = ("no candidate of the space is accepted" if self.best is None else "the best of the space by " + (self.objective or "?") if self.objective else
                "the first accepted in the space's order" + (f" (and {'the only one' if self.accepted == 1 else 'not the only one'})"
                                                              if self.keep >= 2 else ""))
        return ("exact: every candidate was asked or cut" + (" — " + "; ".join(cut) if cut else "") + f"; {what}"
                + (", given that " + " and ".join(relies) if relies else ""))

    def __str__(self):
        head = f"search {self.question}: {self.asked} asked, {self.accepted} accepted"
        lines = [head, f"  best: {self.best!r}" + (f" ({self.objective} = {self.value!r})" if self.objective else ""),
                 f"  {self.why_exact}"]
        if self.rejected:
            lines.append("  rejected by: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.rejected.items(), key=lambda t: -t[1])))
        if self.held:
            lines.append("  computed once: " + ", ".join(self.held))
        return "\n".join(lines)

    def to_dict(self):
        from .storage import plain
        return {"question": self.question, "into": self.into, "space": self.space, "best": plain(self.best),
                "value": plain(self.value), "response": self.response.to_dict() if self.response is not None else None,
                "kept": [[plain(c), plain(v)] for c, v in self.kept], "asked": self.asked, "accepted": self.accepted,
                "rejected": dict(self.rejected), "pruned": dict(self.pruned), "exhausted": self.exhausted,
                "budget": self.budget, "objective": self.objective, "maximize": self.maximize, "prune": list(self.prune),
                "bound": self.bound, "keep": self.keep, "held": list(self.held), "escalation": self.escalation,
                "seconds": self.seconds, "exact": self.exact, "why_exact": self.why_exact}

    @classmethod
    def from_dict(cls, d, catalog=None):
        """A stored search back (the response restored with `catalog`'s types, as Response.model_validate does)."""
        from .system import Response
        res = Response.model_validate(d["response"], catalog=catalog) if d.get("response") is not None else None
        return cls(d["question"], d["into"], d["space"], d["best"], d["value"], res, [tuple(x) for x in d["kept"]],
                   d["asked"], d["accepted"], d["rejected"], d["pruned"], d["exhausted"], d["budget"], d["objective"],
                   d["maximize"], d["prune"], d["bound"], d["keep"], d["held"], d.get("escalation"), d.get("seconds", 0.0))

    def replay(self, system, accept="checks", objective=None, trust_models=False):
        """Re-check the result: the winner's trace replays under `system`, the winner is accepted by its response (pass
        `accept` again when it was not "checks"), and its objective recomputes to the recorded value (pass the objective
        function again when it was one; a fact's name is read from the response). → {"ok", "mismatches": [(what, why)],
        "trace": the trace replay}. Not re-checked: that no other candidate is better — run the search again for that."""
        from .refine import accepted
        bad = []
        if self.response is None:
            ok = self.best is None and (self.escalation is not None or self.accepted == 0)
            return {"ok": ok, "mismatches": [] if ok else [("response", "a best candidate without its response")],
                    "trace": None}
        rep = self.response.trace.replay(system, trust_models=trust_models)
        bad += [("trace", f"{m[1]}: {m[2]}") for m in rep["mismatches"]]
        res = self.response
        if res._system is None:
            res._system = system
        if not (accepted(res, self.question, accept) and res[self.question].status != "abstain"):
            bad.append(("accepted", "the recorded best is not accepted by its response"))
        if self.objective is not None:
            if objective is None and self.objective.startswith("fact "):
                got = res.values.get(self.objective[5:])
            elif objective is not None:
                got = objective(self.best)
            else:
                got = None
                bad.append(("objective", f"pass the objective again ({self.objective}) to recompute it"))
            from .runtime import vhash
            from .storage import plain
            if got is not None and vhash(plain(got)) != vhash(plain(self.value)):
                bad.append(("objective", f"recorded {self.value!r}, recomputed {got!r}"))
        return {"ok": not bad, "mismatches": bad, "trace": rep}


class _Budget(Exception):
    pass


def _depends(catalog, keys):
    """The facts that read any of `keys`, directly or through other facts."""
    out = set(keys)
    parts = list(catalog.parts.values())
    changed = True
    while changed:
        changed = False
        for p in parts:
            ins = set(p.inputs)
            for a in p.alternatives or []:
                ins |= set(a.inputs)
            if p.name not in out and ins & out:
                out.add(p.name)
                changed = True
    return out


def _held_system(system, state, vals, keys):
    """A copy of the System whose parts that do not read the candidate are held at the values computed once from
    `state` (`vals`), its own stats and costs (the searched candidates do not count as the System's asks), no storage."""
    from types import SimpleNamespace

    from .counterfactual import _pinned
    from .learned import CostBook
    from .runtime import MISSING, HashSeed
    dep = _depends(system.catalog, keys)
    cat = copy.copy(system.catalog)
    cat.parts = dict(cat.parts)
    held = []
    for name, p in system.catalog.parts.items():
        if name in dep or name not in vals or vals[name] is MISSING or name in state:
            continue
        cat.parts[name] = _pinned(p, SimpleNamespace(value=vals[name], error=None))
        held.append(name)
    cat._hash_seed = seed = HashSeed()                          # the given values and the held facts are the same
    for v in [*state.values(), *(vals[n] for n in held)]:      # objects in every candidate's ask: hashed once here
        seed.add(v)
    s = copy.copy(system)
    s.catalog, s.storage, s.learn, s.cost_policy = cat, None, False, None
    s.stats = {k: 0 for k in system.stats}
    s.costs = CostBook()
    plans = {}
    plan = s._plan

    def cached(questions, init_keys, why_costs=False):           # the same keys every time: plan once
        k = (tuple(q.name for q in questions), tuple(sorted(init_keys)))
        if k not in plans:
            plans[k] = plan(questions, init_keys, why_costs)
        return plans[k]
    s._plan = cached
    return s, held


def _better(a, b, maximize):
    return a > b if maximize else a < b


def search(system, state, question, space, *, into=None, objective=None, maximize=True, prune=(), keep=1,
           budget=10_000, accept="checks", store=True, hold=True):
    """Candidates from `space` through `system`'s checks for `question` → a SearchRun (see the module docs).

    space: a list / iterable of candidates, a dict {fact: [values]}, a Tree, or a function of the facts computed once
    from `state` returning one of them. into: the given fact a candidate is passed as (required except for a dict).
    objective: a function of the candidate, or a fact name read from its Response (None: the first accepted wins);
    maximize: largest (default) or smallest. prune: hard checks that stay false below a Tree node (a Tree's `bound`
    cuts by the objective). keep: how many accepted candidates to keep. budget: asks at most.
    accept: as in solvi.refine (default "checks"). store: store the winner's full ask when the System has storage.
    hold: compute the facts that do not read the candidate once (default) — False re-runs every part per candidate."""
    from .refine import _governing, accepted, causes, failed_checks
    if question not in system.questions:
        raise ValueError(f"{question!r} is not a question of the system")
    if int(keep) < 1 or int(budget) < 1:
        raise ValueError("keep and budget must be at least 1")
    prune = list(prune or [])
    for c in prune:
        p = system.catalog.parts.get(c)
        if p is None or p.kind != "check":
            raise ValueError(f"prune names {c!r}, which is not a check of the catalog (a typo would never prune)")
    if objective is not None and not (callable(objective) or isinstance(objective, str)):
        raise TypeError("objective is a function of the candidate or the name of a fact")
    t0 = time.perf_counter()
    from .runtime import MISSING
    held = []
    vals = system.facts_for(dict(state)) if hold or (callable(space) and not isinstance(space, Tree)) else None
    if callable(space) and not isinstance(space, Tree):
        space = space({k: v for k, v in vals.items() if v is not MISSING})
    kind = "tree" if isinstance(space, Tree) else "domains" if isinstance(space, dict) else "candidates"
    if kind == "domains":
        if into is not None:
            raise ValueError("a dict of domains gives each candidate as its facts: leave into=None")
        keys = list(space)
    else:
        if not isinstance(into, str):
            raise ValueError("into= names the given fact a candidate is passed as")
        if kind == "candidates" and not hasattr(space, "__iter__"):
            raise TypeError("space is a list, an iterable, a dict of domains, a Tree, or a function of the facts")
        keys = [into]
    clash = [k for k in keys if k in state]
    if clash:
        raise ValueError(f"the state already gives {', '.join(clash)}: a candidate would replace it")
    bound = space.bound if kind == "tree" else None
    if bound is not None and objective is None:
        raise ValueError("a Tree's bound needs an objective")
    if prune and kind != "tree":
        raise ValueError("prune= cuts partial nodes: it needs a Tree")
    asker = system
    if hold:
        asker, held = _held_system(system, state, vals, keys)
    obj_name = None if objective is None else (f"fact {objective}" if isinstance(objective, str)
                                               else getattr(objective, "__name__", "objective"))
    run = SearchRun(question, into, kind, budget=int(budget), objective=obj_name, maximize=maximize, prune=prune,
                    bound=None if bound is None else getattr(bound, "__name__", "bound"), keep=int(keep), held=held)
    kept = []                                                 # [(value, order, candidate)]
    rejected, pruned = {}, {}

    def ask(cand, partial=False):
        if run.asked >= run.budget:
            raise _Budget
        run.asked += 1
        st = dict(state)
        st.update(cand if kind == "domains" else {into: cand})
        return asker.ask(st, names=[question], store=False, early_exit=False if partial or prune else None)

    def judge(cand, res):
        """Accepted? → its objective value (or 0 without one), else None; counts the rejections."""
        r = res[question]
        if r.status != "abstain" and accepted(res, question, accept):
            if objective is None:
                return 0
            if callable(objective):
                return objective(cand)
            if objective not in res.values:
                raise ValueError(f"the objective {objective!r} is not computed for the question {question!r}: list it in "
                                 "the question's checkpoints (or give a function of the candidate)")
            return res.values[objective]
        failed = _governing(res, question, failed_checks(res, question))
        why = (failed[0].check if failed else
               "abstained: " + (causes(res, question) or [r.why])[0][:80] if r.status == "abstain" else f"answer {r.answer!r}")
        rejected[why] = rejected.get(why, 0) + 1
        return None

    def take(cand, v):
        run.accepted += 1
        kept.append((v, run.accepted, cand))
        if objective is not None:
            kept.sort(key=lambda t: ((-t[0] if maximize else t[0]), t[1]))
        del kept[run.keep:]

    def full():                                               # without an objective: stop at `keep` accepted
        return objective is None and len(kept) >= run.keep

    def worst():
        return kept[-1][0] if len(kept) >= run.keep else None

    try:
        if kind == "tree":
            children, complete = space.children, space.complete

            def visit(node):
                kids = None
                if complete is None:
                    kids = list(children(node))
                    is_cand = not kids
                else:
                    is_cand = bool(complete(node))
                if is_cand or prune:
                    res = ask(node, partial=not is_cand)
                    if prune:
                        no = [f.check for f in failed_checks(res) if f.check in prune]
                        if no:
                            pruned[no[0]] = pruned.get(no[0], 0) + 1
                            if is_cand:
                                judge(node, res)
                            return False
                    if is_cand:
                        v = judge(node, res)
                        if v is not None:
                            take(node, v)
                            if full():
                                return True
                if bound is not None and worst() is not None:
                    b = bound(node)
                    if not _better(b, worst(), maximize):
                        pruned["bound"] = pruned.get("bound", 0) + 1
                        return False
                for child in (kids if kids is not None else children(node)):
                    if visit(child):
                        return True
                return False
            visit(space.root)
        else:
            cands = (({k: v for k, v in zip(keys, combo)} for combo in itertools.product(*[list(space[k]) for k in keys]))
                     if kind == "domains" else iter(space))
            for cand in cands:
                v = judge(cand, ask(cand))
                if v is not None:
                    take(cand, v)
                    if full():
                        break
    except _Budget:
        run.exhausted = True
    run.rejected, run.pruned = rejected, pruned
    run.kept = [(c, v if objective is not None else None) for v, _, c in kept]
    if kept:
        best = kept[0][2]
        st = dict(state)
        st.update(best if kind == "domains" else {into: best})
        res = system.ask(st, names=[question], store=store)
        if res[question].status == "abstain" or not accepted(res, question, accept):
            run.escalation = ("the best candidate is not accepted when asked in full with the System — a fact held "
                              "for the search differs from the one the full ask computes; search with hold=False")
        else:
            run.best, run.response = best, res
            run.value = (objective(best) if callable(objective) else res.values.get(objective)) if objective else None
    else:
        run.escalation = ("nothing accepted" + (f" within the budget of {run.budget} asks" if run.exhausted else
                                                " in the whole space")
                          + (": " + ", ".join(f"{k} ({v})" for k, v in sorted(rejected.items(), key=lambda t: -t[1])[:3])
                             if rejected else ""))
    run.seconds = round(time.perf_counter() - t0, 3)
    return run
