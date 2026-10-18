"""Admission by probes over a version space of law sets with up to two new laws (`admit_by_probes(..., max_new=2)`).

The one-law admission assumes exactly one law is missing; its witness guard (a candidate refuted only as too narrow
blocks a lone class while the set with every witness added predicts otherwise) closes most of the wrong admissions where
two are missing, but not those where the witnesses together cover a wrong survivor on the pool. Here every hypothesis
is a law set: the current set plus no new law, one candidate, or two. A hypothesis predicts what its most specific
applied new law says (two new laws of equal length: the earlier candidate under ties="earlier", the later under
"newer"), else what the set predicts now. It is refuted by a row (logged or asked) where it predicts wrong and one of
its new laws was the applied law, or none of them holds, the set is wrong and one of their effects is what was observed
(the too-narrow refutation of the one-law rule, extended to two). The single candidates are told apart first; once
they are decided (one class left, none left, or no probe splits them) the pairs of the surviving candidates and the
witnesses (refuted only as too narrow, never wrong where they applied) that every row seen leaves standing join, and
the probes go on splitting the classes of hypotheses that predict alike on every point of the pool. At the end a
single law is dropped where a surviving pair that contains it explains more of the rows seen, and the smallest
hypothesis of the one surviving class is accepted (fewer laws, then fewer atoms, then the order given) — never while
the set without a new law is alive or in that class. Pairs are formed only from the witnesses and the survivors, so
the classes are fewer than the full pair space where one law is missing (its narrow witnesses are its own
specializations); it still needs more probes than the one-law rule there (see docs/experimental.md, `laws`)."""
from __future__ import annotations

from typing import Any, Callable

from ._core import Admission, Candidate, Transition, Unknown, _arity2, _key

__all__ = ["admit_up_to_two"]


def admit_up_to_two(ls, cands: list[Candidate], ask: Callable[[Any, Any], Any], *, pool: list, safe, budget: int,
                    buffer, exclude, read_only, cost, eq) -> Admission:
    from ._probes import groups_of, split
    E = eq or (lambda u, v: u == v)
    kf = _key if eq is None else (lambda x: x)
    n = len(cands)
    conds = [tuple(c.condition) for c in cands]
    sizes = [len(c) for c in conds]
    acts = {_key(c.action): c.action for c in cands}

    def info(x, a):
        ms = ls.matching(x, a)
        return set(ls.grammar.true(x, a)), ls.predict(x, a, disagree="first"), \
            max((len(law.condition) for law in ms), default=-1)

    def per_cand(x, a, inf):
        """For each candidate at one point: (applied, holds, its effect's prediction there)."""
        T, cur, shadow = inf
        out = []
        for c, cond, s in zip(cands, conds, sizes):
            if _key(c.action) != _key(a):
                out.append((False, False, None))
                continue
            hold = set(cond) <= T
            applied = hold and (s >= shadow if ls.ties == "newer" else s > shadow)
            out.append((applied, hold, ls.apply(c.effect, x)))
        return out

    def first(i, j):
        """Of two new laws both applied at a point, the one that speaks."""
        if sizes[i] != sizes[j]:
            return i if sizes[i] > sizes[j] else j
        lo, hi = min(i, j), max(i, j)
        return hi if ls.ties == "newer" else lo

    def predict(h, pc, cur):
        if not h:
            return cur
        if len(h) == 1:
            ap, _, v = pc[h[0]]
            return v if ap else cur
        i, j = h
        ai, aj = pc[i][0], pc[j][0]
        if ai and aj:
            return pc[first(i, j)][2]
        if ai:
            return pc[i][2]
        if aj:
            return pc[j][2]
        return cur

    def refuted(h, pc, cur, obs):
        p = predict(h, pc, cur)
        if E(p, obs):
            return False
        if not h:
            return True
        if any(pc[m][0] for m in h):
            return True                                  # one of its laws spoke and was wrong
        if any(pc[m][1] for m in h):
            return False                                 # one holds, shadowed by a more specific active law
        return (not E(cur, obs)) and any(E(pc[m][2], obs) for m in h)

    # the hypotheses: single candidates from the start; pairs join when the singles are decided (the null apart)
    hyps: list[tuple] = [(i,) for i in range(n)]
    alive = list(range(n))
    errs = [0] * n                      # rows a surviving hypothesis predicts wrong (another effect's miss)
    narrow: set[int] = set()            # single candidates refuted only as too narrow, never wrong where applied
    seen: list = []                     # (per-candidate info, the set's prediction, the observation) of every row seen
    tried: set = set()                  # pairs already formed
    pair_of: dict = {}
    null = True
    refuted_hyps = 0

    def see(x, a, obs, inf):
        nonlocal alive, null, refuted_hyps
        pc = per_cand(x, a, inf)
        cur = inf[1]
        seen.append((pc, cur, obs))
        keep = []
        for k in alive:
            h = hyps[k]
            if refuted(h, pc, cur, obs):
                if len(h) == 1 and not pc[h[0]][0]:
                    narrow.add(h[0])
                continue
            keep.append(k)
            if not E(predict(h, pc, cur), obs):
                errs[k] += 1
        refuted_hyps += len(alive) - len(keep)
        alive = keep
        for m in [m for m in narrow if pc[m][0] and not E(pc[m][2], obs)]:
            narrow.discard(m)                            # wrong where it applied: no longer a possible law
        if null and not E(cur, obs):
            null = False

    def expand():
        """Pairs of the surviving single candidates and the too-narrow witnesses that every row seen leaves
        standing → how many joined."""
        elig = sorted({hyps[k][0] for k in alive if len(hyps[k]) == 1} | narrow)
        added = 0
        for i in range(len(elig)):
            for j in range(i + 1, len(elig)):
                h = (elig[i], elig[j])
                if h in tried:
                    continue
                tried.add(h)
                if any(refuted(h, pc, cur, obs) for pc, cur, obs in seen):
                    continue
                k = len(hyps)
                hyps.append(h)
                pair_of[h] = k
                errs.append(sum(1 for pc, cur, obs in seen if not E(predict(h, pc, cur), obs)))
                alive.append(k)
                added += 1
        return added

    if buffer is not None:
        for a in acts.values():
            for row in buffer.rows(a, exclude):
                see(row.state, row.action, row.observed, info(row.state, row.action))

    pinf = [info(*p) for p in pool]
    ppc = [per_cand(p[0], p[1], pinf[i]) for i, p in enumerate(pool)]
    vec: dict[int, tuple] = {}

    def vector(k):
        if k not in vec:
            if k < 0:
                vec[k] = tuple(kf(pinf[i][1]) for i in range(len(pool)))
            else:
                h = hyps[k]
                vec[k] = tuple(kf(predict(h, ppc[i], pinf[i][1])) for i in range(len(pool)))
        return vec[k]

    veq = None if eq is None else (lambda u, v: len(u) == len(v) and all(eq(x, y) for x, y in zip(u, v)))

    def undominated():
        """The alive hypotheses less the single laws a surviving pair containing them explains more rows than."""
        live = set(alive)
        out = []
        for k in alive:
            h = hyps[k]
            if len(h) == 1 and errs[k] and any(
                    (q := pair_of.get((min(h[0], j), max(h[0], j)))) is not None and q in live and errs[q] < errs[k]
                    for j in range(n) if j != h[0]):
                continue
            out.append(k)
        return out

    def classes(prune=False):
        ids = (undominated() if prune else alive) + ([-1] if null else [])
        return [[ids[i] for i in g] for g in groups_of([vector(k) for k in ids], veq)]

    def size_key(k):
        h = hyps[k]
        return (len(h), sum(sizes[m] for m in h), h)

    def best_of():
        live = undominated()
        return min(live, key=size_key) if live else None

    def against(b):
        """Phase two: every alive hypothesis (and the null) seen only where the best one `b` speaks — its new laws'
        rows on the pool — and as `b` elsewhere: only there is b's own claim at stake. → classes of these views."""
        hb = hyps[b]
        vb = vector(b)
        region = [any(ppc[i][m][0] for m in hb) for i in range(len(pool))]
        ids = undominated() + ([-1] if null else [])
        views = [tuple(x if r else y for x, y, r in zip(vector(k), vb, region)) for k in ids]
        for k, v in zip(ids, views):
            vec[("view", b, k)] = v
        return [[ids[i] for i in g] for g in groups_of(views, veq)]

    asked: list[Transition] = []
    unknown: list[Transition] = []
    done: set[int] = set()
    ok_safe = (lambda i: True) if safe is None else (lambda i: bool(safe(*pool[i])))
    ro = None if read_only is None else (lambda st, ac: read_only(*pool[st]) if _arity2(read_only)
                                         else read_only(pool[st][1]))
    cs = None if cost is None else (lambda st, ac: cost(*pool[st]))
    why = ""
    phase = 1                           # 1: the single candidates; 2: the best hypothesis against every other
    best = None
    while True:
        if phase == 1:
            gs = classes()
            if len(gs) <= 1 or not alive:
                expand()
                phase = 2
                continue
            reps = [(lambda st, ac, k=g[0]: vector(k)[st]) for g in gs]
        else:
            best = best_of()
            if best is None:
                why = "every hypothesis refuted"
                break
            gs = against(best)
            if len(gs) <= 1:
                why = "one class left"
                break
            reps = [(lambda st, ac, k=g[0], b=best: vec[("view", b, k)][st]) for g in gs]
        if len(asked) + len(unknown) >= int(budget):
            why = f"the budget of {int(budget)} probes is spent"
            break
        pr = split(reps, lambda st, ac: ok_safe(st), [(i, pool[i][1]) for i in range(len(pool)) if i not in done],
                   read_only=ro, cost=cs, same=eq)
        if pr is None:
            if phase == 1:
                expand()
                phase = 2
                continue
            why = "no probe left tells the classes apart (they differ only where it was asked or is unsafe)"
            break
        i = pr.state
        done.add(i)
        x, a = pool[i]
        obs = ask(x, a)
        if obs is Unknown:                               # spent; refutes and supports nothing
            unknown.append(Transition(x, a, obs, group="probe", id=f"unknown-{len(unknown) + 1}-{i}"))
            continue
        asked.append(Transition(x, a, obs, group="probe", id=f"probe-{len(asked) + 1}-{i}"))
        see(x, a, obs, pinf[i])
        if phase == 2:
            expand()                                     # new witnesses may form new pairs
    ls.cost["probes"] += len(asked) + len(unknown)
    if phase == 1:                                       # the budget ran out among the singles
        expand()
        best = best_of()
        gs = against(best) if best is not None else []
    else:
        gs = against(best) if best is not None else []
    shown = [[cands[m] for m in hyps[g[0]]] for g in sorted(gs, key=lambda g: (g[0] < 0, size_key(g[0])
                                                                                 if g[0] >= 0 else ()))
             if g[0] >= 0]                               # each class by one of its hypotheses
    alive_c = sorted({m for k in alive for m in hyps[k]})

    def undetermined(reason):
        return Admission("undetermined", None, shown, asked, refuted_hyps, null, reason, [],
                         [cands[m] for m in range(n) if m not in set(alive_c)], [], laws=[], unknown=unknown)

    if best is None:
        return undetermined("no hypothesis survived" + (f" ({why})" if why else ""))
    if len(gs) > 1:
        return undetermined(f"{len(gs)} classes of hypotheses (up to two new laws) disagree where the smallest "
                            f"surviving one speaks: {why}")
    if null:
        return undetermined(f"the set without a new law survives: {why}")
    h = hyps[best]
    if len(ls.laws) + len(h) > ls.cap:
        ls.capped += 1
        return undetermined("the cap of active laws is reached")
    rows = list(asked)
    if buffer is not None:
        rows = [r for a in acts.values() for r in buffer.rows(a, exclude)] + rows
    laws = []
    for m in h:
        c = cands[m]
        sup = []
        for r in rows:
            inf = info(r.state, r.action)
            pc = per_cand(r.state, r.action, inf)
            if pc[m][0] and (len(h) == 1 or not pc[h[0] if h[1] == m else h[1]][0] or first(*h) == m):
                sup.append(r)
        hits = sum(1 for r in sup if E(ls.apply(c.effect, r.state), r.observed))
        laws.append((c, sup, hits))
    accepted = []
    for c, sup, hits in laws:
        net = sum(1 for r in sup if E(ls.apply(c.effect, r.state), r.observed)
                  and not E(ls.predict(r.state, r.action, disagree="first"), r.observed))
        law = ls._accept(c, born="probe", support=len(sup), hits=hits, net=net, sources=tuple(r.id for r in sup))
        law.admitted, law.tested = "probes", n
        ls.cost["accepted"] += 1
        accepted.append(law)
    return Admission("accepted", accepted[0], shown, asked, refuted_hyps, null,
                     why + (" (two new laws)" if len(accepted) == 2 else ""), [],
                     [cands[m] for m in range(n) if m not in set(alive_c)], [], laws=accepted, unknown=unknown)
