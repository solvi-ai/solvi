"""Admission by probes when an answer can be wrong (`LawSet.admit_by_probes(..., noise=ε)`): the hypotheses keep a
weight instead of being dropped by one answer, the probes are chosen by expected information gain under that noise
(`split(score="eig")`), and a law is accepted only when one class of hypotheses holds at least `level` of the weight."""
from __future__ import annotations

import math
from typing import Any, Callable

from ._probes import eig, groups_of


def admit_noisy(ls, cands, ask: Callable[[Any, Any], Any], *, pool: list, safe, budget: int, buffer, exclude,
                read_only, cost, eq, noise: float, level: float):
    from ._core import Admission, Transition, _key
    from ._probes import _ro

    E = eq or (lambda u, v: u == v)
    e = float(noise)
    ro = _ro(read_only)

    def info(x, a):
        ms = ls.matching(x, a)
        return set(ls.grammar.true(x, a)), ls.predict(x, a, disagree="first"), \
            max((len(law.condition) for law in ms), default=-1)

    cache: dict[int, tuple] = {}

    def pinfo(i):
        if i not in cache:
            cache[i] = info(*pool[i])
        return cache[i]

    def pred(c, x, a, inf):
        """What the set with candidate `c` added predicts (None: the set without a new law)."""
        T, cur, shadow = inf
        if c is None or _key(c.action) != _key(a):
            return cur
        hold = set(c.condition) <= T
        applied = hold and (len(c.condition) >= shadow if ls.ties == "newer" else len(c.condition) > shadow)
        return ls.apply(c.effect, x) if applied else cur

    hyps = [None] + list(cands)                   # 0: the set without a new law
    n_p = len(pool)
    vecs = [tuple(pred(h, pool[i][0], pool[i][1], pinfo(i)) for i in range(n_p)) for h in hyps]
    veq = None if eq is None else (lambda u, v: len(u) == len(v) and all(eq(x, y) for x, y in zip(u, v)))
    cls = groups_of(vecs, veq)                    # classes: equal predictions on every probe of the pool
    cls.sort(key=lambda g: (min(len(hyps[j].condition) if hyps[j] is not None else -1 for j in g), g[0]))
    of = {j: ci for ci, g in enumerate(cls) for j in g}
    loglik = [0.0] * len(hyps)                    # per hypothesis; a class's weight: its members' mean likelihood
    asked: list = []

    def observe(x, a, obs, inf, G):
        """Every hypothesis's likelihood of this answer: 1 − ε where it predicts it, else ε / (G − 1)."""
        lo_ok, lo_bad = math.log(1 - e), math.log(e / max(1, G - 1))
        for j, h in enumerate(hyps):
            loglik[j] += lo_ok if E(pred(h, x, a, inf), obs) else lo_bad

    if buffer is not None:
        acts = {_key(c.action): c.action for c in cands}
        for a in acts.values():
            for row in buffer.rows(a, exclude):
                inf = info(row.state, row.action)
                G = len(groups_of([pred(h, row.state, row.action, inf) for h in hyps], eq))
                observe(row.state, row.action, row.observed, inf, max(G, 2))

    def weights():
        m = max(loglik)
        ws = []
        for g in cls:
            ws.append(sum(math.exp(loglik[j] - m) for j in g) / len(g))
        t = sum(ws)
        return [x / t for x in ws]

    safe_ok = [safe is None or bool(safe(*pool[i])) for i in range(n_p)]
    reps = [g[0] for g in cls]
    pv = [[vecs[r][i] for r in reps] for i in range(n_p)]      # per pool point: each class's prediction
    pg = [groups_of(pv[i], eq) for i in range(n_p)]
    why = ""
    while True:
        w = weights()
        top = max(range(len(cls)), key=lambda ci: (w[ci], -ci))
        if w[top] >= level:
            why = f"one class holds {w[top]:.3f} of the weight"
            break
        if len(asked) >= budget:
            why = f"the budget of {budget} answers is spent (best class {w[top]:.3f} < {level})"
            break
        best = None
        for i in range(n_p):
            if not safe_ok[i] or len(pg[i]) < 2:
                continue
            g_w = [sum(w[ci] for ci in g) for g in pg[i]]
            gain = eig(g_w, e)
            if gain <= 0:
                continue
            c = cost(*pool[i]) if cost else 0
            key = (0 if ro is None or ro(*pool[i]) else 1, -gain, c, i)
            if best is None or key < best[0]:
                best = (key, i)
        if best is None:
            why = "no safe probe tells the classes apart"
            break
        i = best[1]
        x, a = pool[i]
        obs = ask(x, a)
        asked.append(Transition(x, a, obs, group="probe", id=f"probe-{len(asked)}"))
        ls.cost["probes"] += 1
        lo_ok, lo_bad = math.log(1 - e), math.log(e / (len(pg[i]) - 1))
        for j in range(len(hyps)):
            loglik[j] += lo_ok if E(vecs[j][i], obs) else lo_bad

    w = weights()
    classes = [[hyps[j] for j in g if hyps[j] is not None] for g in cls]
    top = max(range(len(cls)), key=lambda ci: (w[ci], -ci))
    refuted = sum(len([j for j in cls[ci] if hyps[j] is not None]) for ci in range(len(cls)) if w[ci] < 1e-6)

    def undetermined(reason):
        return Admission("undetermined", None, classes, asked, refuted, 0 in cls[top], reason)

    if w[top] < level:
        return undetermined(why)
    if 0 in cls[top]:
        return undetermined(f"the set without a new law is the likely class: {why}")
    if len(ls.laws) >= ls.cap:
        ls.capped += 1
        return undetermined("the cap of active laws is reached")
    j = min((j for j in cls[top] if hyps[j] is not None), key=lambda j: (len(hyps[j].condition), j))
    c = hyps[j]
    rows = list(asked)
    if buffer is not None:
        rows = [r for r in buffer.rows(c.action, exclude)] + rows
    sup = [r for r in rows if _key(r.action) == _key(c.action)
           and set(c.condition) <= set(ls.grammar.true(r.state, r.action))]
    hits = sum(1 for r in sup if E(ls.apply(c.effect, r.state), r.observed))
    net = sum(1 for r in sup if E(ls.apply(c.effect, r.state), r.observed)
              and not E(ls.predict(r.state, r.action, disagree="first"), r.observed))
    law = ls._accept(c, born="probe", support=len(sup), hits=hits, net=net, sources=tuple(r.id for r in sup))
    law.admitted, law.tested = "probes", len(cands)
    ls.cost["accepted"] += 1
    return Admission("accepted", law, classes, asked, refuted, False, why, [], [], [], laws=[law])


__all__ = ["admit_noisy"]
