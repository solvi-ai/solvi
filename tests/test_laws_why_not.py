"""LawSet.why_not(candidate, buffer): one call naming the test a candidate failed and by how many rows."""
from __future__ import annotations

import random
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition, sized_need


def _world(rows, seed=0):
    rng = random.Random(seed)
    g = Grammar([Atom(f"f{i}", (lambda s, i=i: s[i]), feature=f"f{i}") for i in range(6)], k=2)
    b = Buffer(g)
    for r in range(rows):
        s = tuple(rng.random() < 0.5 for _ in range(6))
        b.add(Transition(s, "do", "E" if s[0] else "N", group=r % 5, id=f"r{r}"))
    return g, b


def test_a_true_law_on_enough_rows_passes_and_matches_the_test():
    g, b = _world(400)
    ls = LawSet(g, base=lambda s, a: "N")
    c = Candidate("do", ("f0",), "E")
    w = ls.why_not(c, b)
    assert w["passes"] and w["failed"] is None
    assert ls.test([c], b) and ls.laws[0].condition == ("f0",)


def test_too_few_rows_names_the_sized_need_and_the_rows_short():
    g, b = _world(12)
    ls = LawSet(g, base=lambda s, a: "N")
    w = ls.why_not(Candidate("do", ("f0",), "E"), b)
    assert not w["passes"] and w["failed"] in ("min_support", "sized_need")
    assert w["short"] == w["need"] - w["support"] > 0
    assert w["tested"] == b.tested + 1                     # the N `test` would score it at: this candidate counted
    assert w["need"] == max(ls.min_support, sized_need(w["tested"], ls.delta, w["null"]))
    assert "rows" in w["reason"]
    assert not ls.test([Candidate("do", ("f0",), "E")], b)


def test_a_wrong_effect_fails_on_net_or_precision():
    g, b = _world(400)
    ls = LawSet(g, base=lambda s, a: "N")
    w = ls.why_not(Candidate("do", ("f1",), "E"), b)
    assert not w["passes"] and w["failed"] in ("net", "min_precision")
    assert w["support"] > 0 and w["hits"] < w["support"]


def test_an_active_law_is_named_and_the_buffer_is_not_touched():
    g, b = _world(400)
    ls = LawSet(g, base=lambda s, a: "N")
    c = Candidate("do", ("f0",), "E")
    ls.test([c], b)
    before = (b.tested, dict(ls.cost))
    w = ls.why_not(c, b)
    assert w["failed"] == "active" and not w["passes"]
    assert (b.tested, dict(ls.cost)) == before


def test_why_not_agrees_with_test_on_every_small_candidate():
    import itertools
    agree = 0
    for rows in (15, 40, 200):
        for cond in [()] + [(f"f{i}",) for i in range(6)] + [tuple(p) for p in itertools.combinations(
                [f"f{i}" for i in range(4)], 2)]:
            for eff in ("E", "N"):
                g, b = _world(rows, seed=rows)
                ls = LawSet(g, base=lambda s, a: "N")
                c = Candidate("do", tuple(sorted(cond)), eff)
                w = ls.why_not(c, b)
                assert w["passes"] == bool(ls.test([c], b)), (rows, cond, eff, w)
                agree += 1
    assert agree == 3 * 13 * 2
