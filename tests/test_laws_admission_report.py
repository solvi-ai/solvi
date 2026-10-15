"""solvi.experimental.laws: what the probe admission reports — the refuted candidates themselves, a law's ground next to
its prediction (`explain`), and a pool that never varies an atom the shortest candidate drops (`coverage`)."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Enumerate, Grammar, LawSet, Undetermined


def craft(table_always: bool, seed: int = 0):
    """make → 1 only next to the table with wood; a pool that is next to the table everywhere (or not)."""
    g = Grammar([Atom("near_table", lambda s: s["t"]), Atom("wood1", lambda s: s["w"] >= 1),
                 Atom("stone1", lambda s: s["st"] >= 1)], k=3)
    ls = LawSet(g, base=lambda s, a: 0)
    rng = random.Random(seed)
    pool = [({"t": True if table_always else rng.random() < 0.5, "w": rng.randint(0, 2), "st": rng.randint(0, 2)},
             "make") for _ in range(30)]
    cands = Enumerate().propose(ls, {"t": True, "w": 1, "st": 0}, "make", 1)
    return ls, cands, pool


def world(s, a):
    return 1 if s["t"] and s["w"] >= 1 else 0


def test_refuted_is_a_count_and_refuted_list_holds_the_candidates():
    ls, cands, pool = craft(table_always=False)
    adm = ls.admit_by_probes(cands, world, probes=pool, budget=8)
    assert isinstance(adm.refuted, int)
    assert len(adm.refuted_list) == adm.refuted > 0
    assert all(c in cands for c in adm.refuted_list)


def test_explain_tells_a_law_saying_zero_from_no_law():
    ls, cands, pool = craft(table_always=False)
    adm = ls.admit_by_probes(cands, world, probes=pool, budget=8)
    assert adm.status == "accepted" and set(adm.law.condition) == {"near_table", "wood1"}
    assert ls.explain({"t": True, "w": 1, "st": 0}, "make") == (1, adm.law)
    assert ls.explain({"t": False, "w": 0, "st": 0}, "make") == (0, None)          # the base says 0, no law
    assert ls.predict({"t": False, "w": 0, "st": 0}, "make") == 0
    ls2 = LawSet(ls.grammar)
    assert ls2.explain({"t": True, "w": 1, "st": 0}, "make") == (None, None)


def test_explain_abstains_where_the_laws_disagree():
    from solvi.experimental.laws import Candidate
    ls, _, _ = craft(table_always=False)
    a = ls.add(Candidate("make", ("near_table",), 1))
    b = ls.add(Candidate("make", ("wood1",), 2))
    v, law = ls.explain({"t": True, "w": 1, "st": 0}, "make", disagree="abstain")
    assert isinstance(v, Undetermined) and law is None
    assert ls.explain({"t": True, "w": 1, "st": 0}, "make") == (1, a) and b in ls.laws


def test_a_pool_that_never_varies_an_atom_warns_and_reports_it():
    ls, cands, pool = craft(table_always=True)
    with pytest.warns(UserWarning, match="near_table"):
        adm = ls.admit_by_probes(cands, world, probes=pool, budget=8)
    assert adm.status == "accepted" and adm.law.condition == ("wood1",)          # as before: accepted, now flagged
    assert adm.untested == ["near_table"]


def test_coverage_undetermined_refuses_the_too_general_law():
    ls, cands, pool = craft(table_always=True)
    adm = ls.admit_by_probes(cands, world, probes=pool, budget=8, coverage="undetermined")
    assert adm.status == "undetermined" and adm.law is None and adm.untested == ["near_table"]
    assert not ls.laws


def test_coverage_ignore_and_a_varied_pool_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ls, cands, pool = craft(table_always=True)
        adm = ls.admit_by_probes(cands, world, probes=pool, budget=8, coverage="ignore")
        assert adm.status == "accepted" and adm.untested == []
        ls, cands, pool = craft(table_always=False)
        adm = ls.admit_by_probes(cands, world, probes=pool, budget=8)
        assert adm.status == "accepted" and adm.untested == []
    with pytest.raises(ValueError):
        ls.admit_by_probes(cands, world, probes=pool, coverage="maybe")
