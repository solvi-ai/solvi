"""Opt-in empirical compaction is separate from positive-net ordinary admission."""
import pickle
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def leap(s):
    return s[0] == "leap"


def on(s):
    return s[1] == "on"


def off(s):
    return s[1] == "off"


def world(*, support_on="matched", **options):
    g = Grammar([Atom("leap", leap), Atom("on", on, feature="beat"),
                 Atom("off", off, feature="beat")], k=2)
    b = Buffer(g)
    for i in range(60):
        b.add(Transition(("leap", "on" if i % 2 else "off"), "do", "yes", group=i, id=f"r{i}"))
    ls = LawSet(g, support_on=support_on, **options)
    old = [ls.add(Candidate("do", ("leap", beat), "yes")) for beat in ("on", "off")]
    return ls, b, old, Candidate("do", ("leap",), "yes")


@pytest.mark.parametrize("support_on", ["matched", "applied"])
def test_original_witness_compacts_without_changing_ordinary_net_gate(support_on):
    ls, b, old, c = world(support_on=support_on)
    assert ls.score(c, b) == ((60, 60, 0) if support_on == "matched" else (0, 0, 0))
    assert ls.test([c], b) == []
    assert ls.predict(("leap", "mid"), "do") is None
    law = ls.consolidate(c, b)
    assert law and law.support == law.hits == 60 and law.level is not None
    assert law.consolidation["original_net"] == 0
    assert law.consolidation["counterfactual_net"] == 60
    assert law.consolidation["replaced"] == tuple(x.id for x in old)
    assert law.consolidation["domain_guarantee"] is False
    assert ls.predict(("leap", "mid"), "do") == "yes"
    assert all(x in ls.retired and x.why == f"subsumed by {law.id}" for x in old)
    assert ls.support_on == support_on and ls.laws == [law]


def test_contrary_overlap_and_default_require_full_matched_support():
    ls, b, old, c = world(support_on="applied")
    for i in range(60):
        b.add(Transition(("leap", "mid"), "do", "no", group=100+i, id=f"bad{i}"))
    exception = ls.add(Candidate("do", ("leap",), "no"))
    # An equal-condition opposite effect must not hide the contrary matched rows.
    assert ls.consolidate(c, b) is None
    assert ls.laws == old + [exception]
    for i in range(120):
        b.add(Transition(("step", "mid"), "do", "no", group=200+i, id=f"default{i}"))
    assert ls.consolidate(Candidate("do", (), "yes"), b) is None
    assert all(x.status == "active" for x in old)


def test_dryrun_replay_pickle_and_independent_observation():
    ls, b, old, c = world()
    before = (b.tested, ls.version, dict(ls.cost), tuple(ls.laws), tuple(ls.retired))
    first = ls.consolidate(c, b, commit=False)
    second = ls.consolidate(c, b, commit=False)
    assert first.status == second.status == "proposed"
    assert first.consolidation == second.consolidation
    assert (b.tested, ls.version, dict(ls.cost), tuple(ls.laws), tuple(ls.retired)) == before
    law = ls.consolidate(c, b)
    ls2, b2 = pickle.loads(pickle.dumps((ls, b)))
    assert ls2.laws[0].consolidation == law.consolidation
    assert ls2.predict(("leap", "mid"), "do") == "yes"
    assert ls2.consolidate(c, b2) is None
    for i in range(5):
        ls2.observe(Transition(("leap", "mid"), "do", "no", group=100+i, id=f"independent{i}"))
    # Observe alone counts independent contradictions; check recomputes held evidence.
    assert ls2.laws[0].hits_after == 0 and ls2.laws[0].n == 5
    for i in range(200):
        ls2.observe(Transition(("leap", "mid"), "do", "no", group=300+i, id=f"refute{i}"))
    assert ls2.laws == [] and ls2.retired[-1].id == law.id


@pytest.mark.parametrize("raises", [False, True])
def test_confirmation_rejection_is_transactional(raises):
    def confirm(c):
        if raises:
            raise RuntimeError("validator unavailable")
        return False
    ls, b, old, c = world(confirm=confirm)
    assert ls.consolidate(c, b, commit=False).status == "proposed"
    if raises:
        with pytest.raises(RuntimeError, match="validator unavailable"):
            ls.consolidate(c, b)
    else:
        assert ls.consolidate(c, b) is None
    assert ls.laws == old and ls.retired == []
    assert all(x.status == "active" for x in old)


def test_sized_support_exclude_falsified_and_readmit_are_not_bypassed():
    ls, b, old, c = world(min_support=61)
    assert ls.consolidate(c, b) is None
    ls.min_support = 5
    ls._falsified_at[ls._ck(c)] = ls._clock(b)
    assert ls.consolidate(c, b) is None
    del ls._falsified_at[ls._ck(c)]
    ls.readmit = "grown"
    ls._retired_at[ls._ck(c)] = ls._clock(b)
    assert ls.consolidate(c, b) is None
    ls._retired_at.clear()
    law = ls.consolidate(c, b, exclude=0)
    assert law.support == 59 and "r0" not in law.sources


def test_counterfactual_net_and_sized_gate_still_apply():
    ls, b, old, c = world(base=lambda s, a: "yes")
    assert ls.consolidate(c, b) is None  # deleting the patchwork adds no errors either
    ls, b, old, c = world(delta=1e-30)
    assert ls.consolidate(c, b) is None
    assert ls.laws == old


def test_cap_and_existing_exception_are_preserved():
    ls, b, old, c = world(cap=3)
    exception = ls.add(Candidate("do", ("on",), "no"))
    # Equal-specificity priorities change after replacement; losing correct rows blocks commit.
    assert ls.consolidate(c, b) is None
    assert ls.laws == old + [exception]
    assert all(x.status == "active" for x in old)
    ls, b, old, c = world(cap=2)
    law = ls.consolidate(c, b)
    assert law and len(ls.laws) == 1


def test_indistinguishable_rival_still_blocks_consolidation():
    ls, b, old, c = world(rivals=True)
    ls.grammar.add(Atom("pulse", leap))
    rival = ls.add(Candidate("do", ("pulse",), "yes"))
    assert ls.consolidate(c, b) is None
    assert set(ls.laws) == set(old + [rival])
