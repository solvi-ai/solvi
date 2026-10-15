"""One count of what a proposer offered and what the checks did with it, named the same in laws, refine and compile:
proposed = dropped (refused before any check could judge it) + refuted (a check judged it and said no) + admitted."""
from __future__ import annotations

import random
import warnings

from solvi.core.slow.proposals import Proposals, proposals_of
from solvi.core.slow.refine import refine
from test_compile import GOOD, INPUTS, OFF_BY_ONE, POLICY, QS, TEST_OK, Writer
from test_refine import FakeServer, Slot, _system
from test_refine import proposals as scripted

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.compile import Spec, compile_spec
    from solvi.experimental.laws import Atom, Buffer, CallableProposer, Grammar, LawSet, Transition


def _sums(p):
    assert isinstance(p, Proposals) and p.proposed == p.dropped + p.refuted + p.admitted
    assert set(p.to_dict()) == {"proposed", "dropped", "refuted", "admitted"}


def test_refine_counts_its_rounds():
    s, _ = _system()
    p = scripted(Slot(day="Sun", start=13), Slot(day="Mon", start=13), Slot(day="Mon", start=15))
    run = refine(s, {}, "ok", propose=p, into="proposal", rounds=4)
    assert run.proposals == Proposals(3, 0, 2, 1) == proposals_of(run)
    _sums(run.proposals)


def test_a_rejected_reply_is_dropped_in_refine():
    from solvi.core.slow.generate import generator
    srv = FakeServer('{"day": "Sun", "start": 9}', "sorry, no JSON", '{"day": "Mon", "start": 9}')
    g = generator("http://127.0.0.1:9/v1", "m", opener=srv)
    s, _ = _system()
    run = refine(s, {}, "ok", propose=g.proposer("Propose a slot as JSON.", schema=Slot), into="proposal", rounds=3)
    assert run.proposals == Proposals(3, 1, 1, 1)


def test_compile_counts_drafts_per_round():
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [OFF_BY_ONE, GOOD]], tests=[TEST_OK[1]]))
    assert c.accepted
    assert c.proposals == Proposals(4, 0, 2, 2) == proposals_of(c)   # round 1: both drafts sent back; round 2: both pass
    bad = compile_spec(Spec(POLICY), QS, INPUTS, Writer([["def f(x):\n    return 1\n"], [GOOD]]), rounds=1)
    assert not bad.accepted and bad.proposals.dropped == 1 and bad.proposals.proposed == 2
    _sums(bad.proposals)


def test_laws_count_grammar_drops_tests_and_acceptances():
    rng = random.Random(0)
    g = Grammar([Atom(f"f{i}", (lambda s, i=i: s[i]), feature=f"f{i}") for i in range(4)], k=1)
    stream = []
    for r in range(300):
        s = tuple(rng.random() < 0.5 for _ in range(4))
        stream.append(Transition(s, "do", "E" if s[0] else "N", group=r % 5, id=f"r{r}"))
    b = Buffer(g)
    b.extend(stream)
    prop = CallableProposer(lambda st, a, obs, true, gr: [["f0"], ["f1"], ["nope"]])
    ls = LawSet(g, base=lambda s, a: "N", proposer=prop)
    ls.learn(stream, b)
    p = proposals_of(ls)
    _sums(p)
    assert p.dropped == prop.dropped > 0 and p.admitted == ls.cost["accepted"] >= 1
    assert p.refuted == ls.cost["candidates"] - ls.cost["accepted"]
