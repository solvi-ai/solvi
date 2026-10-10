"""Ranking candidate laws where each check is expensive (a live probe, a planner call): RankedProposer reorders another
proposer's candidates by a scorer over features read off the Buffer's bitsets; the candidates themselves never change.

The scorer: BagOfAtomsRanker, a numpy logistic regression over per-atom features (score(condition) = Σ over its atoms
of the atom's log-odds), trained on synthetic tasks drawn from the grammar itself. The mandatory control: FeatureRanker,
the raw statistic with no training. Blind order: by length, then by name (Enumerate's order)."""
import random
import warnings
from collections import Counter

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, BagOfAtomsRanker, Buffer, Candidate, Enumerate, FeatureRanker, Grammar,
                                         LawSet, RankedProposer, Transition, atom_features, candidate_features,
                                         split, synthetic_task)

CATS = [f"c{i}" for i in range(7)]
COUNTERS = ["n0", "n1", "n2"]


def grammar():
    atoms = [Atom(f"{c}={v}", (lambda s, c=c, v=v: s[c] == v), feature=c) for c in CATS for v in "abc"]
    atoms += [Atom(f"{n}>={t}", (lambda s, n=n, t=t: s[n] >= t), feature=n) for n in COUNTERS for t in (1, 2, 3)]
    return Grammar(atoms, k=3)


def sample(rng: random.Random):
    s = {c: rng.choice("abc") for c in CATS}
    s.update({n: rng.randint(0, 4) for n in COUNTERS})
    return s


G = grammar()


def checks(order, cond):
    """Checks until the true condition: its 1-based position in the order (one check per distinct condition)."""
    seen = []
    for c in order:
        if c.condition not in seen:
            seen.append(c.condition)
        if c.condition == cond:
            return len(seen)
    return None


def tasks(seed, n=40):
    rng = random.Random(seed)
    return [synthetic_task(G, sample, rng, rows=300, noise=0.02) for _ in range(n)]


@pytest.fixture(scope="module")
def trained():
    return BagOfAtomsRanker(seed=0).fit(G, sample, tasks=150, rows=300, noise=0.02)


def test_a_synthetic_task_is_a_buffer_a_mismatch_and_its_true_condition():
    t = synthetic_task(G, sample, random.Random(5), rows=200)
    assert t.buffer.n == 200 and 1 <= len(t.condition) <= G.k and G.check(t.condition) == t.condition
    assert set(t.condition) <= set(G.true(t.state, t.action))           # the mismatch state satisfies the condition
    cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    assert any(c.condition == t.condition for c in cands)


def test_features_are_fixed_length_and_read_off_the_buffer():
    t = synthetic_task(G, sample, random.Random(6), rows=200)
    names, X = atom_features(t.lawset, t.buffer, t.action, t.observed)
    assert X.shape == (len(G.atoms), len(X[0])) and names == [a.name for a in G.atoms]
    lengths = {candidate_features(Candidate(t.action, c, t.observed), t.buffer, t.lawset).shape
               for c in [(), t.condition, G.conditions(G.true(t.state, t.action))[-1]]}
    assert len(lengths) == 1
    # an atom of the true condition covers (nearly) every row with the effect: its recall is high
    i = names.index(t.condition[0])
    assert X[i, 1] >= 0.9


def test_the_ranker_only_reorders_never_adds_or_drops(trained):
    t = tasks(11, 1)[0]
    inner = Enumerate()
    base = inner.propose(t.lawset, t.state, t.action, t.observed)
    ranked = RankedProposer(inner, trained, t.buffer).propose(t.lawset, t.state, t.action, t.observed)
    assert Counter(base) == Counter(ranked) and len(base) == len(ranked)
    raw = RankedProposer(inner, FeatureRanker(), t.buffer).propose(t.lawset, t.state, t.action, t.observed)
    assert Counter(base) == Counter(raw)


def test_training_and_ranking_are_deterministic():
    a = BagOfAtomsRanker(seed=3).fit(G, sample, tasks=30, rows=200)
    b = BagOfAtomsRanker(seed=3).fit(G, sample, tasks=30, rows=200)
    assert np.array_equal(a.w, b.w) and a.b == b.b
    t = tasks(12, 1)[0]
    o1 = RankedProposer(Enumerate(), a, t.buffer).propose(t.lawset, t.state, t.action, t.observed)
    o2 = RankedProposer(Enumerate(), b, t.buffer).propose(t.lawset, t.state, t.action, t.observed)
    assert [c.condition for c in o1] == [c.condition for c in o2]


def test_the_learned_ranker_needs_a_fifth_of_the_blind_checks_and_the_raw_control_is_reported(trained, capsys):
    blind, learned = [], []
    raw = {stat: [] for stat in FeatureRanker.STATS}
    for t in tasks(1234, 60):
        cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)          # blind: length, then name
        blind.append(checks(cands, t.condition))
        learned.append(checks(RankedProposer(Enumerate(), trained, t.buffer).propose(
            t.lawset, t.state, t.action, t.observed), t.condition))
        for stat in raw:
            raw[stat].append(checks(RankedProposer(Enumerate(), FeatureRanker(stat), t.buffer).propose(
                t.lawset, t.state, t.action, t.observed), t.condition))
    mb, ml = float(np.mean(blind)), float(np.mean(learned))
    with capsys.disabled():
        print(f"\n[rank] checks to the true condition, mean over 60 tasks: blind {mb:.1f}, learned {ml:.2f}, "
              + ", ".join(f"raw {s} {np.mean(v):.2f}" for s, v in raw.items()))
    assert None not in blind + learned
    assert ml <= mb / 5


def test_the_ranked_order_feeds_split_under_a_probe_budget(trained):
    """For a live system: the first m ranked candidates become hypotheses, split picks the probe that tells them
    apart; the truth is among the first m far more often than in the blind order."""
    hit_ranked = hit_blind = 0
    m = 5
    for t in tasks(77, 20):
        cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
        ranked = RankedProposer(Enumerate(), trained, t.buffer).propose(t.lawset, t.state, t.action, t.observed)
        hit_blind += any(c.condition == t.condition for c in cands[:m])
        hit_ranked += any(c.condition == t.condition for c in ranked[:m])
        hyps = [t.lawset.hypothesis(c) for c in ranked[:m]]
        probes = [(row.state, row.action) for row in t.buffer.rows()[:50]]
        p = split(hyps, lambda s, a: True, probes)
        assert p is None or p.score >= 1
    assert hit_ranked >= 2 * max(1, hit_blind)


def test_the_proposer_works_inside_the_law_loop(trained):
    t = tasks(5, 1)[0]
    laws = LawSet(G, base=lambda s, a: "off", proposer=RankedProposer(Enumerate(), trained, t.buffer))
    held = Buffer(G)
    held.extend(Transition(r.state, r.action, r.observed, group=f"h{i % 4}") for i, r in enumerate(t.buffer.rows()))
    st = laws.step(Transition(t.state, t.action, t.observed, group="new"), held)
    assert st.mismatch and st.candidates > 0
