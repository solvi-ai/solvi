"""JointRanker: candidate conditions ranked by the joint statistic of the whole condition over the logged rows — the
Jaccard between the rows the condition matches and the rows where the effect predicts right — with no training.

Positive control: on a planted grammar with deep conditions (3–4 atoms of k = 4, 5% label noise) it needs fewer checks
to the true condition than the raw per-atom control (FeatureRanker) and the trained bag of atoms (BagOfAtomsRanker).
Negative control: the same scorer against an effect mask whose rows are shuffled is worse than the blind order (the
gain comes from the rows, not from the scorer's shape). Invariant: only the order changes. With no rows: the blind
order."""
import importlib.util
import random
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, BagOfAtomsRanker, Buffer, Candidate, Enumerate, FeatureRanker, Grammar,
                                         JointRanker, LawSet, RankedProposer, Transition, checks_to, synthetic_task)


def grammar(ncat, k):
    cats = [f"c{i}" for i in range(ncat)]
    atoms = [Atom(f"{c}={v}", (lambda s, c=c, v=v: s[c] == v), feature=c) for c in cats for v in "abc"]
    atoms += [Atom(f"n{i}>={t}", (lambda s, i=i, t=t: s[f"n{i}"] >= t), feature=f"n{i}") for i in range(3)
              for t in (1, 2, 3)]
    return Grammar(atoms, k=k), cats


def sampler(cats):
    def sample(rng):
        s = {c: rng.choice("abc") for c in cats}
        s.update({f"n{i}": rng.randint(0, 4) for i in range(3)})
        return s
    return sample


DEEP, DEEP_CATS = grammar(8, 4)
DEEP_SAMPLE = sampler(DEEP_CATS)
SMALL, SMALL_CATS = grammar(7, 3)
SMALL_SAMPLE = sampler(SMALL_CATS)


class ShuffledMask(JointRanker):
    """The negative control: the effect's rows shuffled among the allowed rows (same count; seeded by the seed and the
    mask, so each task gets its own permutation)."""

    def __init__(self, seed=0):
        self.seed = seed

    def effect_mask(self, lawset, buffer, action, effect, exclude=None):
        E = super().effect_mask(lawset, buffer, action, effect, exclude)
        allowed = buffer.allowed(action, exclude)
        idx = [i for i in range(allowed.bit_length()) if allowed >> i & 1]
        bits = [E >> i & 1 for i in idx]
        random.Random(self.seed * 1_000_003 + E).shuffle(bits)    # a permutation per task, seeded
        return sum(b << i for b, i in zip(bits, idx))


def deep_tasks(seed, n):
    rng = random.Random(seed)
    return [synthetic_task(DEEP, DEEP_SAMPLE, rng, rows=300, noise=0.05, size=rng.randint(3, 4)) for _ in range(n)]


def order(t, scorer):
    return RankedProposer(Enumerate(), scorer, t.buffer).propose(t.lawset, t.state, t.action, t.observed)


@pytest.fixture(scope="module")
def deep_bag():
    return BagOfAtomsRanker(seed=0).fit(DEEP, DEEP_SAMPLE, tasks=100, rows=300, noise=0.05)


class _Fixed:
    def __init__(self, cands):
        self.cands = cands

    def propose(self, lawset, state, action, observed):
        return list(self.cands)


def test_the_score_is_the_jaccard_of_the_whole_condition_and_ties_go_to_fewer_atoms():
    g = Grammar([Atom("a", lambda s: s["a"], feature="a"), Atom("b", lambda s: s["b"], feature="b")], k=2)
    buf = Buffer(g)
    # the effect "on" exactly where a and b both hold; a alone and b alone each cover half of their rows
    for i, (a, b) in enumerate([(1, 1), (1, 1), (1, 0), (0, 1), (0, 0), (1, 1)]):
        buf.add(Transition({"a": a, "b": b}, "act", "on" if a and b else "off", group=f"g{i % 2}"))
    laws = LawSet(g, base=lambda s, a: "off")
    cands = [Candidate("act", c, "on") for c in [("a",), ("b",), ("a", "b")]]
    r = JointRanker()
    assert r.jaccard(cands, laws, buf) == [0.75, 0.75, 1.0]
    s = r.score(cands, laws, buf)
    assert s[2] > s[0] == s[1]
    # equal Jaccard, different length: the shorter first (a and (a, b) both match exactly the rows where b holds
    # among the effect "x" rows below)
    g2 = Grammar([Atom("a", lambda s: s["a"], feature="a"), Atom("b", lambda s: s["b"], feature="b")], k=2)
    b2 = Buffer(g2)
    for i, (a, b) in enumerate([(1, 1), (1, 1), (0, 0), (0, 0)]):
        b2.add(Transition({"a": a, "b": b}, "act", "on" if a else "off", group=f"g{i}"))
    l2 = LawSet(g2, base=lambda s, a: "off")
    two = [Candidate("act", ("a", "b"), "on"), Candidate("act", ("a",), "on")]
    assert r.jaccard(two, l2, b2) == [1.0, 1.0]
    s2 = r.score(two, l2, b2)
    assert s2[1] > s2[0]
    assert [c.condition for c in RankedProposer(_Fixed(two), r, b2).propose(l2, None, "act", "on")] == [("a",), ("a", "b")]
    tie = [Candidate("act", ("a", "b"), "on"), Candidate("act", ("a", "b"), "on")]
    assert len(set(r.score(tie, laws, buf))) == 1
    # exclude: the rows of one group are left out (left: rows 1, 3, 5 — a on 1, 5; b on 1, 3, 5; the effect on 1, 5)
    assert r.jaccard(cands, laws, buf, exclude="g0") == [1.0, 2 / 3, 1.0]


def test_with_no_rows_the_order_is_the_blind_one():
    t = deep_tasks(3, 1)[0]
    empty = Buffer(DEEP)
    blind = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    ranked = RankedProposer(Enumerate(), JointRanker(), empty).propose(t.lawset, t.state, t.action, t.observed)
    assert [c.condition for c in ranked] == [c.condition for c in blind]


def test_the_joint_ranker_only_reorders_and_is_deterministic():
    t = deep_tasks(11, 1)[0]
    base = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    for scorer in (JointRanker(), ShuffledMask()):
        ranked = order(t, scorer)
        assert Counter(base) == Counter(ranked) and len(base) == len(ranked)
    assert [c.condition for c in order(t, JointRanker())] == [c.condition for c in order(t, JointRanker())]
    assert JointRanker().score([], t.lawset, t.buffer) == []


def test_on_deep_conditions_the_joint_statistic_beats_the_per_atom_scorers_and_the_shuffled_control_loses(deep_bag,
                                                                                                         capsys):
    arms = {"blind": None, "raw": FeatureRanker(), "bag": deep_bag, "joint": JointRanker(), "shuffled": ShuffledMask()}
    got = {a: [] for a in arms}
    for t in deep_tasks(2024, 20):
        for a, scorer in arms.items():
            o = Enumerate().propose(t.lawset, t.state, t.action, t.observed) if scorer is None else order(t, scorer)
            got[a].append(checks_to(o, t.condition))
    assert all(None not in v for v in got.values())
    mean = {a: float(np.mean(v)) for a, v in got.items()}
    with capsys.disabled():
        print("\n[rank joint] checks to the true condition, deep conditions, mean over 20 tasks: "
              + ", ".join(f"{a} {m:.2f}" for a, m in mean.items()))
    assert mean["joint"] <= mean["blind"] / 5
    assert mean["joint"] <= mean["raw"] / 1.5
    assert mean["joint"] <= mean["bag"]
    assert mean["shuffled"] >= mean["blind"] / 2 and mean["shuffled"] >= 20 * mean["joint"]


def test_on_the_small_grammar_the_joint_statistic_reaches_the_condition_first():
    rng = random.Random(1234)
    hits = [checks_to(order(t, JointRanker()), t.condition)
            for t in (synthetic_task(SMALL, SMALL_SAMPLE, rng, rows=300, noise=0.02) for _ in range(30))]
    assert float(np.mean(hits)) <= 1.5


@pytest.mark.slow
def test_the_benchmark_settings_a_and_c():
    """benchmarks/laws_rank_joint.py on 20 tasks of A and C (one seed): the joint statistic needs at most a fifth of the
    blind checks and two thirds of the raw control's, no more than the trained bag of atoms; the shuffled control keeps
    none of the gain (its mean at least half the blind mean and 20 times the joint statistic's)."""
    spec = importlib.util.spec_from_file_location(
        "laws_rank_joint", Path(__file__).resolve().parents[1] / "benchmarks" / "laws_rank_joint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in "AC":
        r = mod.run(name, seed=0, tasks=20, train=60)
        m = {o: v["mean"] for o, v in r["orders"].items()}
        assert m["joint"] <= m["blind"] / 5 and m["joint"] <= m["feature"] / 1.5 and m["joint"] <= m["bag"]
        assert m["joint_shuffled"] >= m["blind"] / 2 and m["joint_shuffled"] >= 20 * m["joint"]
