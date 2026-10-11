"""PriorRanker and HandoverRanker: an order for candidate conditions before the buffer has rows, and the hand-over to
the joint statistic once it has them.

PriorRanker scores a condition by its own features — its number of atoms, its atoms' kinds (feature name, comparison
type), each atom's prior usefulness and the pairwise co-occurrence of its atoms' features in true conditions — learned
from synthetic tasks drawn from the grammar; it reads no rows. Positive control: on a planted grammar whose condition
sampler favours some features, at 0 rows it needs at most half the blind checks. Negative control: the same prior fitted
on shuffled labels is not better than the blind order. Fallback: a fit whose held-out checks are not 5% below both the
blind order's and the shuffled-label fit's on its own training tasks records learned=False and scores as the blind
order — on a family whose true condition is any candidate with equal chance it falls back, on the planted grammar it
does not; on the rank test's grammar (drawn uniformly, a small signal left) it is not worse than blind by more than 5%. HandoverRanker scores by the prior below `min_rows` logged rows of
the action and by JointRanker from there on, and says which one scored. Deterministic for a seed."""
import importlib.util
import random
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, Buffer, Candidate, Enumerate, Grammar, HandoverRanker, JointRanker,
                                         LawSet, PriorRanker, RankedProposer, Transition, atom_kind, checks_to,
                                         synthetic_task)

CATS = [f"c{i}" for i in range(7)]
COUNTERS = ["n0", "n1", "n2"]
FAVOURED = ("c0", "c1", "c2")


def atoms():
    out = [Atom(f"{c}={v}", (lambda s, c=c, v=v: s[c] == v), feature=c) for c in CATS for v in "abc"]
    out += [Atom(f"{n}>={t}", (lambda s, n=n, t=t: s[n] >= t), feature=n) for n in COUNTERS for t in (1, 2, 3)]
    return out


class Favoured(Grammar):
    """The planted grammar: its condition sampler draws the features c0, c1, c2 `weight` times as often as the others."""
    weight = 30

    def random_condition(self, rng, size):
        by_feat: dict = {}
        for a in self.atoms:
            by_feat.setdefault(a.feature, []).append(a.name)
        feats = sorted(by_feat)
        chosen: list = []
        while len(chosen) < min(size, len(feats)):
            rest = [f for f in feats if f not in chosen]
            chosen.append(rng.choices(rest, weights=[self.weight if f in FAVOURED else 1 for f in rest])[0])
        return tuple(sorted(rng.choice(by_feat[f]) for f in chosen))


def sample(rng):
    s = {c: rng.choice("abc") for c in CATS}
    s.update({n: rng.randint(0, 4) for n in COUNTERS})
    return s


FAV = Favoured(atoms(), k=3)


def fav_tasks(seed, n):
    rng = random.Random(seed)
    return [synthetic_task(FAV, sample, rng, rows=200, noise=0.02) for _ in range(n)]


def order(t, scorer, buffer=None):
    return RankedProposer(Enumerate(), scorer, t.buffer if buffer is None else buffer).propose(
        t.lawset, t.state, t.action, t.observed)


@pytest.fixture(scope="module")
def prior():
    return PriorRanker(seed=0).fit(FAV, sample, tasks=150, rows=200, noise=0.02)


@pytest.fixture(scope="module")
def shuffled():
    return PriorRanker(seed=0, shuffle_labels=True, fallback=False).fit(FAV, sample, tasks=150, rows=200, noise=0.02)


class _Fixed:
    def __init__(self, cands):
        self.cands = cands

    def propose(self, lawset, state, action, observed):
        return list(self.cands)


def test_atom_kinds_are_read_off_the_name():
    assert atom_kind("c0=a") == "=" and atom_kind("n0>=2") == ">=" and atom_kind("x<=1") == "<="
    assert atom_kind("x!=1") == "!=" and atom_kind("x==1") == "=" and atom_kind("x<3") == "<"
    assert atom_kind("x>3") == ">" and atom_kind("door_open") == "bool"


def test_the_prior_reads_no_rows_and_an_unfitted_prior_is_the_blind_order(prior):
    t = fav_tasks(3, 1)[0]
    cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    assert prior.score(cands, t.lawset, t.buffer) == prior.score(cands, t.lawset, Buffer(FAV))
    assert prior.score(cands, t.lawset, t.buffer, exclude="g0") == prior.score(cands, t.lawset, t.buffer)
    assert prior.score([], t.lawset, t.buffer) == []
    blank = PriorRanker()
    assert set(blank.score(cands, t.lawset, t.buffer)) == {0.0}
    assert [c.condition for c in order(t, blank)] == [c.condition for c in cands]
    assert prior.trained_on == 150 and len(prior.weights) == len(prior.feature_names)


def test_the_prior_learns_the_favoured_features_and_the_size(prior):
    # prior usefulness: an atom of a favoured feature is in the true condition far more often when it is true
    assert min(prior.usefulness["feature"][f] for f in FAVOURED) > max(
        prior.usefulness["feature"][f] for f in CATS[3:])
    # the top of the order is made of favoured features; the empty condition (never a true one) is not near the top
    t = fav_tasks(4, 1)[0]
    cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    ranked = RankedProposer(Enumerate(), prior, Buffer(FAV)).propose(t.lawset, t.state, t.action, t.observed)
    assert all(FAV.feature[x] in FAVOURED for c in ranked[:3] for x in c.condition)
    assert [c.condition for c in ranked].index(()) >= len(cands) // 2


def test_the_prior_only_reorders_and_is_deterministic(prior):
    t = fav_tasks(11, 1)[0]
    base = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    for scorer in (prior, HandoverRanker(prior, JointRanker())):
        ranked = order(t, scorer)
        assert Counter(base) == Counter(ranked) and len(base) == len(ranked)
    again = PriorRanker(seed=0).fit(FAV, sample, tasks=150, rows=200, noise=0.02)
    assert np.array_equal(again.weights, prior.weights)
    assert [c.condition for c in order(t, again)] == [c.condition for c in order(t, prior)]
    other = PriorRanker(seed=1).fit(FAV, sample, tasks=40, rows=200, noise=0.02)
    assert other.trained_on == 40


def test_at_zero_rows_the_prior_needs_at_most_half_the_blind_checks_and_the_shuffled_prior_does_not(prior, shuffled,
                                                                                                       capsys):
    got: dict = {"blind": [], "prior": [], "shuffled": []}
    for t in fav_tasks(2024, 40):
        empty = Buffer(FAV)
        got["blind"].append(checks_to(Enumerate().propose(t.lawset, t.state, t.action, t.observed), t.condition))
        got["prior"].append(checks_to(order(t, prior, empty), t.condition))
        got["shuffled"].append(checks_to(order(t, shuffled, empty), t.condition))
    assert all(None not in v for v in got.values())
    mean = {a: float(np.mean(v)) for a, v in got.items()}
    with capsys.disabled():
        print("\n[rank prior] checks to the true condition at 0 rows, planted grammar, mean over 40 tasks: "
              + ", ".join(f"{a} {m:.2f}" for a, m in mean.items()))
    assert mean["prior"] <= mean["blind"] / 2
    assert mean["shuffled"] >= mean["blind"]


def test_the_hand_over_happens_at_min_rows_and_says_who_scored(prior):
    t = fav_tasks(7, 1)[0]
    cands = Enumerate().propose(t.lawset, t.state, t.action, t.observed)
    h = HandoverRanker(prior, JointRanker(), min_rows=20)
    assert h.last is None
    rows = t.buffer.rows()
    for n, who in ((0, "prior"), (5, "prior"), (19, "prior"), (20, "joint"), (100, "joint")):
        b = Buffer(FAV)
        b.extend(rows[:n])
        s = h.score(cands, t.lawset, b)
        assert h.last == who
        ref = (prior if who == "prior" else JointRanker()).score(cands, t.lawset, b)
        assert s == ref
    assert h.scored == Counter({"prior": 3, "joint": 2})
    # the rows that count are the action's rows without the group `exclude`
    b = Buffer(FAV)
    b.extend(Transition(r.state, r.action, r.observed, group="g0" if i < 10 else "g1") for i, r in enumerate(rows[:25]))
    h.score(cands, t.lawset, b)
    assert h.last == "joint"
    h.score(cands, t.lawset, b, exclude="g0")
    assert h.last == "prior"
    assert h.score([], t.lawset, b) == []
    with pytest.raises(ValueError):
        HandoverRanker(prior, JointRanker(), min_rows=-1)
    # rows of another action do not count
    b2 = Buffer(FAV)
    b2.extend(Transition(r.state, "other", r.observed, group=r.group) for r in rows[:50])
    h.score(cands, t.lawset, b2)
    assert h.last == "prior"


def test_the_ranked_proposer_takes_either_and_reports_who_scored(prior):
    t = fav_tasks(9, 1)[0]
    p = RankedProposer(Enumerate(), HandoverRanker(prior), Buffer(FAV))
    p.propose(t.lawset, t.state, t.action, t.observed)
    assert p.scored_by == "prior"
    p2 = RankedProposer(Enumerate(), HandoverRanker(prior), t.buffer)
    p2.propose(t.lawset, t.state, t.action, t.observed)
    assert p2.scored_by == "joint"
    p3 = RankedProposer(Enumerate(), prior, t.buffer)
    p3.propose(t.lawset, t.state, t.action, t.observed)
    assert p3.scored_by == "PriorRanker"
    # inside the law loop: the acceptance stays the held-out test
    laws = LawSet(FAV, base=lambda s, a: "off", proposer=RankedProposer(Enumerate(), HandoverRanker(prior), t.buffer))
    held = Buffer(FAV)
    held.extend(Transition(r.state, r.action, r.observed, group=f"h{i % 4}") for i, r in enumerate(t.buffer.rows()))
    st = laws.step(Transition(t.state, t.action, t.observed, group="new"), held)
    assert st.mismatch and st.candidates > 0


def test_fit_with_sizes_and_from_tasks():
    a = PriorRanker(seed=2).fit(FAV, sample, tasks=30, rows=200, sizes=(2, 3))
    rng = random.Random(2)
    ts = []
    for _ in range(30):
        ts.append(synthetic_task(FAV, sample, rng, rows=200, size=rng.randint(2, 3)))
    b = PriorRanker(seed=2).fit_tasks(FAV, ts)
    assert np.array_equal(a.weights, b.weights) and b.trained_on == 30
    with pytest.raises(ValueError):
        PriorRanker().fit_tasks(FAV, [])
    # an atom the prior never saw (the grammar grew) still scores, at the prior's default usefulness
    g = Favoured(atoms(), k=3)
    p = PriorRanker(seed=0).fit(g, sample, tasks=20, rows=100)
    g.add(Atom("new>=1", lambda s: True, feature="new"))
    s = p.score([Candidate("act", ("c0=a", "new>=1"), "on"), Candidate("act", ("c0=a",), "on")],
                LawSet(g, base=lambda s, a: "off"), Buffer(g))
    assert all(np.isfinite(s))
    assert [c.condition for c in RankedProposer(_Fixed([]), p, Buffer(g)).propose(None, None, "act", "on")] == []


@pytest.mark.slow
def test_the_benchmark_curve_has_the_prior_and_the_hand_over():
    """benchmarks/laws_rank_joint.py --curve on 10 tasks of B (one seed): the prior and the hand-over are in every cut;
    at 0 rows every row-statistic order is the blind order; the hand-over equals the prior below 20 rows and the
    joint statistic from 20 rows on."""
    spec = importlib.util.spec_from_file_location(
        "laws_rank_joint", Path(__file__).resolve().parents[1] / "benchmarks" / "laws_rank_joint.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.run("B", seed=0, tasks=10, train=30, curve=True)
    c = r["curve"]
    for o in ("prior", "handover", "prior_shuffled"):
        assert all(o in c[n] for n in c)
    assert c["0"]["joint"]["mean"] == c["0"]["blind"]["mean"]
    assert c["5"]["handover"] == c["5"]["prior"] and c["20"]["handover"] == c["20"]["joint"]
    assert c["300"]["handover"] == c["300"]["joint"]


def test_the_fallback_keeps_a_prior_that_learned_something_and_drops_one_that_did_not(prior, shuffled):
    assert prior.learned is True
    assert prior.cv["prior"] <= 0.95 * min(prior.cv["blind"], prior.cv["shuffled"])
    # the raw control (fallback=False) keeps its weights; with the fallback on, the shuffled fit is the blind order
    assert shuffled.learned is None and np.any(shuffled.weights != 0)
    fell = PriorRanker(seed=0, shuffle_labels=True).fit(FAV, sample, tasks=150, rows=200, noise=0.02)
    assert fell.learned is False
    t = fav_tasks(5, 1)[0]
    blind = [c.condition for c in Enumerate().propose(t.lawset, t.state, t.action, t.observed)]
    assert [c.condition for c in order(t, fell, Buffer(FAV))] == blind
    # a family whose true condition is any candidate of the state with equal chance: nothing to learn → the blind order
    G3 = Grammar(atoms(), k=3)
    rng = random.Random(5)
    flat_tasks = []
    for _ in range(150):
        st = sample(rng)
        conds = [c for c in G3.conditions(G3.true(st, "act")) if c]
        flat_tasks.append(SimpleNamespace(state=st, action="act", condition=rng.choice(conds)))
    flat = PriorRanker(seed=0).fit_tasks(G3, flat_tasks)
    assert flat.learned is False and set(flat.cv) == {"prior", "shuffled", "blind"}
    st = sample(random.Random(9))
    lawset = LawSet(G3, base=lambda s, a: "off")
    cands = Enumerate().propose(lawset, st, "act", "on")
    ranked = RankedProposer(Enumerate(), flat, Buffer(G3)).propose(lawset, st, "act", "on")
    assert [c.condition for c in ranked] == [c.condition for c in cands]
    # the rank test's grammar (conditions drawn uniformly from the grammar, sizes 1..3): its counters' thresholds of
    # different rates and the empty condition the blind order puts first leave a small signal; whichever way the
    # fallback decides, the prior is not worse than blind on fresh tasks by more than 5%
    a3 = PriorRanker(seed=0).fit(G3, sample, tasks=150, rows=200, noise=0.02)
    rng = random.Random(77)
    b, q = [], []
    for _ in range(40):
        u = synthetic_task(G3, sample, rng, rows=200, noise=0.02)
        b.append(checks_to(Enumerate().propose(u.lawset, u.state, u.action, u.observed), u.condition))
        q.append(checks_to(order(u, a3, Buffer(G3)), u.condition))
    assert np.mean(q) <= 1.05 * np.mean(b)
    with pytest.raises(ValueError):
        PriorRanker(margin=1.0)
