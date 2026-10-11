"""Ranking candidate laws by the joint statistic of the whole condition (`JointRanker`, no training) against the blind
order, the raw per-atom control (`FeatureRanker`), the trained bag of atoms (`BagOfAtomsRanker`), a negative control
(the same scorer against an effect mask whose rows are shuffled), the prior order that reads no rows (`PriorRanker`, its
control fitted on shuffled labels) and the hand-over from the prior to the joint statistic at 20 rows
(`HandoverRanker`), on synthetic tasks drawn from three grammars:

    python benchmarks/laws_rank_joint.py                  # A, B, C at seed 0, as JSON (a few minutes on a laptop CPU)
    python benchmarks/laws_rank_joint.py --settings A --tasks 20 --train 50
    python benchmarks/laws_rank_joint.py --settings B --curve     # + the rows curve: 0 / 5 / 20 / 100 / 300 rows
    python benchmarks/laws_rank_joint.py --curve --tasks 20       # the rows curve on A, B and C

A — 7 categorical features (3 values) and 3 counters (thresholds 1, 2, 3): 30 atoms, k = 3, conditions of 1–3 atoms,
    2% label noise (the grammar of tests/test_laws_rank.py).
B — deep conditions: 10 categorical features and 3 counters, k = 5, conditions of 3–5 atoms, 5% label noise.
C — correlated features: the grammar of A, but c1 copies c0, c3 copies c2 and c5 copies c4 with probability 0.8 (a twin of
    a true atom looks almost as good atom by atom); conditions of 2–3 atoms, 2% label noise.

Per task: the candidates of `Enumerate` for the mismatch, reordered by each scorer through `RankedProposer` (the same
set, checked), and `checks_to` the true condition — its 1-based position among the distinct conditions. The bag of atoms
and the prior are trained on `--train` tasks of the same setting (the same condition sizes). Reported per setting and order: mean,
median, the share found first, and the distinct candidates per task. `--curve`: the same tasks with each buffer cut to
its first 0 / 5 / 20 / 100 / 300 logged rows (the bag of atoms as trained) — a statistic over rows needs rows; with
none every row scorer gives the blind order, the prior does not read rows, and the hand-over is the prior below 20 rows
and the joint statistic from 20 on. Deterministic for a seed."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import Counter

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, BagOfAtomsRanker, Buffer, Enumerate, FeatureRanker, Grammar,
                                         HandoverRanker, JointRanker, PriorRanker, RankedProposer, atom_features,
                                         checks_to, synthetic_task)

ORDERS = ("blind", "feature", "bag", "joint", "joint_shuffled", "prior", "handover", "prior_shuffled")
CUTS = (0, 5, 20, 100, 300)


def grammar(ncat: int, ncount: int, k: int):
    cats = [f"c{i}" for i in range(ncat)]
    counters = [f"n{i}" for i in range(ncount)]
    atoms = [Atom(f"{c}={v}", (lambda s, c=c, v=v: s[c] == v), feature=c) for c in cats for v in "abc"]
    atoms += [Atom(f"{n}>={t}", (lambda s, n=n, t=t: s[n] >= t), feature=n) for n in counters for t in (1, 2, 3)]
    return Grammar(atoms, k=k), cats, counters


def sampler(cats, counters, corr: float = 0.0):
    def sample(rng):
        s = {c: rng.choice("abc") for c in cats}
        if corr:
            for i in range(0, len(cats) - 1, 2):
                if i + 1 < 6 and rng.random() < corr:
                    s[cats[i + 1]] = s[cats[i]]
        s.update({n: rng.randint(0, 4) for n in counters})
        return s
    return sample


def settings():
    """name → (grammar, sample, noise, condition sizes or None for uniform 1..k)."""
    ga, ca, na = grammar(7, 3, 3)
    gb, cb, nb = grammar(10, 3, 5)
    return {"A": (ga, sampler(ca, na), 0.02, None),
            "B": (gb, sampler(cb, nb), 0.05, (3, 5)),
            "C": (ga, sampler(ca, na, corr=0.8), 0.02, (2, 3))}


class ShuffledMask(JointRanker):
    """The negative control: the effect's rows shuffled among the allowed rows (same count; seeded by the seed and the
    mask, so each task gets its own permutation)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def effect_mask(self, lawset, buffer, action, effect, exclude=None):
        E = super().effect_mask(lawset, buffer, action, effect, exclude)
        allowed = buffer.allowed(action, exclude)
        idx = [i for i in range(allowed.bit_length()) if allowed >> i & 1]
        bits = [E >> i & 1 for i in idx]
        random.Random(self.seed * 1_000_003 + E).shuffle(bits)    # a permutation per task, seeded
        return sum(b << i for b, i in zip(bits, idx))


def train_bag(g, sample, noise, sizes, tasks: int, seed: int, rows: int = 300) -> BagOfAtomsRanker:
    """BagOfAtomsRanker on `tasks` tasks of the setting itself (its condition sizes): fit's loop with the sizes."""
    r = BagOfAtomsRanker(seed=seed)
    rng = random.Random(seed)
    Xs, ys = [], []
    for _ in range(tasks):
        size = rng.randint(*sizes) if sizes else None
        t = synthetic_task(g, sample, rng, rows=rows, noise=noise, size=size)
        names, X = atom_features(t.lawset, t.buffer, t.action, t.observed, top_pairs=r.top_pairs)
        true = set(g.true(t.state, t.action))
        for i, x in enumerate(names):
            if x in true:
                Xs.append(X[i])
                ys.append(1.0 if x in t.condition else 0.0)
    return r.fit_arrays(np.array(Xs), np.array(ys), tasks=tasks)


def _summary(got: dict) -> dict:
    out = {}
    for o, v in got.items():
        found = [x for x in v if x is not None]
        out[o] = {"mean": float(np.mean(found)), "median": float(np.median(found)),
                  "first": float(np.mean([x == 1 for x in found])), "not_found": len(v) - len(found)}
    return out


def _cut(buffer, n: int):
    b = Buffer(buffer.grammar)
    b.extend(buffer.rows()[:n])
    return b


def run(name: str, seed: int = 0, tasks: int = 60, train: int = 150, rows: int = 300, curve: bool = False) -> dict:
    g, sample, noise, sizes = settings()[name]
    t0 = time.time()
    prior = PriorRanker(seed=seed).fit(g, sample, tasks=train, rows=rows, noise=noise, sizes=sizes)
    scorers = {"feature": FeatureRanker("jaccard"), "bag": train_bag(g, sample, noise, sizes, train, seed, rows),
               "joint": JointRanker(), "joint_shuffled": ShuffledMask(seed), "prior": prior,
               "handover": HandoverRanker(prior, JointRanker(), min_rows=20),
               "prior_shuffled": PriorRanker(seed=seed, shuffle_labels=True).fit(g, sample, tasks=train, rows=rows,
                                                                               noise=noise, sizes=sizes)}
    rng = random.Random(10_000 + seed)
    got = {o: [] for o in ORDERS}
    cut = {n: {o: [] for o in ORDERS} for n in CUTS} if curve else {}
    ncand = []
    for _ in range(tasks):
        size = rng.randint(*sizes) if sizes else None
        t = synthetic_task(g, sample, rng, rows=rows, noise=noise, size=size)
        base = list(Enumerate().propose(t.lawset, t.state, t.action, t.observed))
        ncand.append(len({tuple(c.condition) for c in base}))
        got["blind"].append(checks_to(base, t.condition))
        for o, scorer in scorers.items():
            order = RankedProposer(Enumerate(), scorer, t.buffer).propose(t.lawset, t.state, t.action, t.observed)
            if Counter(order) != Counter(base):
                raise AssertionError(f"{o} changed the candidate set")
            got[o].append(checks_to(order, t.condition))
        for n in cut:
            b = _cut(t.buffer, n)
            cut[n]["blind"].append(checks_to(base, t.condition))
            for o, scorer in scorers.items():
                order = RankedProposer(Enumerate(), scorer, b).propose(t.lawset, t.state, t.action, t.observed)
                cut[n][o].append(checks_to(order, t.condition))
    out = {"setting": name, "seed": seed, "tasks": tasks, "train": train, "candidates": float(np.mean(ncand)),
           "seconds": round(time.time() - t0, 1), "orders": _summary(got)}
    if curve:
        out["curve"] = {str(n): _summary(v) for n, v in cut.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", default="ABC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=60)
    ap.add_argument("--train", type=int, default=150)
    ap.add_argument("--curve", action="store_true", help="also the rows curve (0 / 5 / 20 / 100 / 300 rows)")
    a = ap.parse_args(argv)
    out = [run(s, a.seed, a.tasks, a.train, curve=a.curve) for s in a.settings]
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
