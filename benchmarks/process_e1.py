"""E1 — the process promise on a synthetic insurance claim with a known truth (the pre-registered stand, bars frozen
2026-10-23). The harness for stages 3 and 4 of solvi.experimental.process: union with the equal and the cost split,
per_node, a baseline of the same nodes glued by hand (stage 3), and end_to_end over the two built-in threshold families
at three sizes of the end-labelled calibration set (stage 4). Adding the stage-4 arms changes no stage-3 result (a
unit's data, models and human answers depend on its condition and seed only; the end-labelled set comes from a stream
of its own that only the stage-4 arms draw).

The generator. A claim has hidden true variables — kind (5), amount (its band: low / mid / high), documents complete,
fraud, region — and a common latent h. Expert i sees its own target y through x = e_y + σ_i(√(1−ρ)·ε_i + √ρ·h·(e_{y+1}
− e_y)): ρ = 0.5 pushes every expert towards a wrong answer on the same claims (errors correlated, "one model"), and σ_i
is set so that the arg-max of x is right with the node's accuracy a_i ∈ [0.8, 0.97]. Each node is a solvi.build
decision system whose rule is a small logistic head on x (numpy), fitted on its own fit set and calibrated by CRC at
its share of the risk. The end label is a deterministic function of the truth: payout = deny if fraud, investigate if
the documents are missing or a high claim of kind 3 or 4, else pay; with label noise p it is replaced by another
answer with chance p. Node labels come in both kinds (G1): (a) by the case (the node's question about the true claim)
and (b) conditional (right given the inputs the node was given): they differ at the payout nodes, whose target is the
payout of the kind, band, documents and fraud they are given.

The graph, K=10: three triages in a row (kind, band, documents), then one of four fraud checks by the kind (k0, k1,
k2, k3+k4), then one of three payouts by the band; one path per claim (stage 1 of the process runs one path), K_auto =
10. K=100 (20 regional triages, 60 checks, 20 payouts, merges "all") needs merge "all", which the process does not run
yet: its units are skipped with that reason. A node that does not answer alone hands its question to a person through a
MemoryQueue; a simulated person answers 1–3 steps of the stand later (a step = a batch of claims) with accuracy 0.98.
At the end of the test stream the shares decided / pending / with a person are taken, then every open question is
answered and every run resumed (no ticket may be lost).

Conditions (each × 10 seeds): rho ∈ {0, 0.5} × labels ∈ {a, b} (in place, no noise); noise 0.05; calibrated apart
(on clean inputs) × rho; a faulty expert (one check at accuracy 0.5 after calibration) with and without its
DriftMonitor. Arms: union_equal, union_cost (r_i ∝ 1 / cost of a node's error: triage 1, check 2, payout 4), per_node
(every node at r = 0.05), glued (the union_equal nodes run by a Python loop with run_context by hand); stage 4:
e2e_<family>_n<n> — end_to_end, r = 0.05, δ = 0.1, family shared_rank or per_node_fixed (order end → start), calibrated
by Process.calibrate on the first n ∈ {500, 1000, 3000} claims of the end-labelled stream (labels: the end label, with
the condition's noise). Its nodes are built at max_risk 0.5 (their own guard nearly open), so the process threshold,
not the node's, decides who answers alone; the node calibration (in place or apart) is the same as the other arms'.
Sizes: fit 2000, node calibration 2000, test 5000 claims per seed; r = 0.05.

    OMP_NUM_THREADS=4 uv run python benchmarks/process_e1.py --out DIR --smoke
    OMP_NUM_THREADS=4 uv run python benchmarks/process_e1.py --out DIR --only K=10
    OMP_NUM_THREADS=4 uv run python benchmarks/process_e1.py --out DIR --only K=10 --stage 4   # the end_to_end arms
    uv run python benchmarks/process_e1.py --out DIR --list            # the units, done or not
    uv run python benchmarks/process_e1.py --out DIR --report          # B1, B3, B5 and B2, B6 over the finished units

Every (arm, condition, seed) unit writes DIR/units/<unit>.json (written last, atomically: it marks the unit done)
and its own log DIR/logs/<unit>.log; a rerun skips finished units and redoes an unfinished one from scratch. A unit
that fails (an exception, a NaN in a head) is logged with its traceback, recorded in DIR/errors/<unit>.json and
skipped; the run goes on. A heartbeat line goes to stdout and the run's log at least every 30 s."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
import traceback
import warnings

import numpy as np

R_PROCESS = 0.05
SIZES = {"fit": 2000, "cal": 2000, "test": 5000, "step": 250}
SMOKE_SIZES = {"fit": 300, "cal": 2000, "test": 300, "step": 50}   # cal as the stand: union answers some
HUMAN_ACC = 0.98
KINDS = ["k0", "k1", "k2", "k3", "k4"]
BANDS = ["low", "mid", "high"]
YESNO = ["no", "yes"]
PAYOUT = ["pay", "deny", "investigate"]
GROUPS = {"c_fraud_0": ["k0"], "c_fraud_1": ["k1"], "c_fraud_2": ["k2"], "c_fraud_34": ["k3", "k4"]}
COST = {"triage": 1.0, "check": 2.0, "payout": 4.0}
FAULTY = "c_fraud_34"
E2E_DELTA = 0.1
E2E_NODE_RISK = 0.5            # the end_to_end arms' nodes: their own guard nearly open
N_END_MAX = 3000               # the end-labelled stream; an arm with n takes its first n claims


# --- the graph (K=10)

def nodes_k10():
    """name → {layer, question, options, target, acc}."""
    out = {"t_kind": {"layer": "triage", "question": "kind", "options": KINDS, "acc": 0.95},
           "t_band": {"layer": "triage", "question": "band", "options": BANDS, "acc": 0.93},
           "t_docs": {"layer": "triage", "question": "docs", "options": YESNO, "acc": 0.97}}
    for n, a in zip(GROUPS, (0.90, 0.88, 0.92, 0.85)):
        out[n] = {"layer": "check", "question": "fraud", "options": YESNO, "acc": a}
    for b, a in zip(BANDS, (0.96, 0.94, 0.90)):
        out[f"p_{b}"] = {"layer": "payout", "question": "payout", "options": PAYOUT, "acc": a}
    return out


def payout_of(kind, band, docs, fraud):
    """The end rule: deny if fraud, investigate if the documents are missing or a high claim of kind 3 or 4, else pay."""
    if fraud == "yes":
        return "deny"
    if docs == "no" or (band == "high" and kind in ("k3", "k4")):
        return "investigate"
    return "pay"


def check_of(kind):
    return next(n for n, ks in GROUPS.items() if kind in ks)


# --- the generator

_SIGMA = {}


def sigma_for(acc, C, rho):
    """σ so that the arg-max of e_y + σ(√(1−ρ)ε + √ρ·h·(e_{y+1} − e_y)) is right with chance acc (bisection on a fixed
    Monte Carlo sample: the same σ in every run)."""
    key = (round(acc, 6), C, rho)
    if key in _SIGMA:
        return _SIGMA[key]
    rng = np.random.default_rng(12345)
    n = 40000
    eps, h = rng.normal(size=(n, C)), rng.normal(size=n)
    d = np.zeros(C)
    d[1 % C], d[0] = 1.0, -1.0

    def accuracy(s):
        x = np.zeros((n, C))
        x[:, 0] = 1.0
        x = x + s * (math.sqrt(1 - rho) * eps + math.sqrt(rho) * h[:, None] * d[None, :])
        return float((x.argmax(1) == 0).mean())
    lo, hi = 1e-3, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if accuracy(mid) > acc:
            lo = mid
        else:
            hi = mid
    _SIGMA[key] = (lo + hi) / 2
    return _SIGMA[key]


def observe(y, C, sigma, rho, eps, h):
    """x for targets y (ints, n) from the noise eps (n, C) and h (n)."""
    n = len(y)
    x = np.zeros((n, C))
    x[np.arange(n), y] = 1.0
    d = np.zeros((n, C))
    d[np.arange(n), (y + 1) % C] = 1.0
    d[np.arange(n), y] -= 1.0
    return x + sigma * (math.sqrt(1 - rho) * eps + math.sqrt(rho) * h[:, None] * d)


def gen_cases(rng, n, cond, nodes, faulty=False):
    """n claims → list of dicts: the truth (kept apart from what the nodes read), the given facts, the end label."""
    rho = cond["rho"]
    T = rng.integers(5, size=n)
    amount = rng.lognormal(7.0, 1.0, size=n)
    B = np.digitize(np.log(amount), [7.0 - 0.4307, 7.0 + 0.4307])
    D = (rng.random(n) < 0.8).astype(int)
    F = (rng.random(n) < 0.10 + 0.10 * (T >= 3)).astype(int)
    region = rng.integers(cond["regions"], size=n)
    h = rng.normal(size=n)
    truth = {"kind": T, "band": B, "docs": D, "fraud": F}
    given = [{"h": float(h[i]), "region": int(region[i])} for i in range(n)]
    for name, ns in nodes.items():
        C = len(ns["options"])
        eps = rng.normal(size=(n, C))
        acc = 0.5 if (faulty and name == FAULTY) else ns["acc"]
        sig = sigma_for(acc, C, rho)
        if ns["layer"] == "payout":
            for i in range(n):
                given[i][f"eps_{name}"] = [float(v) for v in eps[i]]
            continue
        x = observe(truth[ns["question"]], C, sig, rho, eps, h)
        for i in range(n):
            given[i][f"obs_{name}"] = [float(v) for v in x[i]]
    noise = rng.random(n) < cond["noise"]
    other = rng.integers(1, 3, size=n)
    out = []
    for i in range(n):
        t = {"kind": KINDS[T[i]], "band": BANDS[B[i]], "docs": YESNO[D[i]], "fraud": YESNO[F[i]]}
        y = payout_of(**t)
        yl = PAYOUT[(PAYOUT.index(y) + int(other[i])) % 3] if noise[i] else y
        out.append({"truth": t, "end": y, "end_label": yl, "given": given[i]})
    return out


# --- the experts: a logistic head on x, a solvi.build decision system

def fit_logistic(X, y, C, iters=300, lr=0.5, l2=1e-3):
    """Multinomial logistic regression by gradient descent → (W, b); raises FloatingPointError on a NaN."""
    n, d = X.shape
    W, b = np.zeros((d, C)), np.zeros(C)
    Y = np.eye(C)[y]
    for _ in range(iters):
        z = X @ W + b
        z -= z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    if not (np.isfinite(W).all() and np.isfinite(b).all()):
        raise FloatingPointError("the logistic head diverged (NaN)")
    return W, b


class Expert:
    """The rule of one node: x from what it is given → Decision(answer, probs). A triage or check reads its x
    (obs_<node>); a payout makes its x from the kind, band, documents and fraud it is given and its own noise (its
    target is the payout of its inputs, so it is right "given right inputs")."""

    def __init__(self, name, ns, W, b, sigma, rho):
        import inspect
        self.name, self.options, self.W, self.b, self.sigma, self.rho = name, ns["options"], W, b, sigma, rho
        self.payout = ns["layer"] == "payout"
        self.reads = ([f"eps_{name}", "h", "kind", "band", "docs", "fraud"] if self.payout else [f"obs_{name}"])
        self.__signature__ = inspect.Signature([inspect.Parameter(r, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for r in self.reads])
        self.__name__ = f"expert_{name}"

    def x(self, **f):
        if not self.payout:
            return np.asarray(f[f"obs_{self.name}"], float)[None, :]
        y = PAYOUT.index(payout_of(f["kind"], f["band"], f["docs"], f["fraud"]))
        return observe(np.array([y]), 3, self.sigma, self.rho, np.asarray(f[f"eps_{self.name}"], float)[None, :],
                       np.array([float(f["h"])]))

    def probs(self, X):
        z = X @ self.W + self.b
        z = z - z.max(1, keepdims=True)
        P = np.exp(z)
        return P / P.sum(1, keepdims=True)

    def __call__(self, **f):
        from solvi import Decision
        p = self.probs(self.x(**f))[0]
        k = int(p.argmax())
        return Decision(self.options[k], probs={o: float(v) for o, v in zip(self.options, p)})


def fit_experts(fit_cases, nodes, rho):
    """Each node's head on its own fit set (clean inputs: the truth upstream) → {node: Expert}."""
    out = {}
    for name, ns in nodes.items():
        C = len(ns["options"])
        sig = sigma_for(ns["acc"], C, rho)
        e = Expert(name, ns, None, None, sig, rho)
        rows, ys = [], []
        for c in fit_cases:
            t = c["truth"]
            if ns["layer"] == "payout":
                rows.append(e.x(**{**c["given"], **t})[0])
                ys.append(PAYOUT.index(c["end"]))
            else:
                rows.append(np.asarray(c["given"][f"obs_{name}"]))
                ys.append(ns["options"].index(t[ns["question"]]))
        e.W, e.b = fit_logistic(np.array(rows), np.array(ys), C)
        out[name] = e
    return out


def build_node(name, ns, expert, examples, max_risk, storage, monitor):
    import solvi
    from solvi import Answer, Catalog, Question
    q = Question(ns["question"], f"{name}: {ns['question']}?", Answer.choice(list(ns["options"])))
    cat = Catalog()
    cat.rule(ns["question"])(expert)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solvi.build(q, examples, catalog=cat, max_risk=max_risk, novel=False, storage=storage,
                           monitor=monitor)


# --- human answers: the same for every arm (a hash of the unit's condition, seed, claim and node)

def _u(*key):
    return int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:13], 16) / 16 ** 13


def human_value(key, case, node, ns):
    """A person's answer to a node's question about the claim: the truth with chance 0.98."""
    right = case["end"] if ns["layer"] == "payout" else case["truth"][ns["question"]]
    if _u(key, "hv", node) < HUMAN_ACC:
        return right
    others = [o for o in ns["options"] if o != right]
    return others[int(_u(key, "ho", node) * len(others))]


def human_delay(key, node):
    return 1 + int(_u(key, "hd", node) * 3)


# --- the conditions, arms and units

def conditions(K):
    regions = 4 if K == 10 else 20
    base = {"K": K, "regions": regions, "noise": 0.0, "calib": "in_place", "faulty": None, "monitor": "auto"}
    out = {}
    for rho in (0.0, 0.5):
        for lab in ("a", "b"):
            out[f"K{K}_rho{int(rho * 10)}_{lab}"] = {**base, "rho": rho, "labels": lab}
    out[f"K{K}_rho0_a_noise5"] = {**base, "rho": 0.0, "labels": "a", "noise": 0.05}
    for rho in (0.0, 0.5):
        out[f"K{K}_rho{int(rho * 10)}_a_apart"] = {**base, "rho": rho, "labels": "a", "calib": "apart"}
    out[f"K{K}_rho0_a_faulty_mon"] = {**base, "rho": 0.0, "labels": "a", "faulty": FAULTY}
    out[f"K{K}_rho0_a_faulty_nomon"] = {**base, "rho": 0.0, "labels": "a", "faulty": FAULTY, "monitor": None}
    return out


ARMS = {
    "union_equal": {"stage": 3, "mode": "union", "split": "equal"},
    "union_cost": {"stage": 3, "mode": "union", "split": "cost"},
    "per_node": {"stage": 3, "mode": "per_node", "split": None},
    "glued": {"stage": 3, "mode": "baseline", "split": "equal"},
}
for _fam in ("shared_rank", "per_node_fixed"):         # stage 4
    for _n in (500, 1000, 3000):
        ARMS[f"e2e_{_fam}_n{_n}"] = {"stage": 4, "mode": "end_to_end", "family": _fam, "n_end": _n}


def shares(arm, nodes):
    """The risk each node is calibrated at → {node: r_i}."""
    a = ARMS[arm]
    if a["mode"] == "end_to_end":
        return {n: E2E_NODE_RISK for n in nodes}
    if a["mode"] == "per_node":
        return {n: R_PROCESS for n in nodes}
    if a["split"] == "cost":
        w = {n: 1.0 / COST[ns["layer"]] for n, ns in nodes.items()}
        s = sum(w.values())
        return {n: R_PROCESS * v / s * (1 - 1e-9) for n, v in w.items()}
    return {n: R_PROCESS / len(nodes) for n in nodes}


def unit_seed(cond_name, seed):
    return int(hashlib.sha256(f"e1|{cond_name}|{seed}".encode()).hexdigest()[:8], 16)


def unit_id(arm, cond_name, seed):
    return f"{arm}__{cond_name}__s{seed}"


class Skip(Exception):
    """A unit this stage cannot run (logged, no result written)."""


# --- one unit

class Status:
    def __init__(self):
        self.lock, self.d = threading.Lock(), {"unit": None, "phase": "start", "i": 0, "n": 0}

    def set(self, **kw):
        with self.lock:
            self.d.update(kw)

    def get(self):
        with self.lock:
            return dict(self.d)


def calibrate_layers(cond, nodes, experts, cal, r_i, stores, monitor, key, log):
    """Build the nodes layer by layer. In place: a node's examples are the claims that reach it with the inputs
    upstream gives (its answer alone, else a person's), its labels (a) by the case or (b) given its inputs; apart: the
    claims whose true route reaches it, with the true inputs. → ({node: system}, the reference routes)."""
    systems, given = {}, [dict(c["given"]) for c in cal]
    routes = [[] for _ in cal]

    def answer_of(sysm, ns, name, state, i):
        res = sysm.system.ask(state, store=False)
        r1 = res[ns["question"]]
        if r1.status != "abstain" and r1.answer is not None:
            return r1.answer
        return human_value(f"{key}|cal|{i}", cal[i], name, ns)

    for layer in ("triage", "check", "payout"):
        for name, ns in nodes.items():
            if ns["layer"] != layer:
                continue
            ex, idx = [], []
            for i, c in enumerate(cal):
                f = given[i]
                if layer == "check":
                    k = f["kind"] if cond["calib"] == "in_place" else c["truth"]["kind"]
                    if check_of(k) != name:
                        continue
                if layer == "payout":
                    b = f["band"] if cond["calib"] == "in_place" else c["truth"]["band"]
                    if f"p_{b}" != name:
                        continue
                st = {r: (f[r] if cond["calib"] == "in_place" or r not in c["truth"] else c["truth"][r])
                      for r in experts[name].reads}
                if layer == "payout":
                    if cond["labels"] == "b":
                        lab = payout_of(st["kind"], st["band"], st["docs"], st["fraud"])
                    else:
                        lab = c["end_label"]
                else:
                    lab = c["truth"][ns["question"]]
                ex.append((st, lab))
                idx.append(i)
            if len(ex) < 2:
                raise Skip(f"node {name}: {len(ex)} calibration examples")
            systems[name] = build_node(name, ns, experts[name], ex, r_i[name], stores[name],
                                       cond["monitor"] if (cond["faulty"] is None or name == cond["faulty"]) else None)
            for (st, _), i in zip(ex, idx):
                given[i][ns["question"]] = answer_of(systems[name], ns, name, st, i)
            log(f"calibrated {name}: {len(ex)} examples, r_i {r_i[name]:.5g}")
    for i, f in enumerate(given):
        routes[i] = [("t_kind", "t_band"), ("t_band", "t_docs"), ("t_docs", check_of(f["kind"])),
                     (check_of(f["kind"]), f"p_{f['band']}")]
    return systems, routes


def make_process(systems, nodes, arm, storage, queue):
    from solvi.experimental.process import Process
    p = Process("claim_e1", storage=storage, human=queue)
    for name, ns in nodes.items():
        p.node(name, systems[name], reads=_reads(name, ns), writes={ns["question"]: "answer"},
               end=ns["layer"] == "payout")
    ok = {"status": "ok"}             # a node that sent its decision to a person has no edge: the person is asked
    p.edge("t_kind", "t_band", when=ok).edge("t_band", "t_docs", when=ok)
    for c, ks in GROUPS.items():
        p.edge("t_docs", c, when={"and": [ok, {"answer": "kind", "in": ks}]})
        for b in BANDS:
            p.edge(c, f"p_{b}", when={"and": [ok, {"answer": "band", "eq": b}]})
    a = ARMS[arm]
    if a["mode"] == "end_to_end":
        p.promise(mode="end_to_end", risk=R_PROCESS, delta=E2E_DELTA, family=a["family"])
    elif a["mode"] == "per_node":
        p.promise(mode="per_node")
    elif a["split"] == "cost":
        r_i = shares(arm, nodes)
        p.promise(mode="union", risk=R_PROCESS, split=r_i)
    else:
        p.promise(mode="union", risk=R_PROCESS)
    return p


def with_store(p, path):
    """The same process (nodes, systems, edges, promise, channel and watch shared) writing its runs to `path`. One
    store per claim keeps the stand linear: the process finds a run's segments by reading its store (stage 1 has no
    index by run), which over one store of 5,000 runs made the stand quadratic."""
    import copy
    q = copy.copy(p)
    q.storage, q._store = path, None
    return q


def _reads(name, ns):
    return ([f"eps_{name}", "h", "kind", "band", "docs", "fraud"] if ns["layer"] == "payout" else [f"obs_{name}"])


def edge_index(p):
    return {(e["from"], e["to"]): i for i, e in enumerate(p.edges)}


def run_unit(arm, cond_name, cond, seed, sizes, unit_dir, log, status):
    """One (arm, condition, seed) → the result dict."""
    a = ARMS[arm]
    if a["stage"] > 4:
        raise Skip(f"arm {arm} is stage {a['stage']}: not implemented yet")
    if cond["K"] != 10:
        raise Skip("K=100 needs merge 'all' (20 regional triages in parallel), which the process does not run yet")
    us = unit_seed(cond_name, seed)
    random.seed(us)                                       # seeds before any model is built
    np.random.seed(us % 2**32)
    rng = np.random.default_rng(us)
    nodes = nodes_k10()
    t0 = time.time()
    status.set(phase="generate")
    fit = gen_cases(rng, sizes["fit"], cond, nodes)
    cal = gen_cases(rng, sizes["cal"], cond, nodes)
    test_rng = np.random.default_rng(us + 1)
    test = gen_cases(test_rng, sizes["test"], cond, nodes, faulty=cond["faulty"] is not None)
    end_cal = None
    if a["mode"] == "end_to_end":                         # a stream of its own: the stage-3 streams are untouched
        end_cal = gen_cases(np.random.default_rng(us + 2), N_END_MAX, cond, nodes)[:a["n_end"]]
    status.set(phase="fit")
    experts = fit_experts(fit, nodes, cond["rho"])
    from solvi.core.store import JSONLStorage
    stores = {n: JSONLStorage(str(os.path.join(unit_dir, f"node_{n}.jsonl"))) for n in nodes}
    r_i = shares(arm, nodes)
    status.set(phase="calibrate")
    key = f"{cond_name}|{seed}"
    systems, routes = calibrate_layers(cond, nodes, experts, cal, r_i, stores, cond["monitor"], key, log)
    t_cal = time.time() - t0
    status.set(phase="test", i=0, n=len(test))
    if a["mode"] == "baseline":
        out = run_glued(systems, nodes, test, key, sizes, log, status)
    else:
        out = run_process(systems, nodes, arm, test, key, sizes, unit_dir, routes, log, status, end_cal=end_cal)
    out.update(arm=arm, condition=cond_name, cond=cond, seed=seed, unit_seed=us, sizes=sizes, r=R_PROCESS,
               r_i=r_i, t_calibrate_s=t_cal, t_total_s=time.time() - t0,
               node_promises={n: s.system.guards[nodes[n]["question"]].promise.record() for n, s in systems.items()},
               drift_flags={n: bool(getattr(getattr(s.dispatcher, "monitor", None), "flagged", False))
                            for n, s in systems.items()})
    return out


def _tally(test, finished, nodes):
    """finished: {i: {"status", "answer", "human", "steps": [(node, by, answer, facts_in)]}} → the metrics."""
    n = len(test)
    auto = [i for i, f in finished.items() if f["status"] == "decided" and not f["human"]]
    hum = [i for i, f in finished.items() if f["status"] == "decided" and f["human"]]
    wrong_auto = sum(finished[i]["answer"] != test[i]["end_label"] for i in auto)
    wrong_auto_true = sum(finished[i]["answer"] != test[i]["end"] for i in auto)
    wrong_hum = sum(finished[i]["answer"] != test[i]["end_label"] for i in hum)
    other = sum(1 for f in finished.values() if f["status"] != "decided")
    per = {m: {"reached": 0, "alone": 0, "wrong_a": 0, "wrong_b": 0} for m in nodes}
    for i, f in finished.items():
        for node, by, ans, fin in f["steps"]:
            ns = nodes[node]
            s = per[node]
            s["reached"] += 1
            if by in ("s1", "s2"):
                s["alone"] += 1
                if ns["layer"] == "payout":
                    ra, rb = test[i]["end_label"], payout_of(fin["kind"], fin["band"], fin["docs"], fin["fraud"])
                else:
                    ra = rb = test[i]["truth"][ns["question"]]
                s["wrong_a"] += ans != ra
                s["wrong_b"] += ans != rb
    return {"n": n, "auto": len(auto), "human": len(hum), "other": other, "pending": n - len(finished),
            "coverage": len(auto) / n, "human_share": len(hum) / n, "pending_share": (n - len(finished)) / n,
            "e_hat": wrong_auto / n, "e_hat_true_label": wrong_auto_true / n,
            "e_cond": wrong_auto / max(1, len(auto)), "e_with_human": (wrong_auto + wrong_hum) / n,
            "wrong_auto": wrong_auto, "wrong_human": wrong_hum, "per_node": per}


def run_process(systems, nodes, arm, test, key, sizes, unit_dir, routes, log, status, end_cal=None):
    from solvi.experimental.process import MemoryQueue, PathWatch
    queue = MemoryQueue()
    step, B = 0, sizes["step"]
    procs, live, finished = {}, {}, {}
    watch = PathWatch(alpha=0.01, horizon=max(1000, len(test)), seed=0)
    p0 = make_process(systems, nodes, arm, None, queue)
    e2e = None
    if end_cal is not None:                               # end_to_end: calibrated on end-labelled runs first
        status.set(phase="calibrate_end", i=0, n=len(end_cal))
        tc = time.time()
        th = p0.calibrate([(dict(c["given"]), c["end_label"]) for c in end_cal])
        pr = p0.spec()["promise"]
        e2e = {"n_end": len(end_cal), "thresholds": th, "calibrated_on": pr["calibrated_on"],
               "family": pr["family"], "t_calibrate_end_s": time.time() - tc}
        log(f"end_to_end calibrated on {len(end_cal)}: {json.dumps(th)}")
    p0.monitor(watch)
    os.makedirs(os.path.join(unit_dir, "process"), exist_ok=True)
    ei = edge_index(p0)
    watch.set_reference([[ei[e] for e in r] for r in routes], names=[f"{e['from']} -> {e['to']}" for e in p0.edges])
    g0 = p0.guarantee()
    t0, asks, tickets = time.time(), 0, 0
    flag_at = None

    def take(i, r, k):
        nonlocal asks, tickets, flag_at
        rec = live.setdefault(i, {"steps": [], "human": False, "proc": k, "run": r.run})
        for st in r.path:
            if st["kind"] != "human":
                asks += 1
            if st["kind"] == "human" or st["by"] == "human":
                rec["human"] = True
            else:                             # one path, no loop: each fact is written once, by this node
                fin = {x: r.facts.get(x) for x in ("kind", "band", "docs", "fraud")}
                rec["steps"].append((st["node"], st["by"], r.facts.get(nodes[st["node"]]["question"]), fin))
        if r.path_drift and r.path_drift.get("drift") and flag_at is None:
            flag_at = r.path_drift["seen"]
        if r.status == "pending":
            tickets += 1
            req = queue._q[r.ticket]["request"]
            rec.update(ticket=r.ticket, node=req.node, due=step + human_delay(f"{key}|{i}|{tickets}", req.node))
        else:
            rec.update(status=r.status, answer=r.answer, auto=r.automatic, risk=r.risk)
            finished[i] = live.pop(i)

    def answer_due(final=False):
        for i, rec in list(live.items()):
            if "ticket" in rec and (final or rec["due"] <= step):
                ns = nodes[rec["node"]]
                queue.answer(rec["ticket"], human_value(f"{key}|{i}", test[i], rec["node"], ns), reviewer="sim")
                rec.pop("ticket")
                take(i, procs[rec["proc"]].resume(rec["run"]), rec["proc"])

    last = time.time()
    for s0 in range(0, len(test), B):
        answer_due()
        for i in range(s0, min(len(test), s0 + B)):
            procs[i] = with_store(p0, os.path.join(unit_dir, "process", f"{i:05d}.jsonl"))
            take(i, procs[i].run(dict(test[i]["given"])), i)
            if time.time() - last > 20:
                status.set(i=i)
                last = time.time()
        step += 1
        status.set(i=min(len(test), s0 + B))
    snap = _tally(test, finished, nodes)
    pend_runs = [(rec["proc"], rec["run"]) for rec in live.values() if "ticket" in rec][:5]
    replay_pending = [procs[k].replay(run) for k, run in pend_runs]
    for _ in range(10):                                   # every open question answered, every run resumed
        if not live:
            break
        answer_due(final=True)
        step += 1
    final = _tally(test, finished, nodes)
    lost = len(live) + len(queue.open())
    rs = random.Random(7)
    sample = rs.sample(sorted(finished), min(20, len(finished)))
    rep = [procs[finished[i]["proc"]].replay(finished[i]["run"]) for i in sample]
    ok_replays = sum(1 for x in rep if x.get("ok"))
    out = {"arm_mode": ARMS[arm]["mode"], "at_end_of_stream": snap, "after_flush": final, "lost_tickets": lost,
            "tickets": tickets, "asks": asks, "ms_per_claim": 1000 * (time.time() - t0) / len(test),
            "guarantee": {k: g0[k] for k in ("mode", "risk", "in_force", "bound", "statement", "problems")},
            "path_watch": {"flagged": watch.flagged, "at_run": flag_at, "why": watch.why},
            "replay": {"sampled": len(rep), "ok": ok_replays,
                       "pending_sampled": len(replay_pending),
                       "pending_ok": sum(1 for x in replay_pending if x.get("ok") and x.get("pending"))}}
    if e2e is not None:                                   # stage 4 only: a stage-3 unit's JSON is unchanged
        out["e2e"] = e2e
    return out


def run_glued(systems, nodes, test, key, sizes, log, status):
    """The baseline: the same nodes called by a Python loop in the same order, run_context by hand, a person's answer
    after the same delay. It can name no bound for the end."""
    import uuid

    from solvi.core.store import run_context
    step, B = 0, sizes["step"]
    live, finished = {}, {}
    t0, asks, tickets = time.time(), 0, 0

    def nxt(node, f):
        if node == "t_kind":
            return "t_band"
        if node == "t_band":
            return "t_docs"
        if node == "t_docs":
            return check_of(f["kind"])
        if node in GROUPS:
            return f"p_{f['band']}"
        return None

    def go(i, rec):
        nonlocal asks, tickets
        f = rec["facts"]
        with run_context(run=rec["run"]):
            while rec["node"] is not None:
                node = rec["node"]
                ns = nodes[node]
                if rec.get("answer_in") is not None:          # a person's answer to this node
                    f[ns["question"]] = rec.pop("answer_in")
                    rec["human"] = True
                else:
                    with run_context(parent=rec["parent"]):
                        d = systems[node].ask({r: f[r] for r in _reads(node, ns)})
                    asks += 1
                    rec["parent"] = d.stored_id
                    if d.by == "human":
                        tickets += 1
                        rec["due"] = step + human_delay(f"{key}|{i}|{tickets}", node)
                        rec["waiting"] = True
                        return
                    fin = {x: f.get(x) for x in ("kind", "band", "docs", "fraud")}
                    rec["steps"].append((node, d.by, d.answer, fin))
                    f[ns["question"]] = d.answer
                if ns["layer"] == "payout":
                    rec.update(status="decided", answer=f["payout"], node=None)
                    finished[i] = live.pop(i)
                    return
                rec["node"] = nxt(node, f)

    def answer_due(final=False):
        for i, rec in list(live.items()):
            if rec.get("waiting") and (final or rec["due"] <= step):
                rec["waiting"] = False
                rec["answer_in"] = human_value(f"{key}|{i}", test[i], rec["node"], nodes[rec["node"]])
                go(i, rec)

    last = time.time()
    for s0 in range(0, len(test), B):
        answer_due()
        for i in range(s0, min(len(test), s0 + B)):
            live[i] = {"facts": dict(test[i]["given"]), "node": "t_kind", "parent": None, "steps": [], "human": False,
                       "run": uuid.uuid4().hex[:16]}
            go(i, live[i])
            if time.time() - last > 20:
                status.set(i=i)
                last = time.time()
        step += 1
        status.set(i=min(len(test), s0 + B))
    snap = _tally(test, finished, nodes)
    for _ in range(10):
        if not live:
            break
        answer_due(final=True)
        step += 1
    final = _tally(test, finished, nodes)
    return {"arm_mode": "baseline", "at_end_of_stream": snap, "after_flush": final, "lost_tickets": len(live),
            "tickets": tickets, "asks": asks, "ms_per_claim": 1000 * (time.time() - t0) / len(test),
            "guarantee": {"mode": None, "statement": "a loop glued by hand names no bound for the end"}}


# --- the run: units, resume, heartbeat, errors

def all_units(only_k=None, arms=None, seeds=range(10), conds=None):
    out = []
    for K in (10, 100):
        if only_k is not None and K != only_k:
            continue
        for cn, c in conditions(K).items():
            if conds and cn not in conds:
                continue
            for arm in ARMS:
                if arms and arm not in arms:
                    continue
                for s in seeds:
                    out.append((arm, cn, c, s))
    return out


def write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, one seed, two conditions, known-answer checks")
    ap.add_argument("--only", default=None, help="K=10 or K=100")
    ap.add_argument("--arms", default=None, help="comma-separated arm names (default: every arm of --stage)")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--stage", type=int, default=3, help="the stage whose arms run: 3 (union, per_node, glued) or "
                    "4 (end_to_end)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-units", type=int, default=None, help="stop after this many units (for a resume check)")
    args = ap.parse_args(argv)
    only_k = int(args.only.split("=")[1]) if args.only else None
    lo, _, hi = args.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS) if ARMS[a]["stage"] == args.stage]
    conds = args.conditions.split(",") if args.conditions else None
    sizes = SIZES
    if args.smoke:
        sizes, seeds, only_k = SMOKE_SIZES, range(0, 1), 10
        conds = conds or ["K10_rho0_a", "K10_rho5_b"]
    for d in ("units", "logs", "errors", "work"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    units = all_units(only_k, arms, seeds, conds)
    if args.report:
        return report(args.out)
    if args.list:
        for arm, cn, _, s in units:
            u = unit_id(arm, cn, s)
            print(u, "done" if os.path.exists(os.path.join(args.out, "units", f"{u}.json")) else "todo")
        return 0
    main_log = open(os.path.join(args.out, "logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"), "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        main_log.write(line + "\n")
        main_log.flush()
    status = Status()
    stop = threading.Event()

    def beat():
        while not stop.wait(30):
            d = status.get()
            say(f"heartbeat unit={d['unit']} phase={d['phase']} {d['i']}/{d['n']}")
    threading.Thread(target=beat, daemon=True).start()
    say(f"E1 harness: {len(units)} units, sizes {sizes}, out {args.out}")
    done = skipped = failed = ran = 0
    times = []
    for j, (arm, cn, c, s) in enumerate(units):
        u = unit_id(arm, cn, s)
        res_path = os.path.join(args.out, "units", f"{u}.json")
        if os.path.exists(res_path):
            done += 1
            continue
        if args.max_units is not None and ran >= args.max_units:
            break
        wd = os.path.join(args.out, "work", u)
        shutil.rmtree(wd, ignore_errors=True)               # an unfinished unit starts again from scratch
        os.makedirs(wd)
        ulog = open(os.path.join(args.out, "logs", f"{u}.log"), "w")

        def log(msg, ulog=ulog):
            ulog.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            ulog.flush()
        status.set(unit=u, phase="start", i=0, n=0)
        say(f"unit {j + 1}/{len(units)} {u}")
        t = time.time()
        try:
            out = run_unit(arm, cn, c, s, sizes, wd, log, status)
            bad = [k for k, v in out["after_flush"].items() if isinstance(v, float) and not math.isfinite(v)]
            if bad:
                raise FloatingPointError(f"non-finite metrics {bad}")
            out["wall_s"] = time.time() - t
            write_json(res_path, out)
            times.append(out["wall_s"])
            ran += 1
            m = out["after_flush"]
            say(f"done {u} in {out['wall_s']:.1f}s: coverage {m['coverage']:.3f} e_hat {m['e_hat']:.4f} "
                f"human {m['human_share']:.3f} lost {out['lost_tickets']}")
            log(f"done: {json.dumps(m)[:2000]}")
        except Skip as e:
            skipped += 1
            log(f"skip: {e}")
            say(f"skip {u}: {e}")
        except Exception as e:  # noqa: BLE001 — NaN, divergence, a bug: logged, skipped, the run goes on
            failed += 1
            tb = traceback.format_exc()
            log(f"error: {tb}")
            say(f"error {u}: {type(e).__name__}: {e}")
            write_json(os.path.join(args.out, "errors", f"{u}.json"), {"unit": u, "error": repr(e), "traceback": tb})
        finally:
            ulog.close()
            shutil.rmtree(wd, ignore_errors=True) if not args.smoke else None
    stop.set()
    say(f"finished: ran {ran}, already done {done}, skipped {skipped}, failed {failed}; "
        f"mean unit {np.mean(times) if times else float('nan'):.1f}s")
    if args.smoke:
        smoke_checks(args.out, units, say)
    main_log.close()
    return 0


# --- smoke: known answers

def smoke_checks(out, units, say):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        say(f"smoke check {'OK  ' if ok else 'FAIL'} {name} {detail}")
    # the generator: the arg-max of x is right with the node's accuracy
    rng = np.random.default_rng(99)
    nodes = nodes_k10()
    for rho in (0.0, 0.5):
        cond = {"rho": rho, "regions": 4, "noise": 0.0}
        cs = gen_cases(rng, 4000, cond, nodes)
        for name in ("t_kind", "c_fraud_34"):
            ns = nodes[name]
            acc = np.mean([ns["options"][int(np.argmax(c["given"][f"obs_{name}"]))] == c["truth"][ns["question"]]
                           for c in cs])
            check(f"accuracy {name} rho {rho}", abs(acc - ns["acc"]) < 0.03, f"{acc:.3f} vs {ns['acc']}")
        cf = gen_cases(rng, 4000, cond, nodes, faulty=True)
        acc = np.mean([YESNO[int(np.argmax(c["given"][f"obs_{FAULTY}"]))] == c["truth"]["fraud"] for c in cf])
        check(f"faulty expert at 0.5 rho {rho}", abs(acc - 0.5) < 0.03, f"{acc:.3f}")
    # correlated errors at rho 0.5, not at 0
    for rho, want in ((0.0, False), (0.5, True)):
        cs = gen_cases(np.random.default_rng(5), 6000, {"rho": rho, "regions": 4, "noise": 0.0}, nodes)
        w1 = np.array([KINDS[int(np.argmax(c["given"]["obs_t_kind"]))] != c["truth"]["kind"] for c in cs])
        w2 = np.array([BANDS[int(np.argmax(c["given"]["obs_t_band"]))] != c["truth"]["band"] for c in cs])
        corr = float(np.corrcoef(w1, w2)[0, 1])
        check(f"error correlation rho {rho}", (corr > 0.1) == want, f"{corr:.3f}")
    # the end rule and the label noise
    check("end rule", payout_of("k0", "low", "yes", "yes") == "deny" and payout_of("k3", "high", "yes", "no") ==
          "investigate" and payout_of("k1", "high", "yes", "no") == "pay" and payout_of("k1", "low", "no", "no") ==
          "investigate")
    cs = gen_cases(np.random.default_rng(3), 4000, {"rho": 0.0, "regions": 4, "noise": 0.05}, nodes)
    fl = np.mean([c["end"] != c["end_label"] for c in cs])
    check("label noise 0.05", abs(fl - 0.05) < 0.012, f"{fl:.4f}")
    # the end_to_end families: grids monotone, the fixed sequence moves one node per step from the end back
    from solvi.experimental.process import PerNodeFixed, SharedRank
    sig = {n: list(np.random.default_rng(11).random(200)) for n in nodes}
    grid = SharedRank().grid({}, sig)
    mono = all([t[n] for t in grid] == sorted(t[n] for t in grid) for n in nodes)
    let = [sum(SharedRank().automatic(n, x, t) for n in nodes for x in sig[n]) for t in grid]
    check("shared_rank grid monotone (thresholds up, answered alone down)", mono and let == sorted(let, reverse=True)
          and let[0] == 200 * len(nodes), f"{let[0]} … {let[-1]}")
    order = list(reversed(list(nodes)))
    chain = PerNodeFixed().grid({"promise": {"family": {"kind": "per_node_fixed", "order": order}}}, sig)
    one = all(sum(x[k] != y[k] for k in nodes) <= 1 for x, y in zip(chain, chain[1:]))
    check("per_node_fixed chain: one node per step, end nodes first", one and len(chain) == 64 * len(nodes)
          and chain[0][order[0]] is not None and all(chain[0][k] is None for k in order[1:]), f"{len(chain)} steps")
    # the units: written, sane, the promise as declared
    for arm, cn, _, s in units:
        u = unit_id(arm, cn, s)
        pth = os.path.join(out, "units", f"{u}.json")
        if not os.path.exists(pth):
            check(f"unit {u} written", False)
            continue
        r = json.load(open(pth))
        m = r["after_flush"]
        check(f"{u}: shares add up", m["auto"] + m["human"] + m["other"] + m["pending"] == m["n"] and m["pending"] == 0,
              f"auto {m['auto']} human {m['human']} other {m['other']} pending {m['pending']}")
        check(f"{u}: no lost ticket", r["lost_tickets"] == 0, str(r["lost_tickets"]))
        if r["arm_mode"] == "union":
            g = r["guarantee"]
            check(f"{u}: union in force at bound ≤ r", g["in_force"] and g["bound"] <= R_PROCESS + 1e-12,
                  f"bound {g['bound']}, problems {g['problems']}")
            check(f"{u}: replay ok", r["replay"]["ok"] == r["replay"]["sampled"] and
                  r["replay"]["pending_ok"] == r["replay"]["pending_sampled"], json.dumps(r["replay"]))
            want = {n: R_PROCESS / 10 for n in nodes} if r["arm"] == "union_equal" else None
            if want:
                check(f"{u}: r_i = r/10", all(abs(r["r_i"][n] - want[n]) < 1e-12 for n in nodes))
        if r["arm_mode"] == "per_node":
            check(f"{u}: per_node says not bounded", "end error is not bounded" in r["guarantee"]["statement"])
        if r["arm_mode"] == "end_to_end":
            g, e, a = r["guarantee"], r["e2e"], ARMS[r["arm"]]
            th = e["thresholds"]
            check(f"{u}: end_to_end calibrated and in force", g["in_force"] and g["bound"] == R_PROCESS,
                  f"problems {g['problems']}")
            check(f"{u}: n and family as the arm", e["n_end"] == a["n_end"] == th["n"] and e["family"]["kind"] ==
                  a["family"] and th["procedure"] == ("bonferroni" if a["family"] == "shared_rank" else "fixed"),
                  f"n {e['n_end']} {th['procedure']}")
            check(f"{u}: certified threshold has calibration loss ≤ r", th["t"] is None or
                  th["wrong"] / th["n"] <= R_PROCESS, f"{th['wrong']}/{th['n']}, automatic {th['automatic']}")
            tol = 3 * math.sqrt(R_PROCESS * (1 - R_PROCESS) / m["n"])    # the promise holds w.p. 1 − δ, on a small test
            check(f"{u}: e_hat ≤ r + 3 sd of a {m['n']}-claim test", m["e_hat"] <= R_PROCESS + tol,
                  f"{m['e_hat']:.4f} (≤ r: {m['e_hat'] <= R_PROCESS}), coverage {m['coverage']:.3f}")
            check(f"{u}: replay ok", r["replay"]["ok"] == r["replay"]["sampled"] and
                  r["replay"]["pending_ok"] == r["replay"]["pending_sampled"], json.dumps(r["replay"]))
        m0 = r["at_end_of_stream"]
        check(f"{u}: e_hat counts the automatic share only", m["e_with_human"] >= m["e_hat"]
              and m0["pending"] + m0["auto"] + m0["human"] + m0["other"] == m0["n"], "")
    # process against the hand-glued loop: the same nodes, the same coverage
    for cn in {u[1] for u in units}:
        a = os.path.join(out, "units", f"{unit_id('union_equal', cn, 0)}.json")
        b = os.path.join(out, "units", f"{unit_id('glued', cn, 0)}.json")
        if os.path.exists(a) and os.path.exists(b):
            ra, rb = json.load(open(a))["after_flush"], json.load(open(b))["after_flush"]
            check(f"{cn}: process = glued (coverage, errors)", ra["auto"] == rb["auto"] and
                  ra["wrong_auto"] == rb["wrong_auto"], f"{ra['auto']}/{rb['auto']} {ra['wrong_auto']}/{rb['wrong_auto']}")
    write_json(os.path.join(out, "smoke_checks.json"), checks)
    say(f"smoke: {sum(c['ok'] for c in checks)}/{len(checks)} checks ok")


# --- the report: bars B1, B3, B5 (stage 3) and B2, B6 (stage 4) over the finished units

def report(out):
    rows = [json.load(open(os.path.join(out, "units", f))) for f in sorted(os.listdir(os.path.join(out, "units")))
            if f.endswith(".json")]
    by = {}
    for r in rows:
        by.setdefault((r["arm"], r["condition"]), []).append(r)
    print(f"{'arm':14s} {'condition':24s} seeds  e_hat≤r   coverage  e_hat(mean)  human  pending@end  lost")
    for (arm, cn), rs in sorted(by.items()):
        m = [r["after_flush"] for r in rs]
        ok = sum(x["e_hat"] <= R_PROCESS for x in m)
        print(f"{arm:14s} {cn:24s} {len(rs):5d}  {ok:2d}/{len(rs):<4d} {np.mean([x['coverage'] for x in m]):8.3f}  "
              f"{np.mean([x['e_hat'] for x in m]):10.4f}  {np.mean([x['human_share'] for x in m]):5.3f}  "
              f"{np.mean([r['at_end_of_stream']['pending_share'] for r in rs]):10.3f}  "
              f"{sum(r['lost_tickets'] for r in rs)}")
    cov = {k: [r["after_flush"]["coverage"] for r in rs] for k, rs in by.items()}
    conds = sorted({cn for _, cn in by})
    e2e = [a for a in ARMS if ARMS[a]["mode"] == "end_to_end"]
    if any((a, cn) in by for a in e2e for cn in conds):
        print("\nB2: coverage(end_to_end, shared_rank) / coverage(union, equal), means over seeds (bar ≥ 1.5)")
        for cn in conds:
            base = np.mean(cov[("union_equal", cn)]) if ("union_equal", cn) in cov else float("nan")
            parts = []
            for n in (500, 1000, 3000):
                k = (f"e2e_shared_rank_n{n}", cn)
                if k in cov:
                    parts.append(f"n{n} {np.mean(cov[k]):.3f}/{base:.3f} = {np.mean(cov[k]) / base:.2f}×")
            if parts:
                print(f"  {cn:24s} " + "   ".join(parts))
        print("\nB6: per_node_fixed — valid (e_hat ≤ r in ≥ 9/10 seeds) and coverage / shared_rank (≥ 1.2× at n=3000 "
              "goes to the author)")
        for cn in conds:
            parts = []
            for n in (500, 1000, 3000):
                kf, ks = (f"e2e_per_node_fixed_n{n}", cn), (f"e2e_shared_rank_n{n}", cn)
                if kf in by:
                    ok = sum(r["after_flush"]["e_hat"] <= R_PROCESS for r in by[kf])
                    ratio = (np.mean(cov[kf]) / np.mean(cov[ks])) if ks in cov else float("nan")
                    parts.append(f"n{n} valid {ok}/{len(by[kf])} cov {np.mean(cov[kf]):.3f} ({ratio:.2f}×)")
            if parts:
                print(f"  {cn:24s} " + "   ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
