"""solvi.experimental.stream, additive coverage: the guarantee measured on synthetic streams with known per-zone risk
(spec test 1). Each scenario runs an AdapterLoop(additive=True) over many seeds and checks what the documentation
promises:

(a) a candidate whose true risk on its zone is at least α at every grid point is certified in at most δ_v of runs
    (read as the one-sided binomial 95 % upper bound of δ_v·runs: Ville's inequality bounds each index's e-process);
(b) across every index a run spends — the base, a bad candidate, a restart of the base and a second bad candidate —
    the runs with any false certification are at most δ (the same binomial reading);
(c) a candidate with a true zone risk below α certifies within the documented sample counts (×1.5);
(d) with the joint kind (max_risk), the realised joint risk inside each certified zone stays ≤ α and the stream-wide
    composite stays ≤ α·(1 + Σ_{j≥1} P(Z_j)) — the multiplier the documentation states.

The default suite runs two seeds per scenario (about 15 s: the machinery and the bounds, no statistical power);
`-m slow` runs the spec's 400 seeds per scenario in SOLVI_VALIDITY_WORKERS processes (default 4; minutes, not
seconds)."""
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy
    from solvi.experimental.stream._eprocess import DeltaBudget, clean_need

KEY = b"additive-validity-key"


class _M:
    """A frozen model: x → f(x) = (would, score)."""

    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


class _Hand:
    """A batch learner handing out the given models in turn (the last one again when they run out)."""

    def __init__(self, models):
        self.models, self._seen = list(models), 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return self.models.pop(0) if len(self.models) > 1 else self.models[0]


# x = (label, u, w, e0, e1): u is the base's score, w the candidates', e0 / e1 independent coins for their errors.
# The base L0 is right iff u > 0.5 (its zone below the frozen Λ_0 ≈ 0.5 is half the stream); its selective error at a
# threshold λ < 0.5 is (0.5 − λ)/(1 − λ), at λ ≥ 0.5 zero.
def _base_wrong_at(lam):                                   # L0's true selective error at threshold lam
    return max(0.0, 0.5 - lam) / (1.0 - lam)


L0 = _M("l0", lambda x: (x[0] if x[1] > 0.5 else "wrong", x[1]))


def _flip(label):
    return "b" if label == "a" else "a"


def _cand(name, p):
    """A candidate wrong with probability p at every score (its true selective error is p at every grid point)."""
    return _M(name, lambda x: (_flip(x[0]) if x[4] < p else x[0], x[2]))


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random(), r.random(), r.random()) for _ in range(n)]


def _policy(**kw):
    kw.setdefault("mature_patience", None)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False,
                         probe=False, **kw)


def _loop(models, *, policy, **kw):
    return AdapterLoop(_Hand(models), question="q", policy=policy, secret=KEY, initial=L0, additive=True,
                       monitor=None, _detector_alone=True, wait=True, clock=lambda: 0.0, **kw)


def _certs(loop, v):
    """Version v's promise's certification events (its threshold went finite or loosened)."""
    p = loop._vs[v]["promise"]
    return [e for e in p.events if e["event"] in ("certified", "loosened")]


def _binom_upper(p, n):
    """One-sided 95 % upper bound on a Binomial(n, p) count (normal approximation + 1, conservative at small n·p)."""
    return n * p + 1.645 * math.sqrt(n * p * (1 - p)) + 1


# ───────────────────────────── (a) + (b): bad candidates at α, a restart of the base, one δ over every index
GRID = [round(1 - 0.05 * i, 2) for i in range(1, 20)]   # 0.95 … 0.05, fixed (the selective kind's auto grid starts
#                                                         at a point so strict that almost nothing is alone there)
ALPHA_SEL, DELTA_SEL = 0.10, 0.30                         # a large δ, so that δ_v is large enough to be measured


def _run_bad(seed, n=6000):
    """Selective kind (max_error = α): L0 certifies (honestly, unless the certificate errs); bad candidates — error
    exactly α at every grid point, the null's worst case — are attached one after another and tried until stale;
    L0 is restarted at a fixed decision (its next index is taken at its first bet after it).
    → {"indices": [(version, v, δ_v, falsely certified)], "spent": Σ δ_v, "any_false": bool}."""
    pol = _policy(min_novel=10 ** 9, stale=1500, trial=1500)   # the gate never passes: every candidate stays in trial
    loop = _loop([_cand("bad", ALPHA_SEL)], policy=pol, max_error=ALPHA_SEL, delta=DELTA_SEL,
                 promise_options={"grid": GRID})
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
        if loop.n == 3000:
            loop._restart(0, loop.n, "test: a restart at a fixed decision")   # what an alarm's "restart" runs
    rows = []
    for v, ver in loop._vs.items():
        p = ver["promise"]
        if p is None:
            continue
        idx = {e["v_index"]: False for e in p.events if e.get("v_index") is not None}
        if p.v is not None:
            idx.setdefault(p.v, False)
        for e in p.events:
            if e["event"] in ("certified", "loosened"):
                # false: the certified threshold's true selective error is ≥ α — any point of a bad candidate; for
                # L0 a point where (0.5 − λ)/(1 − λ) ≥ α
                idx[e["v_index"]] |= v != 0 or _base_wrong_at(e["threshold"]) >= ALPHA_SEL
        rows += [(v, i, loop.budget.delta_at(i), f) for i, f in idx.items()]
    assert sorted(i for _, i, _, _ in rows) == list(range(1, loop.budget.v + 1))   # every index charged is counted
    return {"indices": rows, "spent": loop.budget.spent, "any_false": any(f for *_, f in rows),
            "restarted": any(e["event"] == "recertify" for e in loop._vs[0]["promise"].events),
            "candidates": sum(1 for v in loop._vs if v != 0)}


def _check_bad(runs):
    n = len(runs)
    assert all(r["restarted"] and r["candidates"] >= 2 for r in runs)          # both layers and one restart, each run
    assert all(r["spent"] <= DELTA_SEL + 1e-12 for r in runs)
    by_index = {}                                          # (a): per index of a bad candidate, its false share
    for r in runs:
        for v, i, d, f in r["indices"]:
            if v != 0:
                k = by_index.setdefault(i, [d, 0, 0])
                k[1] += 1
                k[2] += f
    for i, (d, m, false) in by_index.items():
        assert false <= _binom_upper(d, m), (i, d, m, false)
    any_false = sum(r["any_false"] for r in runs)          # (b): any false certification in a run, over every index
    assert any_false <= _binom_upper(DELTA_SEL, n), (any_false, n)
    return {"runs": n, "per_index": {i: {"delta_v": round(d, 4), "runs": m, "false": f}
                                     for i, (d, m, f) in sorted(by_index.items())},
            "any_false": any_false, "delta": DELTA_SEL}


# ───────────────────────────── (c): a good candidate certifies within the documented sample counts
# coin samples to certify a point of true joint risk p ≤ 0.01 at α = δ = 0.05 (lane Mixture bet), by index v
NEED_P01 = {2: 166, 3: 209, 4: 222}


def _run_good(seed, n=4000):
    """Joint kind, α = δ = 0.05: L0 right iff u > 0.5; the candidate is wrong with probability 0.01 at every score.
    → (its index, zone labels its promise read when it certified, its threshold) or None if it never certified."""
    loop = _loop([_cand("good", 0.01)], policy=_policy(), max_risk=0.05, delta=0.05)
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
        if 1 in loop._vs and math.isfinite(loop._threshold(1)):
            break
    p = loop._vs[1]["promise"] if 1 in loop._vs else None
    cert = None if p is None else next((e for e in p.events if e["event"] == "certified"), None)
    return None if cert is None else (cert["v_index"], cert["n"])


def _check_good(runs):
    assert all(r is not None for r in runs)                # every run's candidate certified
    for v, k in runs:
        assert k <= 1.5 * NEED_P01[v], (v, k)
        assert k >= clean_need(0.05, DeltaBudget(0.05).delta_at(v))   # never faster than an all-clean stream
    ks = sorted(k for _, k in runs)
    return {"runs": len(runs), "index": sorted({v for v, _ in runs}), "samples_median": ks[len(ks) // 2],
            "samples_max": ks[-1], "bound": {v: 1.5 * NEED_P01[v] for v in sorted({v for v, _ in runs})}}


# ───────────────────────────── (d): per-zone joint risk ≤ α, stream-wide joint ≤ α·(1 + Σ P(Z_j))
ALPHA_J = 0.05
# L0: right iff u > 0.5, and wrong with probability 0.08 above it — its joint risk at λ is 0.08·(1 − λ) + max(0,
# 0.5 − λ): matured (the default gate in additive mode) it certifies near λ ≈ 0.49, close to α. Layer 1 is wrong with probability 0.03 at every score. The
# stream-wide joint risk is then above α (≈ 0.045 + 0.5·0.03) and below α·(1 + P(Z_1)).
L0_NOISY = _M("l0n", lambda x: (x[0] if x[1] > 0.5 and x[3] >= 0.08 else _flip(x[0]), x[1]))


def _run_joint(seed, n=9000):
    """→ per-zone counts after layer 1 began to serve: zone decisions, alone-and-wrong; and the stream's."""
    loop = AdapterLoop(_Hand([_cand("l1", 0.03)]), question="q", policy=_policy(stale=6000, trial=6000, mature_patience="auto"),
                       secret=KEY, initial=L0_NOISY, additive=True, monitor=None, _detector_alone=True, wait=True,
                       clock=lambda: 0.0, max_risk=ALPHA_J, delta=0.05, max_layers=2)
    serving = None
    z = {0: [0, 0], 1: [0, 0]}                            # layer j: [decisions in Z_j, alone and wrong by j]
    for x in _xs(seed, n):
        out = loop.step(x, lambda x=x: x[0])
        if serving is None:
            if len(loop._layers) == 2:
                serving = loop.n                           # layer 1 owns its zone from the next decision
            continue
        lay = out["layer"]
        wrong = out["by"] == "local" and out["answer"] != x[0]
        for j in range(lay + 1):
            z[j][0] += 1
        z[lay][1] += wrong
    share = None if serving is None else z[1][0] / max(1, z[0][0])
    return {"serving_at": serving, "zones": z, "share": share}


def _check_joint(runs):
    on = [r for r in runs if r["serving_at"] is not None]
    assert len(on) >= 0.9 * len(runs)                      # layer 1 served in (almost) every run
    tot = {j: [sum(r["zones"][j][0] for r in on), sum(r["zones"][j][1] for r in on)] for j in (0, 1)}
    risk = {j: tot[j][1] / tot[j][0] for j in (0, 1)}
    share = tot[1][0] / tot[0][0]
    stream = (tot[0][1] + tot[1][1]) / tot[0][0]
    assert risk[0] <= ALPHA_J and risk[1] <= ALPHA_J, risk                 # each certified zone: ≤ α
    assert stream <= ALPHA_J * (1 + share), (stream, share)               # the documented joint multiplier
    return {"runs": len(runs), "layered": len(on), "risk_zone": {j: round(x, 4) for j, x in risk.items()},
            "zone_share": round(share, 3), "stream_joint": round(stream, 4),
            "bound": round(ALPHA_J * (1 + share), 4)}


# ───────────────────────────── the runs
def _map(fn, seeds):
    """Sequential for a few seeds; the 400-seed runs in SOLVI_VALIDITY_WORKERS processes (default 4)."""
    import os
    workers = 1 if len(seeds) < 50 else int(os.environ.get("SOLVI_VALIDITY_WORKERS", "4"))
    if workers <= 1:
        return [fn(s) for s in seeds]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, seeds, chunksize=4))


FAST = range(2)                                            # the default suite: the machinery and the bounds (~15 s)
FULL = range(1000, 1400)                                   # -m slow: the spec's 400 seeds per scenario (minutes)
SEEDS = [pytest.param(FAST, id="fast"), pytest.param(FULL, id="400", marks=pytest.mark.slow)]


@pytest.mark.parametrize("seeds", SEEDS)
def test_bad_candidates_are_falsely_certified_at_most_delta_v_and_delta_over_every_index(seeds):
    print("\n(a)+(b)", _check_bad(_map(_run_bad, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_a_good_candidate_certifies_within_the_documented_sample_counts(seeds):
    print("\n(c)", _check_good(_map(_run_good, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_joint_risk_is_at_most_alpha_per_zone_and_within_the_multiplier_stream_wide(seeds):
    print("\n(d)", _check_joint(_map(_run_joint, seeds)))
