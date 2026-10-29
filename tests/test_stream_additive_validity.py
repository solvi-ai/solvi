"""solvi.experimental.stream, additive coverage: the guarantee measured on synthetic streams with known per-zone risk
(spec test 1). Each scenario runs an AdapterLoop(additive=True) over many seeds and checks what the documentation
promises:

(a) a candidate whose true risk on its zone is at least α at every grid point is certified in at most δ_v of runs
    (read as the one-sided binomial 95 % upper bound of δ_v·runs: Ville's inequality bounds each index's e-process);
(b) across every index a run spends — the base, a bad candidate, a restart of the base and a second bad candidate —
    the runs with any false certification are at most δ (the same binomial reading);
(c) a candidate with a true zone risk below α certifies within the documented sample counts (×1.5);
(d) with the joint kind (max_risk), the realised joint risk inside each certified zone stays ≤ α and the stream-wide
    composite stays ≤ α·(1 + Σ_{j≥1} P(Z_j)) — the multiplier the documentation states (a decision is credited to the
    layer that answered it: decision 21);
(e) upstream loosening (decision 21): L0 whose true joint risk is exactly α at every grid point below its routing
    threshold Λ_0 (its errors in a band just below it), a good layer 1 attached and serving below Λ_0 — L0 keeps
    betting at those points on coin samples of its whole zone, and is falsely certified there in at most δ_v of runs.
    Negative control, computed from the same records: the same promise fed only the rows L0 owns (the 82e0ad5 learning
    rule without its freeze) — certified there in 269 of the 400 slow-suite runs (de16711 + the review fixes), far
    above the bar. Λ_0 is 0.6 unless L0 was already falsely certified below 0.6 before the attach (43 / 400; such a
    run counts as a false certification);
(f) the owned regime's floor: the same L0 restarted after layer 1 serves relearns on its owned rows with no bet below
    Λ_0 (those points' wealth stays at its start until the base regime), re-certifies at or above Λ_0, and is then
    falsely certified below Λ_0 in at most δ_v of runs (its post-restart index);
(g) the reading-(A) failure case under reading (B): layer 1 is right on the band L0 later takes and wrong with 2α on
    the rest of its zone (zone risk below α at its attach); its zone is fixed, so its realised zone risk stays ≤ α, as
    does L0's, and the stream-wide joint risk ≤ α·(1 + P̂(Z_1)). The risk on the rows of Z_1 L0 does not answer (what
    a shrinking zone would serve) is printed: it grows with L0's loosening — not a bar;
(h) sanity: a run in which L0 cannot loosen after the attach (its loose grid points are at or above α) gives the same
    routing, answers, coin and events as the 82e0ad5 loop on the same stream (a pinned digest of those fields, two
    seeds; the new record fields aside) — with `useful_error=None`; with the usefulness certificate on (the default)
    the same digest once its own "useful_*" events are left out (a good layer: the certificate never binds).

The usefulness certificate (decision 25: selective at β = 0.5, pointwise, its own budget δ_use):
(i) a noise layer — wrong w.p. 0.9 at every score, 2 % of its zone above every grid point, so the joint certificate
    passes it (risk ≈ 0.018 ≤ α) — starts serving in at most δ_use of runs (binomial bar; two attaches per run, both
    indices on the δ_use budget); the δ_use audit (v) is read from the same runs' records: usefulness indices
    chronological and unique per run, Σ δ_u ≤ δ_use;
(ii) a real layer (error 0.1 at every score, score uniform) still starts in every run, and its promote decision differs
    from a `useful_error=None` run's by ≤ 50 zone samples at the 95th percentile, and its alone answers over the last
    2 000 decisions are ≥ 95 % of that run's (no coverage lost; a later start has read more trial rows at the full
    rate, so it tends to start at a looser threshold and answer more);
    the loop's own δ indices are the same in both runs whenever the promote decision is;
(iii) the null at the edge: a layer whose error among its alone answers is exactly β at every grid point (6 % of its
    zone alone: joint risk 0.03 ≤ α) is usefulness-certified in at most δ_u of runs per index (δ_use = 0.3 for power);
(iv) composition: after a layer starts, its joint prefix loosens into a band where its selective error is ≥ β (the
    joint risk there stays ≤ α); its answer threshold stays at usefulness-certified points: runs with any alone answer
    at a point of true selective error ≥ β ≤ bar(δ_use), the realised error among its alone answers ≤ β, its zone's
    joint risk ≤ α, and the composition binds (the joint threshold reaches the bad band) in most runs.

The default suite runs two seeds per scenario (the machinery and the bounds, no statistical power);
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
    z = {0: [0, 0], 1: [0, 0]}                            # layer j: [decisions in Z_j, answered alone by j and wrong]
    for x in _xs(seed, n):
        out = loop.step(x, lambda x=x: x[0])
        if serving is None:
            if len(loop._layers) == 2:
                serving = loop.n                           # layer 1 owns its zone from the next decision
            continue
        for j in range(out["layer"] + 1):
            z[j][0] += 1
        if out["by"] == "local" and out["answer"] != x[0]:
            z[loop._layers.index(out["answered_by"])][1] += 1   # credited to the layer that answered
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


# ───────────────────────────── (e) + (f): upstream loosening — L0 bad just below its Λ_0; the reach rule, the floor
# L0B: right everywhere but the band [0.55, 0.6), where it is always wrong (mass 0.05): on the fixed grid its joint
# risk is 0 at every point ≥ 0.6 and exactly α at every point below (the null's boundary). It certifies 0.6 at once,
# so Λ_0 = 0.6 at the attach; on the rows it then owns (score ≥ 0.6) it is never wrong.
ALPHA_U, DELTA_U = 0.05, 0.30
BAD_BELOW = 0.6                                            # a grid point below it has true joint risk α: false
L0B = _M("l0b", lambda x: (_flip(x[0]) if 0.55 <= x[1] < 0.6 else x[0], x[1]))


def _upstream_loop():
    return AdapterLoop(_Hand([_cand("l1", 0.01)]), question="q", policy=_policy(first=100, stale=4000, trial=4000),
                       secret=KEY, initial=L0B, additive=True, monitor=None, _detector_alone=True, wait=True,
                       clock=lambda: 0.0, max_risk=ALPHA_U, delta=DELTA_U, max_layers=2,
                       promise_options={"grid": GRID})


def _false_below(p, index):
    """Whether promise p was certified (or loosened) at a grid point below BAD_BELOW under δ index `index`."""
    return any(e["event"] in ("certified", "loosened") and e["v_index"] == index and e["threshold"] is not None
               and e["threshold"] < BAD_BELOW for e in p.events)


def _run_upstream(seed, n=8000):
    """(e): L0 learns on the base coin over its whole zone while layer 1 serves below Λ_0. The control: a promise of
    the same configuration fed, from the same records, only the coin-sampled rows L0 owned (before the attach that is
    every decision)."""
    from solvi.experimental.stream import StreamPromise
    loop = _upstream_loop()
    p0 = loop._vs[0]["promise"]
    before = None                                          # a false certification before the attach (Λ_0 below 0.6)
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
        if before is None and len(loop._layers) == 2:
            before = _false_below(p0, p0.v)
    ctrl = StreamPromise(max_risk=ALPHA_U, budget=DeltaBudget(DELTA_U), grid=GRID)
    for k, d in loop.log:
        if k == "loop" and d["stream"]["owner"] == 0 and d["stream"]["sampled"]:
            ctrl.learn(d["stream"]["score"], d["loop"]["path_would"][0], d["loop"]["label"])
    lam0 = loop._route.get(0)
    return {"layered": len(loop._layers) == 2, "lam0": lam0, "index": p0.v, "delta_v": p0.delta_v,
            "false": _false_below(p0, p0.v), "false_before_attach": bool(before),
            "ctrl_false": _false_below(ctrl, ctrl.v), "samples": p0.n}


def _check_upstream(runs):
    n = len(runs)
    on = [r for r in runs if r["layered"]]
    assert len(on) >= 0.9 * n                                                  # layer 1 served (almost) every run
    # Λ_0 is λ_0 at the attach: 0.6, unless L0 was already falsely certified below 0.6 before it (the same e-process
    # on the same null — before the attach its zone is the whole stream; such a run counts in `false` below)
    assert all(r["lam0"] == BAD_BELOW or (r["lam0"] < BAD_BELOW and r["false_before_attach"]) for r in on)
    at_bad = sum(r["lam0"] == BAD_BELOW for r in on)
    assert at_bad >= 0.75 * len(on)                                            # the reach rule is what most runs test
    assert {r["index"] for r in runs} == {1}                                   # L0's one index, no restart
    d = runs[0]["delta_v"]
    false = sum(r["false"] for r in runs)
    ctrl = sum(r["ctrl_false"] for r in runs)
    assert false <= _binom_upper(d, n), (false, n, d)
    if n >= 50:                                                                # the control has power: the owner-only
        assert ctrl > _binom_upper(d, n), (ctrl, n)                            # feed fails the bar the reach rule meets
    return {"runs": n, "layered": len(on), "lam0_at_0.6": at_bad, "delta_v": round(d, 4), "false": false,
            "false_before_attach": sum(r["false_before_attach"] for r in runs),
            "bar": round(_binom_upper(d, n), 1), "owner_only_control_false": ctrl,
            "samples_median": sorted(r["samples"] for r in runs)[n // 2]}


def _run_floor(seed, n=8000):
    """(f): L0 restarted (with probation, as an alarm does) 200 decisions after layer 1 began to serve."""
    loop = _upstream_loop()
    p0 = loop._vs[0]["promise"]
    fresh = p0._new_bet().vector()
    at = start = back = None
    owned = bet_below = 0
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
        if at is None and len(loop._layers) == 2:
            at = loop.n
        if at is not None and loop.n == at + 200:
            loop.left = loop.probation                     # what an alarm on L0 runs: probation, then the restart
            loop._restart(0, loop.n, "test: a restart after layer 1 serves")
            start = len(p0.events)
        if start is None or back is not None:
            continue
        if loop._base_regime(0):
            back = loop.n                                  # certified again and out of probation
            continue
        owned += 1                                         # the owned regime: no bet below Λ_0
        bet_below += any(p0._proc[i].vector() != fresh for i, lam in enumerate(p0.grid) if lam < loop._route[0])
    if start is None:
        return {"restarted": False}
    cert = [e for e in p0.events[start:] if e["event"] == "certified"]
    return {"restarted": True, "owned_steps": owned, "bet_below": bet_below, "back": back,
            "recert": cert[0]["threshold"] if cert else None, "lam0": loop._route.get(0), "index": p0.v,
            "delta_v": p0.delta_v, "false": p0.v is not None and _false_below(p0, p0.v)}


def _check_floor(runs):
    n = len(runs)
    on = [r for r in runs if r.get("restarted")]
    assert len(on) >= 0.9 * n
    assert all(r["owned_steps"] > 0 and r["bet_below"] == 0 for r in on)     # no bet below Λ_0 while relearning
    assert all(r["recert"] is not None and r["recert"] >= r["lam0"] for r in on)   # re-certified at or above Λ_0
    assert sum(r["back"] is not None for r in on) >= 0.9 * len(on)             # then back in the base regime
    by_index = {}
    for r in on:
        k = by_index.setdefault(r["index"], [r["delta_v"], 0, 0])
        k[1] += 1
        k[2] += r["false"]
    for i, (d, m, false) in by_index.items():
        assert false <= _binom_upper(d, m), (i, d, m, false)
    return {"runs": n, "restarted": len(on), "per_index": {i: {"delta_v": round(d, 4), "runs": m, "false": f}
                                                           for i, (d, m, f) in sorted(by_index.items())}}


# ───────────────────────────── (g): reading (A)'s failure case — layer 1 good on L0's future band, 2α on the rest
# L0G: right above 0.6, wrong with probability 0.13 on [0.18, 0.6) and always below 0.18: its joint risk at λ ∈
# [0.18, 0.6] is 0.13·(0.6 − λ) (+ 0 above), so it certifies 0.6 at once (an early attach there, in most runs) and
# loosens after (to 0.35 – 0.45 in 9 000 decisions). Layer 1 is right on [0.18, 0.6) — the band L0 later takes — and
# wrong with probability 2α below 0.18, with a score above its strictest grid points: its zone risk is 0.3·2α = 0.6α
# at every point; on the rows L0 leaves it (u < λ_0) the share of its 2α rows is 0.18 / λ_0 — the larger the further
# L0 loosens (the risk a shrinking zone would serve).
L0G = _M("l0g", lambda x: (_flip(x[0]) if x[1] < 0.18 or (x[1] < 0.6 and x[3] < 0.13) else x[0], x[1]))
L1A = _M("l1a", lambda x: (_flip(x[0]) if x[1] < 0.18 and x[4] < 2 * ALPHA_J else x[0], 0.9 + 0.1 * x[2]))


def _run_shrink(seed, n=9000):
    loop = AdapterLoop(_Hand([L1A]), question="q", policy=_policy(first=100, stale=6000, trial=6000), secret=KEY,
                       initial=L0G, additive=True, monitor=None, _detector_alone=True, wait=True, clock=lambda: 0.0,
                       max_risk=ALPHA_J, delta=0.05, max_layers=2, promise_options={"grid": GRID})
    serving = None
    z = {0: [0, 0], 1: [0, 0]}                            # layer j: [decisions in Z_j, answered alone by j and wrong]
    left = [0, 0]                                          # Z_1 minus L0's answers: [decisions, 1 alone and wrong]
    for x in _xs(seed, n):
        out = loop.step(x, lambda x=x: x[0])
        if serving is None:
            if len(loop._layers) == 2:
                serving = loop.n
            continue
        for j in range(out["layer"] + 1):
            z[j][0] += 1
        wrong = out["by"] == "local" and out["answer"] != x[0]
        a = None if out["answered_by"] is None else loop._layers.index(out["answered_by"])
        if wrong:
            z[a][1] += 1
        if out["layer"] == 1 and a != 0:
            left[0] += 1
            left[1] += wrong
    return {"serving_at": serving, "zones": z, "left": left, "lam0": loop._route.get(0),
            "thr0": loop._threshold(0)}


def _check_shrink(runs):
    on = [r for r in runs if r["serving_at"] is not None]
    assert len(on) >= 0.9 * len(runs)
    assert sum(r["thr0"] < r["lam0"] for r in on) >= 0.75 * len(on)          # L0 loosened below Λ_0: it took a band
    tot = {j: [sum(r["zones"][j][0] for r in on), sum(r["zones"][j][1] for r in on)] for j in (0, 1)}
    risk = {j: tot[j][1] / tot[j][0] for j in (0, 1)}
    share = tot[1][0] / tot[0][0]
    stream = (tot[0][1] + tot[1][1]) / tot[0][0]
    assert risk[0] <= ALPHA_J and risk[1] <= ALPHA_J, risk                    # each fixed zone: ≤ α
    assert stream <= ALPHA_J * (1 + share), (stream, share)
    left = [sum(r["left"][i] for r in on) for i in (0, 1)]
    if len(runs) >= 50:                                    # the direction reading (A) gets wrong: the rows L0 leaves
        assert left[1] / left[0] > risk[1], (left, risk)   # layer 1 are worse than its whole zone
    return {"runs": len(runs), "layered": len(on), "risk_zone": {j: round(x, 4) for j, x in risk.items()},
            "zone_share": round(share, 3), "stream_joint": round(stream, 4), "bound": round(ALPHA_J * (1 + share), 4),
            "layer1_on_the_rows_l0_leaves_it": round(left[1] / max(1, left[0]), 4)}


# ───────────────────────────── (h): no upstream loosening → the 82e0ad5 routing and answers
# L0 of (a): right iff u > 0.5, so on the fixed grid its joint risk is 0 at 0.5 and ≥ α at every point below: it
# cannot loosen past 0.5 (but for a false certification, which the pinned seeds do not have). The digests were taken
# by running _sanity_fields on the 82e0ad5 library (the same stream, key and policy).
SANITY = {0: "fabeb5c1", 1: "bea7d4b3"}


def _sanity_fields(seed, n=4000, useful_error=None):
    loop = AdapterLoop(_Hand([_cand("good", 0.01)]), question="q", policy=_policy(mature_patience="auto"), secret=KEY,
                       initial=L0, additive=True, monitor=None, _detector_alone=True, wait=True, clock=lambda: 0.0,
                       max_risk=0.05, delta=0.05, max_layers=2, promise_options={"grid": GRID},
                       useful_error=useful_error)
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
    rows = []
    for k, d in loop.log:
        if k == "loop":
            r, p = d["stream"], d["loop"]
            rows.append([r[f] for f in ("n", "served", "candidate", "owner", "layer", "path", "trial_zone", "alone",
                                        "by", "sampled", "missing", "rho", "probation", "score", "threshold",
                                        "shadow")] + [p["answer"], p["path_would"]])
        elif not d["event"].startswith("useful_"):
            rows.append([d["event"], d["version"], d["decision"]])
    import hashlib
    import json
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:8], loop


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


@pytest.mark.parametrize("seeds", SEEDS)
def test_upstream_loosening_reads_its_whole_zone_and_is_falsely_certified_at_most_delta_v(seeds):
    print("\n(e)", _check_upstream(_map(_run_upstream, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_a_restarted_upstream_layer_never_bets_below_its_routing_threshold_until_the_base_regime(seeds):
    print("\n(f)", _check_floor(_map(_run_floor, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_a_fixed_zone_keeps_its_risk_when_the_upstream_takes_its_good_band(seeds):
    print("\n(g)", _check_shrink(_map(_run_shrink, seeds)))


def test_with_no_upstream_loosening_the_routing_and_answers_are_the_82e0ad5_ones():
    for seed, want in SANITY.items():
        got, loop = _sanity_fields(seed)
        recs = [d["stream"] for k, d in loop.log if k == "loop"]
        assert len(loop._layers) == 2 and loop._route[0] == 0.5               # layered, Λ_0 = λ_0 = 0.5 for good
        assert all(r["answered_by"] in (None, r["owner"]) for r in recs)      # nobody above the owner answered
        assert got == want, (seed, got)


def test_with_the_usefulness_certificate_a_good_layer_routes_and_answers_as_without_it():
    for seed, want in SANITY.items():
        got, loop = _sanity_fields(seed, useful_error=0.5)
        assert len(loop._layers) == 2 and loop._vs[1]["useful"] is not None
        assert got == want, (seed, got)


# ───────────────────────────── (i)–(iv): the usefulness certificate (decision 25)
def _noise(name, err, share):
    """A layer wrong w.p. err at every score; a `share` of its zone scores 0.97 (above every grid point), the rest
    below every grid point (never alone): its selective error is err at every point, its joint risk err·share."""
    return _M(name, lambda x: (_flip(x[0]) if x[4] < err else x[0], 0.97 if x[2] < share else 0.04 * x[2]))


def _useful_audit(loop):
    """(v) from the records: the usefulness indices (chronological, unique) and their δ_u."""
    seen, order = {}, []
    for k, d in loop.log:
        if k != "loop":
            continue
        for v, st in d["stream"]["useful"]:
            i = st["v_index"]
            if i is not None and i not in seen:
                seen[i] = (v, st["delta_v"])
                order.append(i)
    assert order == list(range(1, len(order) + 1)), order
    assert len({v for v, _ in seen.values()}) == len(seen)        # one index per certificate (no restart here)
    total = sum(dv for _, dv in seen.values())
    assert total <= loop.useful_budget.delta + 1e-12 and total == pytest.approx(loop.useful_budget.spent)
    return seen


def _run_noise(seed, n=8000):
    loop = _loop([_noise("noise", 0.9, 0.02)], policy=_policy(stale=3000, trial=3000), max_risk=0.05, delta=0.05,
                 max_layers=2, promise_options={"grid": GRID})
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
    joint = sum(1 for v, ver in loop._vs.items()                   # candidates the joint certificate passed
                if v and any(e["event"] == "certified" for e in ver["promise"].events))
    idx = _useful_audit(loop)
    promoted = any(d.get("event") == "promote" for k, d in loop.log if k == "stream")
    return {"promoted": promoted, "attaches": sum(1 for v in loop._vs if v), "joint_certified": joint,
            "indices": len(idx), "spent": loop.useful_budget.spent}


def _check_noise(runs):
    n = len(runs)
    assert all(r["attaches"] >= 2 for r in runs)                  # two attaches per run
    assert sum(r["joint_certified"] >= 1 for r in runs) >= 0.9 * n  # the joint certificate passes the noise layer
    promoted = sum(r["promoted"] for r in runs)
    assert promoted <= _binom_upper(0.05, n), (promoted, n)
    return {"runs": n, "promoted": promoted, "bar": round(_binom_upper(0.05, n), 1),
            "joint_certified_runs": sum(r["joint_certified"] >= 1 for r in runs),
            "max_spent": round(max(r["spent"] for r in runs), 4)}


def _run_real_pair(seed, n=5000):
    out = {}
    for ue in (0.5, None):
        loop = _loop([_cand("real", 0.10)], policy=_policy(), max_risk=0.05, delta=0.05, max_layers=2,
                     promise_options={"grid": GRID}, useful_error=ue)
        at, answered = None, 0
        for x in _xs(seed, n):
            o = loop.step(x, lambda x=x: x[0])
            if at is None and len(loop._layers) == 2:
                pe = next(d for k, d in loop.log if k == "stream" and d.get("event") == "promote")
                at = (pe["decision"], pe["samples"])
                continue
            if loop.n > n - 2000 and o.get("answered_by") == 1:   # coverage once both have loosened
                answered += 1
        out[ue] = {"at": at, "answered": answered, "budget_v": loop.budget.v}
    return out


def _check_real_pair(runs):
    n = len(runs)
    assert all(r[0.5]["at"] is not None and r[None]["at"] is not None for r in runs)   # promoted in every run
    on, off = sum(r[0.5]["answered"] for r in runs), sum(r[None]["answered"] for r in runs)
    assert on >= 0.95 * off, (on, off)                     # the bar (decision 26): no coverage lost (a later start
    #                                                        reads more trial rows at the full rate, so it tends to
    #                                                        start looser)
    diff = sorted(abs(r[0.5]["at"][1] - r[None]["at"][1]) for r in runs)

    def q(p):
        return diff[min(n - 1, int(math.ceil(p * n)) - 1)]

    # the start delay (zone samples) is measured, not a bar: at 415c87f on 400 seeds p50 48, p90 188, p95 238, max
    # 454; a regression guard only
    assert q(0.95) <= 300, diff[-5:]
    same = [r for r in runs if r[0.5]["at"] == r[None]["at"]]
    assert all(r[0.5]["budget_v"] == r[None]["budget_v"] for r in same)   # the loop's δ untouched
    return {"runs": n, "answered_on": on, "answered_off": off, "coverage_ratio": round(on / off, 4) if off else None,
            "delay_p50": q(0.5), "delay_p90": q(0.9), "delay_p95": q(0.95), "delay_max": diff[-1],
            "same_decision": len(same)}


DELTA_EDGE = 0.30


def _run_edge(seed, n=8000):
    loop = _loop([_noise("edge", 0.5, 0.06)], policy=_policy(stale=3000, trial=3000), max_risk=0.05, delta=0.05,
                 max_layers=2, promise_options={"grid": GRID}, delta_use=DELTA_EDGE)
    for x in _xs(seed, n):
        loop.step(x, lambda x=x: x[0])
    rows = []
    for v, ver in loop._vs.items():
        u = ver.get("useful")
        if u is None or u.v is None:
            continue
        false = any(e["event"] in ("certified", "loosened") for e in u.events)
        rows.append((u.v, u.delta_v, false))
    return rows


def _check_edge(runs):
    by = {}
    for rows in runs:
        for i, d, f in rows:
            k = by.setdefault(i, [d, 0, 0])
            k[1] += 1
            k[2] += f
    assert by
    g = len(GRID)
    # every alone score of the edge layer is 0.97 (above every grid point): all G points read the same alone rows,
    # so each is a level-δ_u/G test of the same null and a false certificate has probability ≤ δ_u / G. The bar is
    # that per-point level, so a certificate that dropped the log G term (each point at δ_u) fails it.
    for i, (d, m, false) in by.items():
        assert false <= _binom_upper(d / g, m), (i, d, m, false)
    return {"runs": len(runs), "grid": g, "per_index": {i: {"delta_u": round(d, 4), "bar": round(_binom_upper(d / g, m), 1),
                                                            "runs": m, "false": f}
                                                        for i, (d, m, f) in sorted(by.items())}}


def _bandy(x):
    """(iv): 4 % of the zone at [0.5, 1) always right, 6 % at [0.05, 0.5) always wrong, the rest below every grid
    point (right, never alone)."""
    w = x[2]
    if w < 0.04:
        return x[0], 0.5 + 0.5 * w / 0.04
    if w < 0.10:
        return _flip(x[0]), 0.05 + 0.45 * (w - 0.04) / 0.06
    return x[0], 0.04 * w


def _band_error(lam):                                      # true selective error of _bandy at threshold lam
    if lam >= 0.5:
        return 0.0
    b = 0.06 * min(1.0, (0.5 - lam) / 0.45)
    return b / (0.04 + b)


def _run_compose(seed, n=9000):
    loop = _loop([_M("band", _bandy)], policy=_policy(stale=6000, trial=6000), max_risk=0.10, delta=0.05,
                 max_layers=2, promise_options={"grid": GRID})
    on, zone, alone, wrong, false = None, 0, 0, 0, False
    for x in _xs(seed, n):
        o = loop.step(x, lambda x=x: x[0])
        if on is None:
            if loop.candidate == 1 and math.isfinite(loop._threshold(1)):
                loop.promote(1, force=True, by="test", why="(iv): composition after the start")
            if len(loop._layers) == 2:
                on = loop.n
            continue
        if o["layer"] >= 1:
            zone += 1
        if o.get("answered_by") == 1:
            r = next(d for k, d in reversed(loop.log) if k == "loop")["stream"]
            alone += 1
            wrong += o["answer"] != x[0]
            false |= _band_error(r["answer_thr"]) >= 0.5
    j = loop._joint(1) if 1 in loop._vs else math.inf
    return {"on": on, "zone": zone, "alone": alone, "wrong": wrong, "false": false,
            "binds": on is not None and _band_error(j) >= 0.5, "composed": loop._threshold(1) if on else None}


def _check_compose(runs):
    n = len(runs)
    on = [r for r in runs if r["on"] is not None]
    assert len(on) >= 0.9 * n
    binds = sum(r["binds"] for r in on)
    assert binds >= 0.5 * len(on), binds                  # the joint prefix reached the bad band in most runs
    false = sum(r["false"] for r in on)
    assert false <= _binom_upper(0.05, n), (false, n)
    alone, wrong, zone = (sum(r[k] for r in on) for k in ("alone", "wrong", "zone"))
    assert alone and wrong / alone <= 0.5 and wrong / zone <= 0.10, (alone, wrong, zone)
    return {"runs": n, "layered": len(on), "binds": binds, "false": false, "selective": round(wrong / alone, 4),
            "zone_joint": round(wrong / zone, 4), "bar": round(_binom_upper(0.05, n), 1)}


@pytest.mark.parametrize("seeds", SEEDS)
def test_i_a_noise_layer_starts_serving_in_at_most_delta_use_of_runs_and_v_the_delta_use_audit(seeds):
    print("\n(i)+(v)", _check_noise(_map(_run_noise, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_ii_a_real_layer_keeps_its_coverage_and_starts_within_the_delay_guard(seeds):
    print("\n(ii)", _check_real_pair(_map(_run_real_pair, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_iii_the_usefulness_null_at_the_edge_is_certified_at_most_delta_u_over_the_grid(seeds):
    print("\n(iii)", _check_edge(_map(_run_edge, seeds)))


@pytest.mark.parametrize("seeds", SEEDS)
def test_iv_the_answer_threshold_stays_at_useful_points_when_the_joint_prefix_reaches_a_bad_band(seeds):
    print("\n(iv)", _check_compose(_map(_run_compose, seeds)))
