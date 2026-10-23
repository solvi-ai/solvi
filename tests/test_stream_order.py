"""solvi.experimental.stream: labels that arrive late and out of order, labels that never arrive, and the e-processes
against reference values computed here from the published formulas.

Simulated stream (seeded): score s ~ U(0, 1), System 1 wrong with probability C·(1 − s). Joint risk at a threshold λ:
R(λ) = C(1 − λ)²/2; the error among the answers given alone: e(λ) = C(1 − λ)/2. A threshold is "bad" when its true
risk is above the promised rate α; a run fails when a bad threshold is ever in force. Ville + the fixed sequence give
P(fail) ≤ δ_1 (the first index of the δ budget) when the samples are read in sampling order."""
import heapq
import math
import random
import warnings
from fractions import Fraction

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AgentOracle, MirrorBound, Mixture, RiskDetector, StreamPromise, TicketDesk
    from solvi.experimental.stream._eprocess import FRACTIONS, WEIGHTS, DeltaBudget
    from solvi.experimental.stream.sampling import StreamDriver

from solvi.experimental._evalue import MeanBound

GRID = [round(0.1 * k, 1) for k in range(9, 0, -1)]          # 0.9 … 0.1, strict → loose
C = 0.6
ALPHA, DELTA = 0.10, 0.10
DELTA_1 = DeltaBudget(DELTA).delta_at(1)                       # 6δ/π² ≈ 0.0608
SLACK = 0.06                                                   # ~2.5 binomial sd at 120 runs and p ≈ δ_1


def _risk(lam, kind):
    return C * (1 - lam) ** 2 / 2 if kind == "joint" else C * (1 - lam) / 2


def _bad(sp):
    return not math.isinf(sp.threshold) and _risk(sp.threshold, sp.kind) > sp.level


def _promise(kind="joint", alpha=ALPHA, delta=DELTA, grid=GRID):
    return StreamPromise(**{("max_risk" if kind == "joint" else "max_error"): alpha}, delta=delta, grid=grid)


# --- 1. late labels, through the driver the Dispatcher runs
def _stream(seed, delay="none", k=0, kind="joint", T=1000, in_order=True, missing=0.0, trace=None):
    """One simulated stream through a StreamDriver: decision t issued at t (its coin drawn from the rate in force),
    its label delivered at t + d_t. delay: "none"; "random" (d_t uniform on 0..k, independent of everything);
    "wrong" (d_t = k exactly when System 1's answer is wrong: an oracle slow on the hard cases). in_order: the
    driver is told the numbers (issue/after(n=)) — False is the arrival-order reading the driver did before.
    missing: a share of sampled labels that never arrive (MCAR; delivered as lost). → (ever bad, promise)."""
    rng, drng = random.Random(seed), random.Random(-seed)     # the delays draw from their own generator
    sp = _promise(kind)
    drv = StreamDriver(sp, floor=1.0)                           # every decision sampled: the hardest case for delays
    heap, bad = [], False

    def deliver(item):
        nonlocal bad
        _, n, kw = item
        drv.after(**kw, **({"n": n} if in_order else {}))
        if trace is not None:
            trace.append((n, sp.state()))
        bad = bad or _bad(sp)

    for t in range(1, T + 1):
        rho, _ = drv.before()
        s, u = rng.random(), rng.random()
        wrong = u < C * (1 - s)
        sampled = rng.random() < rho
        lost = sampled and rng.random() < missing
        thr = sp.threshold
        if in_order:
            drv.issue(t, sampled)
        kw = {"score": s, "would": "a", "alone": s >= thr, "label": None if lost else ("b" if wrong else "a"),
              "label_source": "s2", "sampled": sampled, "free": s < thr, "rho": rho,
              "in_force": None if math.isinf(thr) else thr, "lost": lost}
        d = 0 if delay == "none" else (drng.randint(0, k) if delay == "random" else (k if wrong else 0))
        heapq.heappush(heap, (t + d, t, kw))
        while heap and heap[0][0] <= t:
            deliver(heapq.heappop(heap))
    while heap:
        deliver(heapq.heappop(heap))
    return bad, sp


def _fail_rate(runs, **kw):
    return sum(_stream(1000 + s, **kw)[0] for s in range(runs)) / runs


def test_arrival_order_reading_breaks_the_promise_when_the_delay_depends_on_the_answer():
    """The hole the driver closes: read in arrival order, wrong answers labelled 300 decisions late certify a bad
    threshold on nearly every stream (the clean labels arrive first)."""
    assert _fail_rate(20, delay="wrong", k=300, in_order=False) >= 0.9


@pytest.mark.parametrize("delay,k,kind", [("none", 0, "joint"), ("random", 300, "joint"), ("wrong", 50, "joint"),
                                          ("wrong", 300, "joint"), ("wrong", 300, "selective")])
def test_late_labels_keep_the_promise(delay, k, kind):
    rate = _fail_rate(120, delay=delay, k=k, kind=kind)
    assert rate <= DELTA_1 + SLACK, f"P(a threshold with risk > α ever in force) = {rate:.3f}"


def test_the_result_does_not_depend_on_the_arrival_order():
    """The same samples delivered in sampling order, with random delays and with answer-dependent delays: after the
    last label the certificate is the same state, and whenever a delivery has read as many samples as the in-order
    run had at that decision, the evidence and threshold are the same."""
    finals, traces = [], []
    for delay, k in (("none", 0), ("random", 300), ("wrong", 300)):
        tr = []
        finals.append(_stream(7, delay=delay, k=k, T=600, trace=tr)[1].state())
        traces.append(dict(tr))                                 # n → the state after the delivery of n
    # ρ = 1 throughout (floor=1.0), so the coins and the samples are the same whatever the delays
    assert finals[0] == finals[1] == finals[2]
    ins = traces[0]
    for tr in traces[1:]:                                       # a delivery that released samples up to n (and no
        for n, st in tr.items():                                # later) shows the in-order state at n
            if st["n"] == ins[n]["n"]:
                assert st["evidence_fp"] == ins[n]["evidence_fp"] and st["threshold"] == ins[n]["threshold"]


def test_a_held_sample_waits_for_every_earlier_open_one():
    sp = _promise(grid=[0.0])
    drv = StreamDriver(sp, floor=1.0)
    for n in (1, 2, 3):
        drv.issue(n, True)
    kw = {"score": 1.0, "would": "a", "alone": False, "label": "a", "label_source": "s2", "sampled": True,
          "free": True, "rho": 1.0, "in_force": None}
    rec, _, _ = drv.after(**kw, n=3)
    assert sp.n == 0 and rec["certificate"]["held"] == 1
    rec, _, _ = drv.after(**kw, n=2)
    assert sp.n == 0 and rec["certificate"]["held"] == 2
    rec, _, _ = drv.after(**{**kw, "label": None, "lost": True}, n=1)    # expired: a missing label releases them
    assert sp.n == 3 and sp.missing == 1 and "held" not in rec["certificate"]
    drv.issue(4, False)                                         # an unsampled decision holds nothing
    drv.after(**{**kw, "sampled": False}, n=4)
    drv.issue(5, True)
    drv.after(**kw, n=5)
    assert sp.n == 4


# --- 1b. the same through tickets: answer-dependent delays, a restart in the middle, an exact replay
def _system():
    import hashlib

    from solvi import Answer, Catalog, Decision, Question, System

    def p(case):
        return 0.02 + 0.96 * (int(hashlib.sha256(str(case).encode()).hexdigest()[:8], 16) / 16 ** 8)

    cat = Catalog()

    @cat.rule("intent")
    def intent(case: int):
        q = p(case)
        return Decision("billing" if q > 0.5 else "cards", {"billing": q, "cards": 1 - q}, confidence=max(q, 1 - q))
    return System(cat, [Question("intent", "Which team handles this case?", Answer.choice(["billing", "cards"]))]), p


def _desk(store):
    from solvi.core.dispatch import Dispatcher
    s1, p = _system()
    sp = StreamPromise(max_risk=0.05, grid=[round(1 - 0.05 * k, 2) for k in range(1, 20)])
    s1.guarantee("intent", promise=sp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=RiskDetector(0.05, arl=200),
                       on_drift="recertify", probation=50, secret="k", storage=store)
    return TicketDesk(d, ttl=10 ** 6, clock=lambda: 1000.0), sp, p


def test_tickets_held_on_wrong_answers_restart_and_replay_exactly(tmp_path):
    from solvi.core.store import JSONLStorage
    path = tmp_path / "t.jsonl"
    desk, sp, p = _desk(JSONLStorage(path))

    def gold(c):                                                # System 1 wrong on 1 case in 7
        right = "billing" if p(c) > 0.5 else "cards"
        return right if c % 7 else ("cards" if right == "billing" else "billing")

    held = []                                                   # tickets of wrong answers: resolved 40 cases later
    for c in range(1, 241):
        r = desk.decide({"case": c})
        if r["status"] == "needs_verdict":
            if c % 7 == 0:
                held.append((c + 40, c, r["ticket"]))
            else:
                desk.resolve(r["ticket"], gold(c))
        while held and held[0][0] <= c:
            _, cc, t = held.pop(0)
            desk.resolve(t, gold(cc))
        if c == 120:                                            # a restart with tickets open
            state = sp.state()
            desk, sp, p = _desk(JSONLStorage(path))
            assert sp.state() == state and desk.open()
    for _, cc, t in held:
        desk.resolve(t, gold(cc))
    assert not desk.open()
    rep = desk.dispatcher.replay_stream()
    assert rep["ok"], rep["mismatches"][:3]
    recs = [x.data for x in JSONLStorage(path).iter(kind="dispatch")]
    assert any((r["stream"]["certificate"].get("held") or 0) > 0 for r in recs)   # samples did wait
    assert sp.n == sum(1 for r in recs if r["stream"]["sample"]["sampled"])        # every sample read at the end


# --- 2. missing labels: missing-as-loss under MCAR and adversarial missingness
def _missing_run(seed, m, how, kind="joint", T=1500):
    """Samples read in order (ρ = 1); a share m of labels missing — "mcar", "wrong" (missing exactly when System 1
    is wrong, with probability min(1, m / P(wrong))) or "right" (exactly when it is right). → (ever bad, coverage of
    the final threshold = P(s ≥ λ̂))."""
    rng = random.Random(seed)
    sp = _promise(kind)
    p_wrong = C / 2
    bad = False
    for _ in range(T):
        s = rng.random()
        wrong = rng.random() < C * (1 - s)
        r = rng.random()
        miss = {"mcar": r < m, "wrong": wrong and r < min(1.0, m / p_wrong),
                "right": (not wrong) and r < min(1.0, m / (1 - p_wrong))}[how]
        sp.learn(s, "a", None if miss else ("b" if wrong else "a"))
        bad = bad or _bad(sp)
    return bad, 0.0 if math.isinf(sp.threshold) else 1.0 - sp.threshold


@pytest.mark.parametrize("how,kind", [("mcar", "joint"), ("wrong", "joint"), ("right", "joint"),
                                      ("mcar", "selective"), ("wrong", "selective")])
def test_missing_labels_keep_the_promise(how, kind):
    runs = 80
    rate = sum(_missing_run(2000 + s, 0.3, how, kind)[0] for s in range(runs)) / runs
    assert rate <= DELTA_1 + SLACK, f"{how}/{kind}: P(fail) = {rate:.3f}"


def test_missing_labels_cost_coverage_as_m_grows():
    """Coverage (share answered alone at the final threshold), mean over 40 streams, joint kind, MCAR. Measured
    (seeds 3000..3039, 1 500 samples): m = 0 → 0.485, 0.1 → 0.335, 0.3 → 0.203, 0.5 → 0.108; the grid's loosest good
    point 0.5 gives 0.5 (R(λ) = α at λ ≈ 0.42)."""
    cov = {}
    for m in (0.0, 0.1, 0.3, 0.5):
        cov[m] = sum(_missing_run(3000 + s, m, "mcar")[1] for s in range(40)) / 40
    assert cov[0.0] >= cov[0.1] >= cov[0.3] >= cov[0.5]
    assert cov[0.0] - cov[0.5] > 0.2


@pytest.mark.parametrize("kind", ["joint", "selective"])
def test_missing_exactly_when_wrong_reads_as_the_full_labels(kind):
    """Missing-as-loss gives a missing label the loss of a wrong answer wherever it would be answered alone: when the
    missing labels are exactly the wrong answers, the evidence is the full-label evidence, bit for bit."""
    rng = random.Random(11)
    a, b = _promise(kind), _promise(kind)
    for _ in range(800):
        s = rng.random()
        wrong = rng.random() < C * (1 - s)
        a.learn(s, "a", "b" if wrong else "a")
        b.learn(s, "a", None if wrong else "a")
        sa, sb = a.state(), b.state()
        assert sa["evidence_fp"] == sb["evidence_fp"] and sa["threshold"] == sb["threshold"]


def test_a_missing_label_never_certifies_more_than_the_full_label():
    """Pathwise, mixture bets: the same stream with any labels removed is never looser than with all of them."""
    rng = random.Random(5)
    full, part = _promise(), _promise()
    for _ in range(1500):
        s = rng.random()
        y = "b" if rng.random() < C * (1 - s) else "a"
        full.learn(s, "a", y)
        part.learn(s, "a", None if rng.random() < 0.3 else y)
        assert part.threshold >= full.threshold


# --- 3. the e-processes against reference values from the formulas
# Waudby-Smith, I. & Ramdas, A. (2024). Estimating means of bounded random variables by betting. JRSSB 86(1), 1–27
# (arXiv:2010.09686). Capital process against H0: E[X] = m (one-sided here): K_t = Π_{i≤t} (1 + λ_i (X_i − m)),
# λ_i predictable; a mixture Σ_j w_j K_t(λ^(j)) of test martingales is one, and P(sup_t K_t ≥ 1/δ) ≤ δ (Ville).

def _ref_mixture(losses, alpha, fractions=FRACTIONS, weights=WEIGHTS):
    """Σ_j w_j Π_i (1 + f_j/(1 − α)·(α − L_i)), exact in rationals (the floats' exact values)."""
    a = Fraction(alpha)
    tot = Fraction(0)
    for f, w in zip(fractions, weights):
        b = Fraction(f) / (1 - a)
        k = Fraction(1)
        for x in losses:
            k *= 1 + b * (a - Fraction(x))
        tot += Fraction(w) * k
    return tot


def _ref_agrapa(xs, m, c=0.75, deviation="paper"):
    """aGRAPA (WSR 2024, §5.1 / App. B; confseq `lambda_aKelly`): λ_t = (μ̂_{t−1} − m)/(σ̂²_{t−1} + (μ̂_{t−1} − m)²),
    truncated to [0, c/m] (one-sided), μ̂_t = (1/2 + Σ_{i≤t} X_i)/(t + 1), σ̂²_t = (1/4 + Σ_{i≤t} (X_i − μ̂_i)²)/(t + 1).
    deviation="paper": X_i − μ̂_i (μ̂_i includes X_i, as in the paper and confseq); "predictable": X_i − μ̂_{i−1}.
    → the log capital after each X."""
    out, logk, s, ss = [], 0.0, 0.0, 0.0
    for t, x in enumerate(xs, start=1):
        mu_prev = (0.5 + s) / t
        var_prev = (0.25 + ss) / t
        d = mu_prev - m
        lam = min(max(d / (var_prev + d * d), 0.0), c / m)
        logk += math.log(1 + lam * (x - m))
        s += x
        mu_now = (0.5 + s) / (t + 1)
        ss += (x - (mu_now if deviation == "paper" else mu_prev)) ** 2
        out.append(logk)
    return out


def _losses(seed, n, p):
    rng = random.Random(seed)
    return [1.0 if rng.random() < p else 0.0 for _ in range(n)]


@pytest.mark.parametrize("alpha,p", [(0.05, 0.0), (0.05, 0.02), (0.1, 0.1), (0.2, 0.35)])
def test_mixture_matches_the_reference_capital(alpha, p):
    xs = _losses(int(alpha * 1000 + p * 100), 400, p)
    proc = Mixture(alpha)
    for t, x in enumerate(xs, start=1):
        got = proc.update(x)
        if t % 25 == 0 or t == len(xs):
            ref = _ref_mixture(xs[:t], alpha)
            assert got == pytest.approx(math.log(ref), rel=1e-12, abs=1e-12)


def test_mixture_known_answer_one_loss():
    """α = 0.05: a clean sample multiplies component j by 1 + f_j/19, a loss by 1 − f_j."""
    want = sum(w * (1 + f / 19) ** 3 * (1 - f) for f, w in zip(FRACTIONS, WEIGHTS))
    proc = Mixture(0.05)
    for x in (0.0, 1.0, 0.0, 0.0):
        proc.update(x)
    assert math.exp(proc.log_wealth) == pytest.approx(want, rel=1e-14)


@pytest.mark.parametrize("m,p", [(0.95, 1.0), (0.9, 0.97), (0.5, 0.7), (0.3, 0.3)])
def test_meanbound_matches_agrapa_with_the_predictable_deviation(m, p):
    """MeanBound is aGRAPA with σ̂² over deviations from μ̂_{i−1} (the mean before X_i); equal to 1e-12."""
    xs = [1.0 - x for x in _losses(int(m * 100 + p * 10), 300, 1 - p)]
    mb = MeanBound(m)
    got = [mb.update(x) for x in xs]
    assert got == pytest.approx(_ref_agrapa(xs, m, deviation="predictable"), rel=1e-12, abs=1e-12)
    mirror = MirrorBound(1 - m)
    assert [mirror.update(1.0 - x) for x in xs] == pytest.approx(got, rel=1e-12, abs=1e-12)


def test_meanbound_differs_from_the_paper_variance_estimate():
    """Recorded deviation (not a validity issue: either bet is predictable): the paper and confseq use X_i − μ̂_i.
    On a clean stream at α = δ = 5 % the paper's variant needs 91 samples, MeanBound 93 (the design's number)."""
    def need(dev):
        xs, k = [], 0
        while True:
            xs.append(1.0)
            k += 1
            if _ref_agrapa(xs, 0.95, deviation=dev)[-1] >= math.log(20) - 1e-12:
                return k
    assert need("predictable") == 93
    assert need("paper") == 91
