"""StreamPromise: an anytime-valid learn-then-test certificate on a threshold grid, earned on the stream (experimental).

One e-process per grid point λ_i (strict → loose) against H0 "the risk at λ_i is above α"; a point is certified once
its wealth reaches 1/δ_v (Ville), and betting stops there. The threshold in force λ̂ is the loosest grid point such
that every stricter point is certified (fixed-sequence testing: the order is fixed before any label), +inf (answer
nothing alone) until the first one. The wealths only grow to certification, so λ̂ only loosens; a model that degrades
is never un-certified here — that needs a change detector and `restart()` (a later stage).

Kinds. joint (`max_risk=`): the loss of a sample at λ is L = 1{score ≥ λ and the answer is wrong} — "P(answered alone
and wrong) ≤ α, a share of all inputs". selective (`max_error=`): point i reads only samples with score ≥ λ_i, loss
W = 1{wrong} — "the error among the answers given alone ≤ α".

Statement. With probability ≥ 1 − δ_v, at every point in time, the risk at the threshold in force is at most α — on a
stream whose conditional risk given the past is constant (i.i.d. between change points) and whose labelled samples
are drawn by a coin independent of the input. Over restarts, δ_v = `DeltaBudget.next()` (Basel: 6δ/(π²v²)), so every
threshold ever in force holds together with probability ≥ 1 − δ.

A missing label (`learn(..., label=None)`: an expired ticket, a failed oracle) counts as a loss at every point where
the sample would have been answered alone: its factor is ≤ 1, so it can only lower the wealth, whatever the reason the
label is missing.

The grid. "auto": `ltt_grid` over the scores of the first `grid_from` decisions observed (labels unread), at most
`grid_size` points; until then nothing certifies and a labelled sample is not read (counted in `dropped`). "fixed":
`fixed_grid()` (kadi's shape on [0, 1]). A sequence: those thresholds. +inf is always the "answer nothing" point.

The data a decision records (`state()`): v_index, delta_v, threshold, n (samples read), seen (decisions observed),
missing, dropped, certified (points), evidence_fp (SHA-256 of the log-wealth vectors rounded to 1e-9 and the certified
flags), grid_fp. The same samples in the same order give the same state, so the record can be rebuilt and compared.
`fingerprint()` is the configuration only (what a recorded guard is checked against on replay); the state is not in
it."""
from __future__ import annotations

import hashlib
import json
import math

from ._eprocess import FRACTIONS, WEIGHTS, DeltaBudget, MirrorBound, Mixture

KINDS = ("joint", "selective")
BETS = ("mixture", "agrapa")


def fixed_grid(lo: float = 0.0, hi: float = 1.0) -> list:
    """kadi's fixed grid shape {1 − 10^(−e/8), e = 1..32} ∪ {0.05, 0.10, …, 0.95}, mapped onto [lo, hi]; strict → loose."""
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        raise ValueError("the score range needs hi > lo")
    ps = {1.0 - 10.0 ** (-e / 8.0) for e in range(1, 33)} | {round(0.05 * k, 2) for k in range(1, 20)}
    return sorted({lo + (hi - lo) * p for p in ps}, reverse=True)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _score(signal) -> float:
    if signal is None:
        return -math.inf
    s = float(signal)
    return -math.inf if math.isnan(s) else s


class StreamPromise:
    """See the module docstring. The `Promise` interface (method "stream-ltt", level, delta, threshold, nodes=None,
    report, promise, stateful, allows, threshold_of, record, fingerprint), so `System.guarantee(question, promise=sp)`
    installs it as it installs `OpenSetGate`: every gated decision is observed (no label), its state recorded with the
    guard record, and a replay takes the recorded threshold. Labels come in through `learn` (one labelled sample: the
    sampling and the oracle are the caller's — a later stage wires them into the Dispatcher). Groups are not supported."""

    stateful = True
    method = "stream-ltt"
    nodes = None
    by = None
    min_group = None

    def __init__(self, max_risk=None, max_error=None, delta=None, *, kind=None, grid="auto", grid_from=200,
                 grid_size=32, bets="mixture", fractions=FRACTIONS, weights=WEIGHTS, c=0.75, budget=None):
        if (max_risk is None) == (max_error is None):
            raise ValueError("give the promise as max_risk= (joint: P(answered alone and wrong)) or max_error= "
                             "(selective: the error among the answers given alone)")
        want = "joint" if max_risk is not None else "selective"
        if kind is not None and kind != want:
            raise ValueError(f"kind={kind!r} does not match {'max_risk' if want == 'joint' else 'max_error'}= "
                             f"(kind {want!r})")
        alpha = float(max_risk if max_risk is not None else max_error)
        if not 0.0 < alpha < 1.0:
            raise ValueError("the promised rate is strictly between 0 and 1")
        if bets not in BETS:
            raise ValueError(f"bets is one of {BETS}, not {bets!r}")
        if budget is None:
            budget = DeltaBudget(0.05 if delta is None else delta)
        elif delta is not None and abs(float(delta) - budget.delta) > 1e-15:
            raise ValueError(f"delta={delta} differs from the budget's delta {budget.delta}")
        self.kind, self.level, self.budget = want, alpha, budget
        self.delta = budget.delta
        self.bets, self.fractions, self.weights, self.c = bets, tuple(fractions), tuple(weights), float(c)
        self._new_bet()                                    # checks fractions / weights / c once
        self.grid_from, self.grid_size = int(grid_from), int(grid_size)
        if self.grid_from < 1 or self.grid_size < 1:
            raise ValueError("grid_from and grid_size are at least 1")
        if isinstance(grid, str):
            if grid not in ("auto", "fixed"):
                raise ValueError('grid is "auto", "fixed" or a sequence of thresholds')
            self.grid_spec = grid
            pts = fixed_grid() if grid == "fixed" else None
        else:
            pts = sorted({float(x) for x in grid}, reverse=True)
            if not pts or not all(math.isfinite(x) for x in pts):
                raise ValueError("a grid is a non-empty sequence of finite thresholds")
            self.grid_spec = list(pts)
        self.grid = None
        self._buffer = []                                  # the scores the auto grid is read from
        self.seen = 0
        self.events = []                                   # state changes: grid, certified, loosened, recertify
        self.v, self.delta_v = self.budget.next()
        self._fresh()
        if pts is not None:
            self._set_grid(pts)

    # --- the processes
    def _new_bet(self):
        if self.bets == "mixture":
            return Mixture(self.level, self.fractions, self.weights)
        return MirrorBound(self.level, c=self.c)

    def _fresh(self):
        self.n, self.missing, self.dropped = 0, 0, 0
        self.threshold = math.inf
        self._proc = None if self.grid is None else [self._new_bet() for _ in self.grid]
        self._cert = None if self.grid is None else [False] * len(self.grid)

    def _set_grid(self, pts):
        self.grid = list(pts)
        self._proc = [self._new_bet() for _ in self.grid]
        self._cert = [False] * len(self.grid)
        self._event("grid", why=f"{len(self.grid)} points")

    def _event(self, event, why=None):
        st = self.state()
        self.events.append({"event": event, "v_index": self.v, "delta_v": self.delta_v,
                            "threshold": st["threshold"], "n": self.n, "state_fp": _sha(st), "why": why})

    # --- the Promise interface
    @property
    def promise(self) -> str:
        what = ("P(answered alone and wrong) ≤ " if self.kind == "joint" else
                "the error among the answers given alone ≤ ") + f"{self.level:g}"
        # no state in the text (not δ_v of the current restart): a replay re-derives the recorded verdicts with it
        return (f"with probability ≥ 1 − δ_v (version index v, δ_v = 6δ/(π²v²), δ = {self.delta:g}), at every point in "
                f"time, {what} for the threshold in force, on a stream i.i.d. between change points (the conditional "
                f"risk given the past is constant), sampled by a coin independent of the input")

    @property
    def report(self) -> dict:
        return {"method": self.method, "kind": self.kind, "threshold": None if math.isinf(self.threshold) else
                self.threshold, "n": self.n, "answered": None, "promise": self.promise, "v_index": self.v,
                "delta_v": self.delta_v}

    def threshold_of(self, group=None):
        if group is not None:
            raise ValueError("StreamPromise has no groups (one budget per group is a later stage)")
        return self.threshold, None, None

    def allows(self, signal, group=None) -> bool:
        if signal is None or not math.isfinite(float(signal)):
            return False
        return float(signal) >= self.threshold_of(group)[0]

    def observe(self, signal, result=None) -> dict:
        """One gated decision (no label): counted, and its score read for the auto grid until it is fixed."""
        self.seen += 1
        if self.grid is None:
            self._buffer.append(_score(signal))
            if len(self._buffer) >= self.grid_from:
                from ...core.calibration import ltt_grid
                pts = sorted((float(x) for x in ltt_grid(self._buffer, self.grid_size)), reverse=True)
                self._buffer = []
                if pts:
                    self._set_grid(pts)
        return self.state()

    def learn(self, signal, answer, label) -> dict:
        """One labelled sample: the decision's score, the answer it would give alone, its label (None: missing — a loss
        wherever it would have been answered alone). Before the grid is fixed the sample is not read (`dropped`)."""
        if self.grid is None:
            self.dropped += 1
            return self.state()
        s = _score(signal)
        wrong = True if label is None else answer != label
        self.missing += label is None
        level = math.log(1.0 / self.delta_v) - 1e-12
        newly = False
        for i, lam in enumerate(self.grid):
            if self._cert[i]:
                continue                                   # certified: betting stops (Ville bounds the running max)
            alone = s >= lam
            if self.kind == "selective":
                if not alone:
                    continue
                loss = 1.0 if wrong else 0.0
            else:
                loss = 1.0 if (alone and wrong) else 0.0
            if self._proc[i].update(loss) >= level:
                self._cert[i] = newly = True
        self.n += 1
        if newly:
            thr = math.inf
            for i, lam in enumerate(self.grid):            # fixed sequence: strict → loose, stop at the first not certified
                if not self._cert[i]:
                    break
                thr = lam
            if thr < self.threshold:
                first = math.isinf(self.threshold)
                self.threshold = thr
                self._event("certified" if first else "loosened")
        return self.state()

    def restart(self, why=None) -> tuple:
        """A new certification from scratch (an alarm, a person's call): the next (v, δ_v) from the budget, every point
        uncertified, wealth 1, the threshold +inf; the grid is kept (it was read without labels)."""
        self.v, self.delta_v = self.budget.next()
        self._fresh()
        self._event("recertify", why=why)
        return self.v, self.delta_v

    def stream_driver(self, detector=None, **kw):
        """The per-decision step a Dispatcher runs for this promise (`sampling.StreamDriver`): how
        `Dispatcher(promise=sp, detector=, audit="adaptive")` reaches the experimental code without importing it."""
        from .sampling import StreamDriver
        return StreamDriver(self, detector, **kw)

    # --- what is recorded
    def state(self) -> dict:
        ev = None if self._proc is None else [[round(x, 9) for x in p.vector()] for p in self._proc]
        return {"v_index": self.v, "delta_v": self.delta_v,
                "threshold": None if math.isinf(self.threshold) else self.threshold,
                "n": self.n, "seen": self.seen, "missing": self.missing, "dropped": self.dropped,
                "certified": 0 if self._cert is None else sum(self._cert),
                "evidence_fp": _sha([ev, self._cert])[:16],
                "grid_fp": None if self.grid is None else _sha(self.grid)[:16]}

    def record(self) -> dict:
        out = {"method": self.method, "kind": self.kind, "promise": self.promise, "n": self.n,
               "delta": self.delta, "delta_v": self.delta_v, "v_index": self.v, "bets": self.bets}
        out["risk" if self.kind == "joint" else "error"] = self.level
        return out

    def fingerprint(self) -> str:
        from ...core.provenance import digest
        return digest("StreamPromise", self.kind, self.level, self.delta, self.budget.schedule, self.grid_spec,
                      self.grid_from, self.grid_size, self.bets, list(self.fractions), list(self.weights), self.c)

    def __repr__(self):
        thr = "inf" if math.isinf(self.threshold) else f"{self.threshold:.4g}"
        return f"StreamPromise({self.kind}, {self.level:g}, threshold={thr}, n={self.n}, v={self.v})"


__all__ = ["BETS", "KINDS", "StreamPromise", "fixed_grid"]
