"""Calibration sampling for the stream certificate: the coin, the rate, and the per-decision step the Dispatcher runs
(experimental; stage 2).

The coin. u_t = HMAC-SHA256(secret, "sample|{question}|{run}|{n}") read as 52 bits / 2^52 (`coin_message`, the
Dispatcher's keyed hash; run = the Dispatcher's seed, n = the decision's number). **No input bytes in the message**:
the coin does not read the input, so a sample is drawn independently of (x_t, y_t) — Assumption B of the
certificate. Without a secret the same message is hashed with plain SHA-256: reproducible, but predictable by whoever
knows the seed (the Dispatcher warns once, `PredictableDrawWarning`). The draw is not recorded (it needs the secret);
replay recomputes it where the secret is given.

The rate (`Rate`). ρ_t = 1 while nothing is certified (the grid not fixed, the threshold +inf) and during probation;
otherwise ρ_t = min(1, max(floor, budget / ℓ_{t−1})), ℓ_{t−1} the share of the last `window` decisions answered alone
(clamped below at 1e-3). Every input of ρ_t is from before decision t. Defaults floor = 0.05, budget = 0.05 extra oracle
calls per decision, window = 500.

Who pays. An escalated decision that wins the coin is a free sample: the slow path answers it anyway and its accepted
answer is the label. An alone decision that wins the coin is an audit: the slow path checks it (one extra call). A
sample without an accepted slow-path answer (no slow path, no budget, a person decides, the slow path failed) has no
label: a loss wherever it would have been answered alone (`StreamPromise.learn(label=None)`).

The step (`StreamDriver`). Per decision, after System 1 answered (the promise observed it through its guard):
`before()` → (ρ_t, probation) from the past; the Dispatcher draws the coin and dispatches; `after(...)` → the record
this decision carries and the `stream` events it caused: the sample goes to the promise (`learn`), its deployed loss to
the detector (only while a threshold was in force and outside probation), an alarm restarts the promise (the next
(v, δ_v) from its budget at its first post-alarm sample — lazy charge —, threshold +inf) and opens probation
(on_drift="recertify"), and the alone flag enters the
rate's window. The same records in the same order rebuild the same state (`Dispatcher.replay_stream`).

Labels in sampling order. A decision completed later (a ticket) may get its label after later decisions: the
samples reach the promise and the detector in the order of the decisions' numbers, not in the order the labels
arrive. `issue(n, sampled)` (at `Dispatcher.start`) opens a sampled decision; its completion (`after(n=...)`) is held
while an earlier sampled decision is still open, and released, with every held sample after it, once that one is
completed (resolved, or expired: a missing label). Feeding the labels in arrival order would break the certificate
whenever the delay depends on the answer (an oracle slow on the cases System 1 gets wrong: the losses arrive after
the clean samples have certified a threshold whose risk is above α; seen in tests/test_stream_order.py). The cost: a
ticket that stays open holds every later sample until it is resolved or expires (its ttl bounds the wait)."""
from __future__ import annotations

import math
from collections import deque

FLOOR, BUDGET, WINDOW = 0.05, 0.05, 500
MIN_ALONE = 1e-3


def coin_message(question, run, n) -> str:
    """The message of the calibration coin of decision n: no input bytes."""
    return f"sample|{question}|{run}|{n}"


def coin_value(hexdigest: str) -> float:
    """A hex digest → a number in [0, 1) from its first 52 bits."""
    return int(hexdigest[:13], 16) / 16 ** 13


class Rate:
    """ρ_t from the past only. See the module docstring."""

    def __init__(self, floor: float = FLOOR, budget: float = BUDGET, window: int = WINDOW):
        floor, budget, window = float(floor), float(budget), int(window)
        if not 0.0 < floor <= 1.0:
            raise ValueError("floor is a share in (0, 1]")
        if not budget > 0.0:
            raise ValueError("budget (extra oracle calls per decision) is positive")
        if window < 1:
            raise ValueError("window is at least 1 decision")
        self.floor, self.budget, self.window = floor, budget, window
        self._alone = deque(maxlen=window)
        self._count = 0                                    # alone decisions in the window

    def rate(self, full: bool = False) -> float:
        if full or not self._alone:
            return 1.0
        share = max(MIN_ALONE, self._count / len(self._alone))
        return min(1.0, max(self.floor, self.budget / share))

    def push(self, alone: bool):
        if len(self._alone) == self.window:
            self._count -= self._alone[0]
        self._alone.append(bool(alone))
        self._count += bool(alone)

    def config(self) -> list:
        return [self.floor, self.budget, self.window]


class StreamDriver:
    """The per-decision step of the stream certificate inside a Dispatcher. promise: the StreamPromise installed on
    System 1's question; detector: a RiskDetector or None; on_drift: "recertify" (an alarm restarts the promise and
    opens probation), "human" or "accept" (an alarm is the Dispatcher's latched drift flag); probation: live decisions
    escalated after an alarm (and until re-certified, the stricter of the two)."""

    def __init__(self, promise, detector=None, *, on_drift="human", probation=300, floor=FLOOR, budget=BUDGET,
                 window=WINDOW):
        if detector is not None and abs(float(detector.alpha) - float(promise.level)) > 1e-15:
            raise ValueError(f"the detector's alpha {detector.alpha} is not the promise's rate {promise.level}")
        probation = int(probation)
        if probation < 0:
            raise ValueError("probation is a number of decisions ≥ 0")
        self.promise, self.detector, self.on_drift, self.probation = promise, detector, on_drift, probation
        self.rate = Rate(floor, budget, window)
        self.left = 0                                      # probation decisions still to go
        self._seen_events = len(promise.events)
        self._open = set()                                 # numbers of sampled decisions issued and not completed
        self._held = {}                                    # n → a completed sample waiting for an earlier open one

    def config(self) -> list:
        return [self.promise.fingerprint(), None if self.detector is None else self.detector.fingerprint(),
                self.on_drift, self.probation, self.rate.config()]

    def before(self) -> tuple:
        """(ρ_t, in probation) from the past only."""
        p = self.promise
        prob = self.left > 0
        return self.rate.rate(full=p.grid is None or math.isinf(p.threshold) or prob), prob

    def issue(self, n, sampled):
        """Decision n begun (its coin drawn): a sampled one is open until its `after(n=n)`; later samples wait for it."""
        if sampled:
            self._open.add(int(n))

    def _ready(self) -> list:
        """The held samples that no open sampled decision precedes, in the order of their numbers."""
        first = min(self._open, default=math.inf)
        out = [k for k in sorted(self._held) if k < first]
        return [self._held.pop(k) for k in out]

    def after(self, *, score, would, alone, label, label_source, sampled, free, rho, in_force, probation=None,
              label_of=None, lost=None, n=None) -> tuple:
        """One decision done → (its record, the stream events it caused, a drift reason for the Dispatcher or None).
        score: the guard's signal (None: not gated); would: System 1's would-be answer; alone: System 1's answer was
        given alone (within the threshold in force); label: the slow path's accepted answer on a sampled decision (None:
        missing); in_force: the threshold the decision was gated at (None: +inf). A decision completed later (a ticket,
        stage 3): probation — the flag `before()` gave when it was issued (default: now); label_of — the stored id of
        the record that holds the label (the ticket's); lost — no verdict came (an expired ticket): a missing label, and
        the detector's loss is 1 whatever System 1 would have done (the conservative accounting, design §5.4). An
        oracle's decision records label_of and lost; the others record what stage 2 did. n: the decision's number —
        with it, a sample waits for every earlier sampled decision `issue`d and not completed (see the module docs);
        without it, the sample is read at once."""
        p = self.promise
        prob = self.left > 0 if probation is None else bool(probation)
        version = p.v
        drift = None
        todo = []
        if sampled:
            item = (score, would, alone, label, in_force, prob, lost)
            if n is None:
                todo = [item]
            else:
                self._open.discard(int(n))
                self._held[int(n)] = item
                todo = self._ready()
        if prob and self.left > 0:                         # only decisions issued in probation count it down
            self.left -= 1
        events = self._events()
        for it in todo:
            ev, why = self._read(*it)
            events += ev
            drift = drift or why
        self.rate.push(alone)
        st = p.state()
        rec = {"sample": {"rho": rho, "sampled": bool(sampled), "free": bool(sampled and free),
                          "label_source": label_source if sampled and label is not None else None,
                          "label_of": label_of if sampled and label is not None else None,
                          "alone": bool(alone), "probation": prob, "score": score, "threshold": in_force,
                          "version": 1, **({"lost": True} if sampled and lost else {})},
               "certificate": {"version": 1, "v_index": st["v_index"], "decided_under": version,
                               "threshold": st["threshold"], "n": st["n"], "evidence_fp": st["evidence_fp"],
                               "grid_fp": st["grid_fp"], "probation": self.left,
                          **({"held": len(self._held)} if self._held else {})}}
        if self.detector is not None:
            rec["detector"] = self.detector.state()
        return rec, events, drift

    def _read(self, score, would, alone, label, in_force, prob, lost) -> tuple:
        """One sample read: the promise learns it, the detector gets its deployed loss → (events, drift or None)."""
        p, det = self.promise, self.detector
        alarm = None
        p.learn(score, would, label)
        if det is not None and in_force is not None and not prob:
            wrong = label is None or would != label
            if lost:
                loss, fed = 1.0, True                      # a lost sample: a loss for the detector, every kind
            elif p.kind == "joint":
                loss = 1.0 if (alone and wrong) else 0.0
                fed = True
            else:                                          # selective: the error among the alone answers
                loss, fed = (1.0 if wrong else 0.0), bool(alone)
            if fed and det.update(loss):
                alarm = det.state()
        events = self._events()
        if alarm is None:
            return events, None
        why = det.report()["why"][0]
        events.append({"event": "alarm", "v_index": p.v, "delta_v": p.delta_v,
                       "threshold": None if math.isinf(p.threshold) else p.threshold, "n": p.n,
                       "R": alarm["R"], "detector_n": alarm["n"], "why": why})
        det.reset()
        if self.on_drift == "recertify":
            p.restart("alarm")
            self.left = self.probation
            return events + self._events(), None
        return events, why

    def _events(self) -> list:
        new = self.promise.events[self._seen_events:]
        self._seen_events = len(self.promise.events)
        return [dict(e) for e in new]


__all__ = ["BUDGET", "FLOOR", "MIN_ALONE", "Rate", "StreamDriver", "WINDOW", "coin_message", "coin_value"]
