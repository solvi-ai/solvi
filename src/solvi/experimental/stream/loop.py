"""The adapter loop: frozen versions, each with its own stream certificate, promoted under δ-spending, driven by a
`LoopPolicy` (experimental; the LoRA learning loop, stages 2–3 — no checkpoint or attach yet).

    from solvi.experimental.stream import AdapterLoop, DefaultPolicy, HashedTextTrainer

    loop = AdapterLoop(HashedTextTrainer(), question="intent", policy=DefaultPolicy(), max_risk=0.05, secret=key)
    out = loop.step(text, oracle)          # oracle: () → the label (None: missing), called only when one is needed
    loop.pause() / loop.resume()           # no training, freezing or promotion while paused; labels and promises go on
    loop.train_now()                       # → TrainJob: ignores want_train, not the budget
    loop.promote(force=False) / loop.rollback(v)
    loop.versions(), loop.status(), loop.replay()

This is the lane's `LearningLoop` (`stream.versions`) with its constants turned into `LoopPolicy` hooks
(`stream.policy`); `LearningLoop` is now this class with `DefaultPolicy` set to the lane's constants. The protocol
is the lane's: version 0 is "no model" (or `initial=`, with its own `StreamPromise`); every version frozen after it
gets its own `StreamPromise`, fed only by coin samples drawn after its freeze, δ from one `DeltaBudget`; the calibration
coin is HMAC(secret, "sample|question|seed|n"); each decision is scored by the served version and, in shadow, by the
candidate before its label is read; labels reach the learner only after the promises read them. See `stream.versions`
for the statement and the records.

What the policy decides, per decision:

- accept(st, x, y, source) — a label at all; refused → `UntrustedLabel` before anything changes. A slow-path source
  ("s2", "verified") taken is recorded as "s2".
- split(st, key) — "train", "calib" or "holdout" by the input's content key (`key=` of step, else `question|vhash(x)`,
  the change pipeline's `content_key` for a dict input); the loop keeps the first split a key got, so a relabelled key
  never crosses. Only "train" labels reach the learner; "calib" / "holdout" labels are kept for the pre-freeze report;
  a "holdout" label also records the served version's (alone, wrong) — the improvement signal's window, reset at a
  promotion or a rollback.
- want_train(st) — asked when no candidate is in trial, no training is in flight and the loop is not paused. An online
  learner (no `train`) is frozen as a `snapshot()`; a batch learner (`train(job)`) gets a `TrainJob` with the rows of
  `select(st, train rows)` and the settings of `hyper(st)`. The loop checks `policy.budget.refuse` before every
  training (a policy cannot spend past its budget); `status()["job"]` and `LoopState.job` show the training while it
  runs (in process: stage 2 has no runner).
- freeze(st, report) — the `TrainReport` scores the trained model and the served one on the held-out labels (latest
  label per key); the trained model's threshold is its own `act_threshold` when it has one, else
  `gates.calib_threshold` on the calib labels (a fixed-set stand-in for act_guard). Honesty numbers are not computed
  here (empty: the gate passes; they need a System, stage 5). Refused → the version is recorded "discarded" at birth,
  with no promise and no δ spent, so the policy's retrain floor counts it.
- promote(st, paired) / discard(st, paired) — every decision while a candidate is in trial. `Paired` counts only the
  candidate's post-freeze coin samples since its last restart, at the thresholds in force now: novel ones (key not in
  the candidate's `train_keys`, a frozenset of the keys it was trained on, snapshotted at the freeze) for the
  comparison, seen ones for the memorisation probe. A missing label counts as wrong.
- alarm(st, who, report) — two sources, OR-ed: the `RiskDetector` (as the lane's) and a score monitor (a solvi
  `Monitor`, `observe(score) → {"drift", "why"}`, fed the served version's finite score on every decision outside
  probation; default `NumericDrift`). With a detector and no monitor the loop refuses (`monitor=None`): never the
  detector alone. who is "detector", "monitor" or "detector+monitor". "recertify" is the lane's alarm: the served and
  the candidate's promises restart, retired versions start over when rolled back to, probation opens; "rollback:<v>"
  does that and serves v; "ignore" records the alarm and changes nothing else. The detector is reset and the monitor
  re-referenced (`adapted()`, else `reset()`) on every alarm and on every change of the served version.

Serving: with `system=`, a promotion or a rollback installs the version through its `Served` (`stream.serving`:
`served=` frozen → Served, default `as_served`: a `HeadVersion` over the frozen model's `as_head(feature)`); version
0 is a `BaseVersion` (the head the System had). With `versions_dir=`, a frozen version is saved through `Served.save`
and `replay()` checks each file through `Served.load` (rescore=True: every recorded score recomputed).

Controls (each writes a "stream" event marked `control`, which `replay()` re-applies): `pause()`, `resume()`,
`train_now()` (it returns when the training is over), `promote(v=None, force=False)`
(force skips the policy's promotion rule — novel count, min_share, margin, probe — never the certificate), and
`rollback(v)`. A rollback is allowed while paused; a discard and an alarm go on while paused.

Records: as the lane's, plus per decision its content key, its split, and the wall clock `t` (the budget reads the
clock; replay reads the recorded one), and per training a "train" event (job, trigger, rows, settings, fingerprint, the
TrainReport, whether it froze). Replay re-runs the protocol from the records with a fresh copy of the policy as it was
built (pickled at construction) and the recorded training results. A `train_now` refused writes its "train" event too
(state "refused").

A hook that raises after a decision (or a control) began to change the state — the policy's, the learner's update,
keys or scoring of the held-out labels, the monitor, `Served` — stops the loop (design §1.2): the decision is recorded
with the fields known so far and an "error" event (where: the hook; call: its place in the decision; why), `status()`
says "error", the loop is paused and `step` and every control refuse from then on; the served version stays installed
(the Dispatcher's path is not the loop's). Events of the failing step after the error are not written. Replay raises
the recorded error again at the same place, so a transient fault and a deterministic one both replay exactly. A
refused label source or a failing `policy.accept` / `split` raises before anything changed and does not stop the
loop. A failed training (the learner's `train` raising) is not an error: it is recorded and the served version keeps
serving.

Stage 3 (the LoRA learner, `stream.lora_learner`; the runners, `stream.runner`): a batch learner's `TrainJob` also
carries the calib labels (the latest per key), the policy's `Budget`, the loop's α and the live `status()["job"]`
dict, which a runner fills in (runner, pid, log, device, eta, last line). What a training ends with: a frozen model
(done), `TrainingRefused` (its time estimate is above `max_train_s`: a "train" event with state "refused", counted
against the budget), any other exception (state "failed"), or `RunnerError` — the runner itself failed (the process
died or went silent) — which stops the loop like a hook that raises (status "error", the hook "learner.train"; replay
exact). `initial=` that has `install` / `uninstall` (a `LoraVersion.of_part`) is version 0's `Served` too, so a
rollback to 0 puts the part as attached back. Replay takes the recorded install fingerprints (the System is not the
replay's). A training still runs inside `step` (or `train_now`): the subprocess runner isolates it, but `step` waits
for it; applying a runner's result on a later decision is not built.

Not built here (stages 4–5): the checkpoint, label mode and `attach`. Labels are kept in memory, inputs included (`LabelRow.x`): with dict inputs the memory grows with
the label count (a label store comes with the checkpoint). `status()` takes no lock: it answers while a training runs. Unmeasured: no stand has run on this loop."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import hmac
import inspect
import json
import math
import pickle
import threading
import time
import warnings
from collections import deque
from typing import Any, NamedTuple

from ._eprocess import DeltaBudget
from .certificate import StreamPromise, _sha
from .detector import RiskDetector
from .gates import calib_threshold
from .learner import TrainJob
from .policy import DefaultPolicy, LabelRow, LoopPolicy, LoopState, Paired, TrainReport, VersionInfo
from .runner import RunnerError, TrainingRefused
from .sampling import Rate, coin_message, coin_value
from .serving import BaseVersion, as_served

S2_SOURCES = ("s2", "verified")
ACTIONS = ("recertify", "ignore")
_warned = False


class Verdict(NamedTuple):
    """What an oracle may return instead of a bare label: the label (None: missing), its source (None: step()'s) and
    label_of — the stored id of the record that holds it (a ticket's), recorded with the decision."""
    label: Any
    source: str | None = None
    label_of: str | None = None


def ticket_verdict(desk, resolved) -> Verdict:
    """A ticket resolved (or expired) on a stage-3a TicketDesk → its Verdict, read from the stored dispatch record.
    desk: the TicketDesk (or its TraceStorage); resolved: what `TicketDesk.resolve` returned, or the record's stored
    id. The agent's verdict has source "s2" (the loop takes it only with s2_labels=True), a person's "human"; a
    ticket that expired or whose verdict the contract refused has no label."""
    store = desk.dispatcher.storage if hasattr(desk, "dispatcher") else desk
    sid = resolved["stored_id"] if isinstance(resolved, dict) else resolved
    rec = store.record(sid)
    th = rec.get("s2") if rec.get("kind") == "dispatch" else None
    oracle = ((th or {}).get("record") or {}).get("oracle")
    if oracle is None:
        raise ValueError(f"record {sid} is not a decision completed through a ticket")
    if not th.get("accepted") or oracle.get("lost"):
        return Verdict(None, oracle.get("label_source"), None)
    return Verdict(th.get("answer"), oracle.get("label_source"), oracle.get("label_of"))


def content_key(question, x) -> str:
    """The default content key of an input: "question|vhash(x)" (for a dict, the change pipeline's content_key)."""
    from ...core.runtime import vhash
    return f"{question}|{vhash(x)}"


def _fresh_detector(det):
    return None if det is None else RiskDetector(det.alpha, det.arl, det.scales, det.alternatives)


def _num(x):
    """A score as recorded: None for no score or a non-finite one (the certificate reads both as -inf)."""
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _json(x):
    from ...core.store import plain
    return json.loads(json.dumps(x, default=plain))


def _alone(score, thr) -> bool:
    return score is not None and math.isfinite(thr) and math.isfinite(float(score)) and score >= thr


class _Replayed(Exception):
    """A recorded error raised again by replay at the place it was recorded."""

    def __init__(self, why):
        super().__init__(why)
        self.why = why


def _update_keywords(update) -> frozenset:
    """Which of key= / source= a learner's update takes (read once from its signature)."""
    try:
        params = inspect.signature(update).parameters
    except (TypeError, ValueError):
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return frozenset(("key", "source"))
    return frozenset(k for k in ("key", "source") if k in params)


class _Recorded:
    """A frozen model during replay: its recorded fingerprint, no scores (they are read from the records)."""

    def __init__(self, fp):
        self.fp = fp

    def score(self, x):
        raise RuntimeError("a replay reads the recorded scores")

    def fingerprint(self):
        return self.fp


class _Counting:
    """The learner during replay: only its label count matters to the protocol."""

    def __init__(self):
        self._seen = 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def snapshot(self):
        raise RuntimeError("replay takes the recorded fingerprints")


class AdapterLoop:
    """See the module docstring. learner: an online Learner (snapshot) or a batch one (train(job)); question: the
    question's name (in the coin, the content key and the records); policy: a LoopPolicy (None: DefaultPolicy());
    max_risk= / max_error=, delta: each version's StreamPromise (promise_options: its other keywords); initial: version
    0's frozen model (None: no model); detector: a RiskDetector on the served version; monitor: the score monitor
    ("auto": NumericDrift() when a detector is given, else none); probation: the decisions escalated after an alarm;
    source: the default source of the oracle's labels; secret, seed: the coin's key and run; sampling: {"floor",
    "budget", "window"} of the rate; storage: a TraceStorage for the records (None: kept in memory); versions_dir: where
    frozen versions are saved; system, feature: install the served version on a System (feature: the text fact a text
    model reads); key: input → content key (None: content_key); served: frozen → Served (None: as_served); clock: the
    wall clock (None: time.time)."""

    def __init__(self, learner, *, question, policy=None, max_risk=None, max_error=None, delta=0.05, initial=None,
                 detector=None, monitor="auto", probation=300, source="human", secret=None, seed=0, sampling=None,
                 promise_options=None, storage=None, versions_dir=None, system=None, feature="text", key=None,
                 served=None, clock=None, _detector_alone=False, _batch=None):
        global _warned
        policy = DefaultPolicy() if policy is None else policy
        if not isinstance(policy, LoopPolicy):
            raise TypeError(f"policy is a LoopPolicy (stream.policy), not {type(policy).__name__}")
        try:
            self._policy0 = pickle.dumps(policy)           # replay starts from the policy as built; it must pickle
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"the policy must pickle (the loop's replay and checkpoint keep it): {e}") from e
        self._kw = dict(question=question, max_risk=max_risk, max_error=max_error, delta=delta, initial=initial,
                        probation=probation, source=source, seed=seed, sampling=sampling,
                        promise_options=promise_options, feature=feature, key=key, served=served)
        self._pkw = {"max_risk": max_risk, "max_error": max_error, **(promise_options or {})}
        for k in ("delta", "budget"):
            if k in self._pkw:
                raise ValueError(f"{k} is the loop's own (one budget over every version)")
        StreamPromise(**self._pkw, budget=DeltaBudget(delta))          # the promise's arguments, checked once
        if int(probation) < 0:
            raise ValueError("probation is a number of decisions ≥ 0")
        level = max_risk if max_risk is not None else max_error
        if detector is not None:
            if not isinstance(detector, RiskDetector):
                raise TypeError("detector is a RiskDetector (stream.detector)")
            if abs(detector.alpha - float(level)) > 1e-12:
                raise ValueError(f"the detector's alpha {detector.alpha:g} is not the promised rate {float(level):g}")
            detector.reset()                               # the loop owns it from here
        if isinstance(monitor, str):
            if monitor != "auto":
                raise ValueError(f"monitor is a Monitor, 'auto' or None, not {monitor!r}")
            if detector is not None and not _detector_alone:
                from ...core.guarantees.drift import NumericDrift
                monitor = NumericDrift()
            else:
                monitor = None
        if monitor is not None:
            for a in ("observe", "reset"):
                if not callable(getattr(monitor, a, None)):
                    raise TypeError(f"monitor is a solvi Monitor (observe, reset): missing {a}")
        elif detector is not None and not _detector_alone:
            raise ValueError("a RiskDetector is never the only alarm: give a score monitor (monitor=, or 'auto' for "
                             "NumericDrift on the served version's scores)")
        for a in ("score", "fingerprint"):
            if initial is not None and not hasattr(initial, a):
                raise TypeError(f"initial is a frozen model (score, fingerprint): missing {a}")
        self.policy, self.live, self.question = policy, learner, str(question)
        self.detector, self.monitor, self.probation, self.left = detector, monitor, int(probation), 0
        self._monitor0 = copy.deepcopy(monitor)
        self._detector_alone = bool(_detector_alone)
        self._batch = callable(getattr(learner, "train", None)) if _batch is None else bool(_batch)
        self.alpha = float(level)
        self.source, self.seed, self.feature = str(source), int(seed), feature
        self._key = key if key is not None else (lambda x, q=self.question: content_key(q, x))
        self._served_of = served if served is not None else (lambda f, q=self.question, ft=feature: as_served(f, q, ft))
        self._clock = clock if clock is not None else time.time
        if isinstance(secret, str):
            secret = secret.encode()
        self._secret = bytes(secret) if secret else None
        if self._secret is None and not _warned:
            _warned = True
            from ...core.dispatch import PredictableDrawWarning
            warnings.warn("AdapterLoop: the calibration coin is a public hash of the question, seed and decision "
                          "number; give secret= to key it", PredictableDrawWarning, stacklevel=2)
        self.budget = DeltaBudget(delta)                   # δ over every version (the training budget: policy.budget)
        self.rate = Rate(**(sampling or {}))
        self.storage, self.system = storage, system
        self._base = (initial if initial is not None and callable(getattr(initial, "install", None))
                      and callable(getattr(initial, "uninstall", None)) else BaseVersion(system, self.question))
        from pathlib import Path
        self.versions_dir = None if versions_dir is None else Path(versions_dir)
        if self.versions_dir is not None:
            self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.served, self.candidate = 0, None
        p0 = None if initial is None else StreamPromise(**self._pkw, budget=self.budget)
        self._vs = {0: self._row(0, "served", initial, p0, 0, 0, 0, frozenset(),
                                 None if initial is None else initial.fingerprint())}
        self._rows: list = []                              # every accepted label (LabelRow)
        self._split_of: dict = {}                          # key → the split it first got (never changes)
        self._split_n = {"train": 0, "calib": 0, "holdout": 0}
        self._fed_keys: set = set()                        # keys of the train labels the learner took
        self._holdout: deque = deque(maxlen=10_000)        # the served version's (alone, wrong) on held-out labels
        self._alarms: list = []
        self._history, self._served_since = [0], 0
        self._trainings: list = []
        self._job = None
        self._attempt = 0
        self.paused = False
        self._lock = threading.RLock()
        self.log = []                                      # (kind, record) when there is no storage
        self._ev_seen = {0: 0}
        self._fps = None                                   # replay: the recorded fingerprints by version
        self._replay_trains: list = []
        self._replay_installs: list = []
        self._replay_t = 0.0
        self._step_t = None
        self._upd_kw = _update_keywords(learner.update)
        self.error = None                                  # {"n", "where", "why"} once a hook raised: the loop stops
        self._hk, self._where, self._changing = 0, "loop", False
        self._inject = None                                # replay: {"call", "why"} of a recorded error
        self._inflight = None                              # the training in flight, for an error record

    @staticmethod
    def _row(v, state, model, promise, frozen_at, seen, n_train, keys, fp):
        return {"version": v, "state": state, "model": model, "promise": promise, "frozen_at": frozen_at,
                "seen": seen, "n_train": n_train, "keys": keys, "fingerprint": fp, "samples": [], "fed": [],
                "discarded_at": None, "served_cls": None}

    # --- what the hooks read
    def _now(self) -> float:
        """The wall clock: one reading per decision (the recorded one during replay)."""
        if self._fps is not None:
            return self._replay_t
        return self._step_t if self._step_t is not None else float(self._clock())

    def _threshold(self, v):
        p = self._vs[v]["promise"]
        return math.inf if p is None else p.threshold

    def _state(self) -> LoopState:
        vs = []
        for v, ver in sorted(self._vs.items()):
            thr = self._threshold(v)
            vs.append(VersionInfo(v, ver["state"], n_train=ver["n_train"], samples=len(ver["samples"]),
                                  certified=math.isfinite(thr), threshold=thr, fingerprint=ver["fingerprint"],
                                  frozen_at=None if v == 0 else ver["frozen_at"], train_keys=ver["keys"],
                                  discarded_at=ver["discarded_at"]))
        promises = {v: self._vs[v]["promise"].state() for v in (self.served, self.candidate)
                    if v is not None and self._vs[v]["promise"] is not None}
        return LoopState(n=self.n, served=self.served, candidate=self.candidate, labels=dict(self._split_n),
                         versions=tuple(vs), holdout=tuple(self._holdout), alpha=self.alpha,
                         alarms=tuple(dict(a) for a in self._alarms), served_history=tuple(self._history),
                         served_since=self._served_since, trainings=tuple(self._trainings),
                         job=None if self._job is None else dict(self._job), now=self._now(), paused=self.paused,
                         mode="stream", promises=promises,
                         detector=None if self.detector is None else self.detector.state())

    def _paired(self, c) -> Paired:
        """The candidate's Paired at the thresholds in force now (see the module docstring)."""
        thr_k, thr_s = self._threshold(c), self._threshold(self.served)
        n_nov = a_nov = w_nov = a_srv = w_srv = n_seen = a_seen = w_seen = 0
        for sk, ss, wk, ws, seen in self._vs[c]["samples"]:
            ak = _alone(sk, thr_k)
            if seen:
                n_seen += 1
                a_seen += ak
                w_seen += ak and wk
            else:
                n_nov += 1
                a_nov += ak
                w_nov += ak and wk
                if _alone(ss, thr_s):
                    a_srv += 1
                    w_srv += ws
        return Paired(n_nov, a_nov, w_nov, a_srv, n_seen=n_seen, alone_seen=a_seen, wrong_seen=w_seen,
                      wrong_served=w_srv)

    # --- the per-decision step
    def step(self, x, oracle=None, *, source=None, key=None) -> dict:
        """One input → {"n", "answer", "by", "served", "candidate", "sampled", "label", "label_source", "events"}.
        oracle: () → the label (None: missing), called only when the decision needs one (not answered alone, or
        drawn by the coin); source: its source (default the loop's); key: the input's content key (None: the loop's
        key function)."""
        with self._lock:
            self._refuse_if_stopped()
            s, c = self._vs[self.served], self._vs.get(self.candidate)
            would_s, score_s = s["model"].score(x) if s["model"] is not None else (None, None)
            would_k, score_k = c["model"].score(x) if c is not None else (None, None)
            if oracle is None:
                def oracle():
                    raise ValueError("this decision needs a label: step(x, oracle)")
            k = self._key(x) if key is None else str(key)
            return self._advance(x, k, would_s, score_s, would_k, score_k, oracle,
                                 self.source if source is None else source)

    def _coin(self, n) -> float:
        msg = coin_message(self.question, self.seed, n).encode()
        h = hmac.new(self._secret, msg, hashlib.sha256).hexdigest() if self._secret else hashlib.sha256(msg).hexdigest()
        return coin_value(h)

    def _check(self, src, x, label):
        from ...core.sources import UntrustedLabel, check_source
        rec = "s2" if src in S2_SOURCES else check_source(src)
        if not self.policy.accept(self._state(), x, label, src):
            extra = (" (DefaultPolicy takes slow-path labels only with s2_labels=True; recorded as \"s2\")"
                     if src in S2_SOURCES else "")
            raise UntrustedLabel(f"label source {src!r}: the loop's policy does not take it{extra}")
        return rec

    def _split(self, key):
        sp = self._split_of.get(key)
        if sp is None:
            sp = self.policy.split(self._state(), key)
            if sp not in self._split_n:
                raise ValueError(f"policy.split returned {sp!r}, not 'train', 'calib' or 'holdout'")
            self._split_of[key] = sp                       # the first state change, before the decision's own
        return sp

    def _advance(self, x, key, would_s, score_s, would_k, score_k, oracle, src):
        n = self.n + 1
        sv, c = self.served, self.candidate
        thr_s = self._threshold(sv)
        prob = self.left > 0
        rho = self.rate.rate(full=math.isinf(thr_s) or prob)
        sampled = self._coin(n) < rho
        alone = not prob and score_s is not None and math.isfinite(thr_s) and score_s >= thr_s
        label, lsrc, lof, split = None, None, None, None
        if sampled or not alone:
            got = oracle()
            if isinstance(got, Verdict):
                label, lof = got.label, got.label_of
                src = src if got.source is None else got.source
            else:
                label = got
            if label is not None:
                lsrc = self._check(src, x, label)          # a refused source: nothing has changed yet
                split = self._split(key)
            else:
                lof = None
        t = self._now()
        self._step_t = t
        try:
            return self._advance_at(n, x, key, sv, c, thr_s, prob, rho, sampled, alone, label, lsrc, lof, split,
                                    would_s, score_s, would_k, score_k, t)
        finally:
            self._step_t = None

    def _advance_at(self, n, x, key, sv, c, thr_s, prob, rho, sampled, alone, label, lsrc, lof, split, would_s,
                    score_s, would_k, score_k, t):
        # --- from here the state changes: a hook that raises from here on stops the loop (`_stop`), the decision is
        # recorded with its "error" event and replay raises the same error at the same place
        self.n = n
        self._hk, self._where = 0, "loop"
        det = self.detector
        events, answer, by = [], None, None
        try:
            if self._inject is not None and self._inject["call"] == 0:
                raise _Replayed(self._inject["why"])
            for v, score in ((sv, score_s), (c, score_k)):
                if v is not None and self._vs[v]["promise"] is not None:
                    self._vs[v]["promise"].observe(score)
            if sampled:
                for v, score, would in ((sv, score_s, would_s), (c, score_k, would_k)):
                    if v is None or self._vs[v]["promise"] is None:
                        continue
                    ver = self._vs[v]
                    if not n > ver["frozen_at"]:           # cannot happen: a version is frozen at the end of a step
                        raise RuntimeError(f"decision {n} would calibrate version {v} frozen at {ver['frozen_at']}")
                    ver["promise"].learn(score, would, label)
                    ver["fed"].append(n)
                if c is not None:
                    self._vs[c]["samples"].append((score_k, score_s, label is None or would_k != label,
                                                   label is None or would_s != label, key in self._vs[c]["keys"]))
            det_rep, mon_rep, ps = None, None, self._vs[sv]["promise"]
            if sampled and det is not None and ps is not None and math.isfinite(thr_s) and not prob:
                wrong = label is None or would_s != label
                if ps.kind == "joint":
                    loss, fed = (1.0 if (alone and wrong) else 0.0), True
                else:                                      # selective: the error among the alone answers
                    loss, fed = (1.0 if wrong else 0.0), alone
                if fed and det.update(loss):
                    det_rep = det.report()
            if self.monitor is not None and not prob and score_s is not None and math.isfinite(float(score_s)):
                self._at("monitor")
                rep = self.monitor.observe(float(score_s))
                if isinstance(rep, dict) and rep.get("drift"):
                    mon_rep = rep
            if prob:
                self.left -= 1
            if label is not None:
                self._rows.append(LabelRow(key, x, label, lsrc, n))
                self._split_n[split] += 1
                if split == "train":
                    self._feed(x, label, key, lsrc)
                elif split == "holdout":
                    self._holdout.append((_alone(score_s, thr_s), would_s != label))
            self.rate.push(alone)
            if alone and not sampled:
                answer, by = would_s, "local"
            elif label is not None:
                answer, by = label, "oracle"
            else:
                answer, by = None, "person"
            if n == 1 and self._vs[0]["model"] is not None:
                events.append(self._event("start", 0, n, fingerprint=self._vs[0]["fingerprint"], why=None))
            events += self._promise_events()
            if det_rep is not None or mon_rep is not None:
                events += self._on_alarm(n, sv, c, det_rep, mon_rep)
            events += self._protocol(n)
            events += self._promise_events()
        except Exception as e:  # noqa: BLE001 — recorded, then raised again
            err = self._stop(n, e)
            events += self._promise_events() + [err]
            self._write(self._record(n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s,
                                     score_k, det, key, split, t, error={k: err[k] for k in ("where", "call", "why")}),
                        {"input": x, "would": would_s, "shadow_would": would_k, "label": label, "answer": answer},
                        events)
            raise
        rec = self._record(n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s, score_k, det,
                           key, split, t)
        payload = {"input": x, "would": would_s, "shadow_would": would_k, "label": label, "answer": answer}
        self._write(rec, payload, events)
        return {"n": n, "answer": answer, "by": by, "served": sv, "candidate": c, "sampled": sampled, "label": label,
                "label_source": lsrc, "events": [e["event"] for e in events]}

    def _record(self, n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s, score_k, det, key,
                split, t, error=None):
        st = lambda v: None if v is None or self._vs[v]["promise"] is None else self._vs[v]["promise"].state()  # noqa: E731
        rec = {"n": n, "served": sv, "candidate": c, "rho": rho, "sampled": sampled, "alone": alone, "by": by,
               "label_source": lsrc, "label_of": lof, "missing": (sampled or not alone) and label is None,
               "probation": prob, "left": self.left, "score": _num(score_s),
               "threshold": None if math.isinf(thr_s) else thr_s, "shadow": _num(score_k), "served_state": st(sv),
               "candidate_state": st(c), "detector": None if det is None else det.state(), "seen": self.live.seen(),
               "key": key, "split": split, "t": t}
        if error is not None:
            rec["error"] = error
        return rec

    # --- errors
    def _at(self, where):
        """A hook is about to run: its name and its place (the count of hooks so far in this decision or control),
        for the error record; a replay of a recorded error raises it again here."""
        self._hk += 1
        self._where = where
        inj = self._inject
        if inj is not None and inj["call"] == self._hk:
            raise _Replayed(inj["why"])

    def _stop(self, n, e, **extra) -> dict:
        """A hook raised after the state changed: the loop stops (status "error", paused; `step` and the controls
        refuse from here on); the served version stays installed. → the "error" event."""
        why = e.why if isinstance(e, _Replayed) else f"{type(e).__name__}: {e}"
        self.error = {"n": n, "where": self._where, "why": why}
        self.paused = True
        inflight, self._inflight, self._job = self._inflight, None, None
        return self._event("error", self.served, n, where=self._where, call=self._hk, why=why,
                           candidate=self.candidate, train=inflight, **extra)

    def _refuse_if_stopped(self):
        if self.error is not None:
            e = self.error
            raise RuntimeError(f"the loop stopped on an error at decision {e['n']} ({e['where']}: {e['why']}); the "
                               "served version keeps serving")

    def _guarded(self, of, args, fn):
        """A control that may change the state: an exception after the change stops the loop like a decision's."""
        self._hk, self._where, self._changing = 0, of, False
        try:
            return fn()
        except Exception as e:
            if not self._changing:
                raise
            err = self._stop(self.n, e, control=True, of=of, args=args, t=self._now())
            self._write(None, None, self._promise_events() + [err])
            raise
        finally:
            self._changing = False

    def _feed(self, x, y, key, source):
        """A train label to the learner (key= and source= when its update takes them, read once from its signature;
        a TypeError raised inside update is the learner's own and is not retried)."""
        kw = {k: v for k, v in (("key", key), ("source", source)) if k in self._upd_kw}
        self._at("learner.update")
        self.live.update(x, y, **kw)
        self._fed_keys.add(key)

    # --- alarms
    def _on_alarm(self, n, sv, c, det_rep, mon_rep) -> list:
        who = "+".join(w for w, r in (("detector", det_rep), ("monitor", mon_rep)) if r is not None)
        report = {"detector": det_rep, "monitor": mon_rep}
        self._at("policy.alarm")
        action = self.policy.alarm(self._state(), who, _json(report))
        target = None
        if isinstance(action, str) and action.startswith("rollback:"):
            target = int(action.split(":", 1)[1])
        elif action not in ACTIONS:
            raise ValueError(f"policy.alarm returned {action!r}: 'recertify', 'rollback:<v>' or 'ignore'")
        mon_why = None if mon_rep is None else (mon_rep.get("why") or [None])
        ev = [self._event("alarm", sv, n, who=who, action=action,
                          R=None if det_rep is None else det_rep["R"],
                          detector_n=None if det_rep is None else det_rep["n"],
                          why=det_rep["why"][0] if det_rep is not None else
                          (mon_why[0] if isinstance(mon_why, list) else mon_why))]
        self._alarms.append({"n": n, "who": who, "version": sv})
        self._rereference()
        if action == "ignore":
            return ev
        self.left = self.probation
        ev += self._promise_events()
        ev += self._restart(sv, n)
        if c is not None and self._vs[c]["promise"] is not None:
            ev += self._restart(c, n)                      # its certificate is about the old regime too
        for ver in self._vs.values():                      # rollback targets: restarted when rolled back to
            if ver["state"] == "retired" and ver["promise"] is not None:
                ver["stale"] = True
        if target is not None:
            ev += self._rollback(target, by="policy", why=f"alarm ({who})", control=False)[1]
        return ev

    def _rereference(self):
        if self.detector is not None:
            self.detector.reset()
        m = self.monitor
        if m is not None:
            self._at("monitor.reset")
            (m.adapted if callable(getattr(m, "adapted", None)) else m.reset)()

    # --- the version protocol
    def _event(self, event, v, n, **extra):
        p = self._vs[v]["promise"]
        st = None if p is None else p.state()
        return {"event": event, "version": v, "v_index": None if p is None else p.v,
                "delta_v": None if p is None else p.delta_v, "threshold": None if st is None else st["threshold"],
                "n": None if p is None else p.n, "state_fp": None if st is None else _sha(st), "decision": n, **extra}

    def _protocol(self, n) -> list:
        ev = []
        c = self.candidate
        if c is not None:
            paired = self._paired(c)
            st = self._state()
            if not self.paused:
                self._at("policy.promote")
            if not self.paused and self.policy.promote(st, paired):
                ev.append(self._event("promote", c, n, **{"from": self.served, "to": c, "l_k": paired.share_k,
                                                         "l_s": paired.share_s, "samples": len(self._vs[c]["samples"]),
                                                         "n_novel": paired.n_novel, "n_seen": paired.n_seen,
                                                         "why": None}))
                ev[-1]["install"] = self._serve(c, n)
            else:
                self._at("policy.discard")
                why = self.policy.discard(st, paired)
                if why:
                    ev.append(self._discard(c, n, str(why)))
        if self.candidate is None and not self.paused and self._job is None:
            self._at("policy.want_train")
            if self.policy.want_train(self._state()):
                ev += self._train("schedule", n)[1]
        return ev

    def _restart(self, v, n, why="alarm") -> list:
        """Version v's promise from scratch (the next (v, δ_v), threshold +inf, wealth 1) and, for a candidate, its
        paired samples dropped. → its "recertify" and a "restart" event."""
        ver = self._vs[v]
        ver["promise"].restart(why)
        ver["samples"], ver["stale"] = [], False
        ev = self._promise_events()
        return ev + [self._event("restart", v, n, why=why)]

    @staticmethod
    def _release(*objs):
        """Free what a version holds outside the loop (a LoRA version's adapter staged on the scorer): Served.release,
        optional, called once per object."""
        seen = set()
        for o in objs:
            if o is None or id(o) in seen:
                continue
            seen.add(id(o))
            rel = getattr(o, "release", None)
            if callable(rel):
                rel()

    def _discard(self, c, n, why):
        ver = self._vs[c]
        ver["state"] = "discarded"
        self._release(ver.get("served_as"), ver["model"])
        ver["model"] = None                                # never served: its file (versions_dir) remains
        ver["discarded_at"] = self._split_n["train"]
        self.candidate = None
        return self._event("discard", c, n, samples=len(ver["samples"]), why=why)

    def _serve(self, v, n):
        old = self.served
        if old != v:
            self._vs[old]["state"] = "retired"
        self._vs[v]["state"] = "served"
        self.served = v
        if self.candidate == v:
            self.candidate = None
        self._history.append(v)
        self._served_since = n
        self._holdout.clear()                              # the new served version's own held-out outcomes
        self._rereference()                                # a new served version: its deployed risk from scratch
        self._at("install")
        if self._fps is not None:                          # replay: the System is not the replay's; the recorded
            return self._replay_installs.pop(0) if self._replay_installs else None   # fingerprints stand
        if self.system is None:
            return None
        srv = self._base if v == 0 else self._vs[v].get("served_as")
        return None if srv is None else srv.install(self.system)

    def _report(self, frozen) -> TrainReport:
        """The trained model and the served one on the held-out labels (latest label per key); its threshold on the
        calib labels."""
        latest = {}
        for r in self._rows:
            latest[r.key] = r
        calib = [r for k, r in latest.items() if self._split_of[k] == "calib"]
        hold = [r for k, r in latest.items() if self._split_of[k] == "holdout"]
        thr = getattr(frozen, "act_threshold", None)
        if thr is None and calib:
            scored = []
            for r in calib:
                a, s = frozen.score(r.x)
                scored.append((s, a != r.y))
            thr = calib_threshold(scored, self.alpha)
        served, thr_s = self._vs[self.served]["model"], self._threshold(self.served)
        right = wrong = s_right = s_wrong = better = worse = 0
        items = []
        for r in hold:
            a, s = frozen.score(r.x)
            ck = (1 if a == r.y else -1) if thr is not None and _alone(s, thr) else 0
            if served is not None:
                sa, ss = served.score(r.x)
                cs = (1 if sa == r.y else -1) if _alone(ss, thr_s) else 0
            else:
                cs = 0
            right, wrong = right + (ck == 1), wrong + (ck == -1)
            s_right, s_wrong = s_right + (cs == 1), s_wrong + (cs == -1)
            better, worse = better + (ck > cs), worse + (ck < cs)
            items.append((r.key, ck, cs))
        return TrainReport(len(hold), right, wrong, s_right, s_wrong, better=better, worse=worse, threshold=thr,
                           n_train=0, items=tuple(items))

    def _train(self, trigger, n):
        """One training now (the budget checked) → (TrainJob, events)."""
        st = self._state()
        k = max(self._vs) + 1
        job = TrainJob(id=f"{k}.-", version=k, trigger=trigger, started=self._now())
        if self._batch:
            train = [r for r in self._rows if self._split_of[r.key] == "train"]
            self._at("policy.select")
            job.rows = tuple(self.policy.select(st, train))
            self._at("policy.hyper")
            job.hyper = dict(self.policy.hyper(st))
        self._at("budget")
        refused = self.policy.budget.refuse(self._trainings, job.started)
        if refused:
            job.state, job.why = "refused", refused
            return job, []
        self._changing = True                              # from here a control's state changes
        self._attempt += 1                                 # only trainings that start get an id
        job.id = f"{k}.{self._attempt}"
        self._trainings.append(job.started)
        self._job = {"id": job.id, "version": k, "trigger": trigger, "started": job.started, "eta": None,
                     "runner": None, "pid": None, "log": None, "device": None, "last": None, "last_t": None}
        if self._batch:                                    # what a batch learner's runner reads (stage 3)
            latest = {}
            for r in self._rows:
                if self._split_of[r.key] == "calib":
                    latest[r.key] = r
            job.calib, job.budget, job.alpha, job.status = (tuple(latest.values()), self.policy.budget, self.alpha,
                                                            self._job)
        self._at("learner.train")                          # a runner that dies stops the loop here (RunnerError)
        rec = None
        if self._fps is not None:                          # replay: the recorded result (a training cut by an
            rec = self._replay_trains.pop(0) if self._replay_trains else {}   # error: what its error event kept)
        t0 = time.perf_counter()
        frozen, error, refused = None, None, None
        try:
            if rec is not None:
                if rec.get("state") == "failed":
                    error = rec.get("why")
                elif rec.get("state") == "refused":
                    refused = rec.get("why")
                else:
                    frozen = _Recorded(rec.get("fingerprint"))
            else:
                frozen = self.live.train(job) if self._batch else self.live.snapshot()
        except RunnerError:
            raise                                          # the runner itself failed: the loop stops (status "error")
        except TrainingRefused as e:                       # the time estimate is above max_train_s: not started
            refused = str(e)
        except Exception as e:  # noqa: BLE001 — a failed training is recorded; the served version keeps serving
            error = f"{type(e).__name__}: {e}"
        finally:
            self._job = None
        seconds = time.perf_counter() - t0
        n_train = len(job.rows) if self._batch else self._split_n["train"]
        base = {"job": job.id, "trigger": trigger, "rows": len(job.rows), "hyper": job.hyper, "t": job.started}
        if trigger == "train_now":
            base["control"] = True
        if refused is not None:
            job.state, job.why = "refused", refused
            return job, [self._event("train", self.served, n, state="refused", why=refused, fingerprint=None,
                                     report=None, frozen=False, **base)]
        if error is not None:
            job.state, job.why = "failed", error
            return job, [self._event("train", self.served, n, state="failed", why=error, fingerprint=None,
                                     report=None, frozen=False, **base)]
        self._at("report")
        if rec is not None:
            r = dict(rec["report"])
            r["items"] = tuple(tuple(i) for i in r.get("items") or ())
            report = TrainReport(**r)
        else:
            report = dataclasses.replace(self._report(frozen), seconds=seconds, n_train=n_train)
        self._inflight = {"job": job.id, "state": "done", "fingerprint": None, "report": dataclasses.asdict(report),
                          "extra_keys": []}                # what an error record keeps of this training
        self._at("learner.keys")                           # a learner with state names every key it trained on
        if rec is not None:
            extra = frozenset(rec.get("extra_keys") or ())
        elif callable(getattr(self.live, "keys", None)):
            extra = frozenset(self.live.keys())
        else:
            extra = frozenset()
        keys = frozenset(r.key for r in job.rows) if self._batch else frozenset(self._fed_keys)
        extra = extra - keys
        keys = keys | extra
        base["extra_keys"] = self._inflight["extra_keys"] = sorted(extra)
        self._at("policy.freeze")
        ok = bool(self.policy.freeze(self._state(), report))
        self._at("fingerprint")
        fp = frozen.fingerprint()
        self._inflight["fingerprint"] = fp
        job.state, job.fingerprint, job.seconds, job.frozen = "done", fp, report.seconds, ok
        why = None
        if not ok and hasattr(self.policy, "freeze_gates"):
            self._at("policy.freeze_gates")
            why = "; ".join(g["why"] for g in self.policy.freeze_gates(self._state(), report).values() if not g["ok"])
        ev = [self._event("train", self.served, n, state="done", why=why, fingerprint=fp,
                          report=dataclasses.asdict(report), frozen=ok, **base)]
        if ok:
            ev.append(self._freeze(k, n, frozen, keys, n_train))
        else:
            row = self._row(k, "discarded", None, None, n, self.live.seen(), n_train, keys, fp)
            row["discarded_at"] = self._split_n["train"]
            self._vs[k] = row
            self._ev_seen[k] = 0
            ev.append(self._event("discard", k, n, samples=0, why=f"freeze refused: {why or 'the policy said no'}"))
        self._inflight = None
        return job, ev

    def _freeze(self, k, n, model, keys, n_train):
        if self._fps is not None:
            model = _Recorded(self._fps.get(k))
        fp = model.fingerprint()
        self._at("served")
        srv = self._served_of(model) if self._fps is None else None
        self._at("save")
        if self.versions_dir is not None and srv is not None:
            srv.save(self.versions_dir, k)
        p = StreamPromise(**self._pkw, budget=self.budget)
        self._ev_seen[k] = 0
        row = self._row(k, "candidate", model, p, n, self.live.seen(), n_train, keys, fp)
        row["served_as"], row["served_cls"] = srv, None if srv is None else type(srv)
        self._vs[k] = row
        self.candidate = k
        return self._event("freeze", k, n, frozen_at=n, seen=self.live.seen(), n_train=n_train,
                           train_keys=len(keys), fingerprint=fp, why=None)

    def _promise_events(self) -> list:
        out = []
        for v, ver in self._vs.items():
            p = ver["promise"]
            if p is None:
                continue
            new = p.events[self._ev_seen.get(v, 0):]
            self._ev_seen[v] = len(p.events)
            out += [{**e, "version": v, "decision": self.n} for e in new]
        return out

    # --- controls
    def _control(self, event, **extra) -> dict:
        e = self._event(event, self.served, self.n, control=True, t=self._now(), **extra)
        self._write(None, None, [e])
        return e

    def pause(self, *, by=None, why=None) -> dict | None:
        """No training, freezing or promotion until resume(); serving, labels and promises go on. → the event (None
        when already paused). The candidate's trial clock runs while paused: a discard by `stale` / `trial` still
        happens (its δ_v stays spent), as do alarms and rollbacks."""
        with self._lock:
            self._refuse_if_stopped()
            if self.paused:
                return None
            self.paused = True
            return self._control("pause", by=None if by is None else str(by), why=why)

    def resume(self, *, by=None, why=None) -> dict | None:
        with self._lock:
            self._refuse_if_stopped()
            if not self.paused:
                return None
            self.paused = False
            return self._control("resume", by=None if by is None else str(by), why=why)

    def train_now(self) -> TrainJob:
        """One training now, whatever want_train says — within the budget, never while paused or with a candidate in
        trial. → the TrainJob (state "done", "failed" or "refused" with why). The call returns when the training
        is over (in the learner's runner); a RunnerError stops the loop."""
        with self._lock:
            self._refuse_if_stopped()
            k = max(self._vs) + 1
            why = ("paused" if self.paused else f"version {self.candidate} is in trial" if self.candidate is not None
                   else "a training is in flight" if self._job is not None else None)
            if why:
                job = TrainJob(id=f"{k}.-", version=k, trigger="train_now", started=self._now(), state="refused",
                               why=why)
            else:
                job, ev = self._guarded("train_now", {}, lambda: self._train("train_now", self.n))
                ev += self._promise_events()
                if ev:
                    self._write(None, None, ev)
            if job.state == "refused" and not (why is None and any(e.get("event") == "train" for e in ev)):
                self._control("train", state="refused", why=job.why)   # every control writes an event, a refused one
            return job                                                 # too (refused by the estimate: its own "train")

    def promote(self, v=None, *, force=False, by=None, why=None) -> dict:
        """Serve the candidate now: it must be certified; without force the policy's promote must agree (force skips
        the promotion rule, never the certificate); refused while paused. → the "promote" event."""
        with self._lock:
            self._refuse_if_stopped()
            return self._guarded("promote", {"version": v, "forced": bool(force), "by": by, "why": why},
                                 lambda: self._promote(v, force, by, why))

    def _promote(self, v, force, by, why):
        c = self.candidate
        if self.paused:
            raise ValueError("the loop is paused: resume() before a promotion")
        if c is None or (v is not None and v != c):
            raise ValueError(f"version {v} is not the candidate in trial (candidate: {c})")
        if not math.isfinite(self._threshold(c)):
            raise ValueError(f"version {c} is not certified: a promotion never skips the certificate")
        paired = self._paired(c)
        if not force:
            self._at("policy.promote")
        if not force and not self.policy.promote(self._state(), paired):
            raise ValueError(f"the policy does not promote version {c} now (force=True skips its rule)")
        self._changing = True
        e = self._event("promote", c, self.n, **{"from": self.served, "to": c, "l_k": paired.share_k,
                                                "l_s": paired.share_s, "samples": len(self._vs[c]["samples"]),
                                                "n_novel": paired.n_novel, "n_seen": paired.n_seen, "why": why,
                                                "by": None if by is None else str(by), "forced": bool(force),
                                                "control": True, "t": self._now()})
        e["install"] = self._serve(c, self.n)
        self._write(None, None, [e] + self._promise_events())
        return e

    def rollback(self, v, *, by=None, why=None) -> dict:
        """Serve an earlier version again (its certificate and threshold as it left them, unless an alarm came after it
        retired: then its promise restarts → "restart"; version 0 without a model: nothing alone). A candidate in trial
        is discarded (its δ_v stays spent) → the "rollback" event."""
        with self._lock:
            self._refuse_if_stopped()
            rb, ev = self._guarded("rollback", {"to": v, "by": by, "why": why},
                                   lambda: self._rollback(v, by=by, why=why, control=True))
            self._write(None, None, ev)
            return rb

    def _rollback(self, v, *, by, why, control):
        if v not in self._vs or self._vs[v]["state"] not in ("retired", "served") or v == self.served:
            raise ValueError(f"version {v} is not a retired version (served: {self.served}; "
                             f"{[x for x, ver in self._vs.items() if ver['state'] == 'retired']} retired)")
        ev = []
        self._changing = True
        if self.candidate is not None:
            ev.append(self._discard(self.candidate, self.n, "rollback"))
        frm = self.served
        install = self._serve(v, self.n)
        extra = {"from": frm, "to": v, "by": None if by is None else str(by), "why": why}
        if control:
            extra.update(control=True, t=self._now())
        rb = self._event("rollback", v, self.n, **extra)
        rb["install"] = install
        ev.append(rb)
        if self._vs[v].get("stale"):                       # certified before an alarm: starts over
            ev += self._restart(v, self.n)
        return rb, ev

    # --- the records
    def _write(self, rec, payload, events):
        q = self.question
        evs = [_json({**e, "question": q}) for e in events]
        if self._fps is not None:                          # replay: kept for the comparison
            self.log += [("stream", e) for e in evs]
            if rec is not None:
                self.log.append(("loop", _json({"question": q, "stream": rec, "loop": payload})))
            return
        if self.storage is None:
            if rec is not None:
                self.log.append(("loop", _json({"question": q, "stream": rec, "loop": payload})))
            self.log += [("stream", e) for e in evs]
            return
        from ...core.store import FORMAT
        if rec is not None:
            self.storage._append({"v": FORMAT, "kind": "loop", "question": q, "stream": _json(rec),
                                  "loop": _json(payload)})
        for e in evs:
            self.storage._append({"v": FORMAT, "kind": "stream", "stream": e})

    def _items(self):
        if self.storage is None:
            return list(self.log)
        out = []
        for s in self.storage.iter(kind=None):
            d = s.data
            if d.get("kind") == "loop" and d.get("question") == self.question:
                out.append(("loop", {"question": d["question"], "stream": d["stream"], "loop": d.get("loop")}))
            elif d.get("kind") == "stream" and (d.get("stream") or {}).get("question") == self.question:
                out.append(("stream", d["stream"]))
        return out

    def replay(self, rescore=False) -> dict:
        """Re-run the protocol from the records and compare → {"ok", "mismatches": [(n, what, why)], "decisions",
        "versions"}. The policy is the one the loop was built with (a fresh copy), trainings take their recorded
        results, controls are re-applied at their recorded place."""
        with self._lock:                                   # a snapshot: the replay itself runs outside the lock
            items = self._items()
        stored_ev = [e for k, e in items if k == "stream"]
        fps = {e["version"]: e.get("fingerprint") for e in stored_ev if e.get("event") == "freeze"}
        fresh = AdapterLoop(_Counting(), **{**self._kw, "secret": self._secret}, policy=pickle.loads(self._policy0),
                            detector=_fresh_detector(self.detector), monitor=copy.deepcopy(self._monitor0),
                            _detector_alone=True, _batch=self._batch)
        fresh._fps = fps
        fresh._replay_trains = [e for e in stored_ev if e.get("event") == "train"
                                and (e.get("state") != "refused" or e.get("job") is not None)]
        fresh._replay_trains += [e["train"] for e in stored_ev if e.get("event") == "error" and e.get("train")]
        fresh._replay_installs = [e["install"] for e in stored_ev if "install" in e]
        bad, k = [], 0

        def again(d):                                      # a control that stopped the loop: the same error again
            fresh._inject = {"call": d["call"], "why": d["why"]}
            a = d.get("args") or {}
            try:
                if d.get("of") == "train_now":
                    fresh.train_now()
                elif d.get("of") == "promote":
                    fresh.promote(a.get("version"), force=a.get("forced", False), by=a.get("by"), why=a.get("why"))
                elif d.get("of") == "rollback":
                    fresh.rollback(a.get("to"), by=a.get("by"), why=a.get("why"))
            except _Replayed:
                pass
            finally:
                fresh._inject = None
        controls = {"rollback": lambda d: fresh.rollback(d["to"], by=d.get("by"), why=d.get("why")),
                    "pause": lambda d: fresh.pause(by=d.get("by"), why=d.get("why")),
                    "resume": lambda d: fresh.resume(by=d.get("by"), why=d.get("why")),
                    "promote": lambda d: fresh.promote(d["version"], force=d.get("forced", False), by=d.get("by"),
                                                       why=d.get("why")),
                    "train": lambda d: fresh.train_now(), "error": again}
        for kind, d in items:
            if kind == "stream":
                if d.get("control") and d.get("event") in controls:
                    fresh._replay_t = d.get("t") or 0.0
                    try:
                        controls[d["event"]](d)
                    except ValueError as e:
                        bad.append((d.get("decision"), d["event"], str(e)))
                continue
            k += 1
            st, lp = d["stream"], d.get("loop") or {}
            if st["n"] != fresh.n + 1:
                bad.append((st["n"], "order", f"decision {st['n']} after {fresh.n}"))
                break
            if st.get("served") != fresh.served or st.get("candidate") != fresh.candidate:
                bad.append((st["n"], "version", f"recorded served {st.get('served')} / candidate "
                                                f"{st.get('candidate')}, rebuilt {fresh.served} / {fresh.candidate}"))
            label = lp.get("label")
            fresh._replay_t = st.get("t") or 0.0
            err = st.get("error")
            fresh._inject = None if err is None else {"call": err["call"], "why": err["why"]}
            try:
                fresh._advance(lp.get("input"), st.get("key"), lp.get("would"), st.get("score"),
                               lp.get("shadow_would"), st.get("shadow"),
                               lambda: Verdict(label, st.get("label_source") or fresh.source, st.get("label_of")),
                               fresh.source)
            except _Replayed:
                pass
            except Exception as e:  # noqa: BLE001 — an error the records do not hold
                bad.append((st["n"], "error", f"the replay raised {type(e).__name__}: {e}"))
                break
            finally:
                fresh._inject = None
            got = next(r for kk, r in reversed(fresh.log) if kk == "loop")["stream"]
            for f in sorted(set(st) | set(got)):
                if st.get(f) != got.get(f):
                    bad.append((st["n"], f, f"recorded {st.get(f)}, rebuilt {got.get(f)}"))
        rebuilt_ev = [e for kk, e in fresh.log if kk == "stream"]
        if rebuilt_ev != stored_ev:
            bad.append((None, "stream records", f"{len(stored_ev)} stored, {len(rebuilt_ev)} rebuilt, or they differ"))
        if self.versions_dir is not None:
            bad += self._check_files(fps, items, rescore)
        return {"ok": not bad, "mismatches": bad, "decisions": k, "versions": len(fps)}

    def _check_files(self, fps, items, rescore):
        bad, models = [], {}
        for v, fp in fps.items():
            cls = (self._vs.get(v) or {}).get("served_cls")
            if cls is None:
                bad.append((None, "version file", f"version {v}: no Served to load it with"))
                continue
            srv = (self._vs.get(v) or {}).get("served_as")
            kw = ({"part": srv.part} if getattr(srv, "part", None) is not None
                  and "part" in inspect.signature(cls.load).parameters else {})   # a LoRA version scores on its part
            try:
                models[v] = cls.load(self.versions_dir, v, expect_fp=fp, **kw)
            except FileNotFoundError as e:
                bad.append((None, "version file", f"version {v}: {e.filename} is missing"))
            except Exception as e:  # noqa: BLE001 — a tampered or foreign file
                bad.append((None, "version file", f"version {v}: {e}"))
        if self._vs[0]["model"] is not None:
            models[0] = self._vs[0]["model"]               # version 0: the model given as initial=
        if rescore:
            for kind, d in items:
                if kind != "loop":
                    continue
                st, lp = d["stream"], d.get("loop") or {}
                for v, key in ((st.get("served"), "score"), (st.get("candidate"), "shadow")):
                    if v in models:
                        got = models[v].score(lp.get("input"))[1]
                        if _num(got) != st.get(key):
                            bad.append((st["n"], "rescore", f"version {v}: recorded {st.get(key)}, scores {got}"))
        live = {id(x) for ver in self._vs.values() for x in (ver.get("model"), ver.get("served_as")) if x is not None}
        live_fp = {ver["model"].fingerprint() for ver in self._vs.values() if ver.get("model") is not None}
        for v, mdl in models.items():                      # the copies loaded for the check: unstaged again, unless
            if id(mdl) not in live and mdl.fingerprint() not in live_fp:   # a live version shares their adapter
                self._release(mdl)
        return bad

    # --- what it says
    def versions(self) -> list:
        with self._lock:
            return self._versions()

    def _versions(self) -> list:
        out = []
        for v, ver in sorted(self._vs.items()):
            p = ver["promise"]
            out.append({"version": v, "state": ver["state"], "certified": p is not None and not math.isinf(p.threshold),
                        "threshold": None if p is None or math.isinf(p.threshold) else p.threshold,
                        "n": None if p is None else p.n, "v_index": None if p is None else p.v,
                        "delta_v": None if p is None else p.delta_v, "frozen_at": ver["frozen_at"],
                        "seen": ver["seen"], "n_train": ver["n_train"], "train_keys": len(ver["keys"]),
                        "samples": len(ver["samples"]), "discarded_at": ver["discarded_at"],
                        "fingerprint": ver["fingerprint"],
                        "paired": dataclasses.asdict(self._paired(v)) if v == self.candidate else None})
        return out

    def status(self) -> dict:
        """{"state", "error", "served", "candidate", "n", "labels": {"train", "calib", "holdout"}, "trainings",
        "budget": {...}, "job", "paused", "alarms", "mode", "guarantee", "delta_spent"}. state: "error" (a hook
        raised: the loop stopped; error: {"n", "where", "why"}), "training", "paused" or "running".

        Takes no lock: it answers while a step (and the in-process training inside it) runs on another thread, and
        `job` shows that training. Its fields are read one by one, so two of them may straddle a decision."""
        b, job, err = self.policy.budget, self._job, self.error
        state = ("error" if err is not None else "training" if job is not None else "paused" if self.paused
                 else "running")
        return {"state": state, "error": None if err is None else dict(err), "served": self.served,
                "candidate": self.candidate, "n": self.n, "labels": dict(self._split_n),
                "trainings": len(self._trainings),
                "budget": {"max_trainings": b.max_trainings, "per_day": b.per_day,
                           "refuse": b.refuse(list(self._trainings), self._now())},
                "job": None if job is None else dict(job), "paused": self.paused,
                "alarms": len(self._alarms), "mode": "stream",
                "guarantee": "anytime stream certificate per version (StreamPromise on post-freeze coin samples)",
                "delta_spent": self.budget.spent}

    def fed(self, v) -> list:
        """The decisions whose samples reached version v's promise, in order."""
        return list(self._vs[v]["fed"])

    @property
    def delta_spent(self) -> float:
        return self.budget.spent

    def __repr__(self):
        return (f"{type(self).__name__}({self.question!r}, served {self.served}, candidate {self.candidate}, "
                f"n={self.n}, labels {self.live.seen()})")


__all__ = ["AdapterLoop", "S2_SOURCES", "Verdict", "content_key", "ticket_verdict"]
