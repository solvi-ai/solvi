"""The adapter loop: frozen versions, each with its own stream certificate, promoted under δ-spending, driven by a
`LoopPolicy` (experimental; the LoRA learning loop, stages 2–5).

    from solvi.experimental.stream import AdapterLoop, DefaultPolicy, HashedTextTrainer

    loop = AdapterLoop(HashedTextTrainer(), question="intent", policy=DefaultPolicy(), max_risk=0.05, secret=key)
    out = loop.step(text, oracle)          # oracle: () → the label (None: missing), called only when one is needed
    loop.pause() / loop.resume()           # no training, freezing or promotion while paused; labels and promises go on
    loop.train_now(wait=False)             # → TrainJob: ignores want_train, not the budget; returns at once
    loop.close()                           # cancels a training in flight; step and the controls refuse after it
    loop.promote(force=False) / loop.rollback(v)
    loop.versions(), loop.status(), loop.replay()
    loop.save(dir) / AdapterLoop.load(dir, learner, ...)   # the checkpoint (stream.checkpoint)

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
  comparison, seen ones for the memorisation probe (`DefaultPolicy(novel_only=False)`, for controls: every sample in
  the comparison; it weakens the guarantee). A missing label counts as wrong.
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
`train_now(wait=None)` (its "train_start" event; wait=True returns when the training is over and applied),
`close()`, `promote(v=None, force=False)`
(force skips the policy's promotion rule — novel count, min_share, margin, probe — never the certificate), and
`rollback(v)`. A rollback is allowed while paused; a discard and an alarm go on while paused.

Records: as the lane's, plus per decision its content key, its split, and the wall clock `t` (the budget reads the
clock; replay reads the recorded one), and per training a "train" event (job, trigger, rows, settings, fingerprint, the
TrainReport, whether it froze, where it started and where it was applied). Replay re-runs the protocol from the records with a fresh copy of the policy as it was
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
carries the calib labels (the latest per key), the policy's `Budget`, the loop's α, the live `status()["job"]` dict,
which a runner fills in (runner, pid, log, device, eta, last line), and `cancel`, an event the loop sets to drop it. What
a training ends with: a frozen model (done), `TrainingRefused` (its time estimate is above `max_train_s`: a "train" event
with state "refused", counted against the budget), any other exception (state "failed"), or `RunnerError` — the runner
itself failed (the process died or went silent) — which stops the loop like a hook that raises (status "error", the
hook "learner.train"; replay exact). `initial=` that has `install` / `uninstall` (a `LoraVersion.of_part`) is version
0's `Served` too, so a rollback to 0 puts the part as attached back. Replay takes the recorded install fingerprints
(the System is not the replay's).

Deferred apply (stage 3b, design §3.2; the author's decision 7): a batch learner's training starts at a decision (or in
`train_now`) — a "train_start" event there — and runs in a thread of its own, its runner beside it; the loop goes on
deciding on the served version and no decision waits for it. The pre-freeze report is computed in that thread too, on
the calib and held-out labels and the served version of the moment the training started; it stops between two labels
when the training is cancelled or the served version changed. When the served version changed during the training (a
rollback), the result is not gated on that stale comparison: the "train" event has state "stale" and the candidate is
discarded at birth and released ("served_at_start" in "train_start" and "train" names the version compared with).
Its result is applied on the
first decision after it is in (not while paused: it waits for resume): the gates, then freeze or discard, a failure, a
refusal or a stop on a `RunnerError` — the stage-3 paths — recorded at that decision: the "train" event carries
`started_at` (the decision it started at) and `applied` ("decision", or "train_now" for `train_now(wait=True)`). Replay
applies each recorded result at the decision (or in the control) the record names, with no clock: the position is the
record's. One training at a time; the budget counts it when it starts. `wait=True` (the constructor's, or
`train_now(wait=True)`) runs the training inside the step or the control, as stage 3 did (scripts, tests); an online
learner's snapshot is always taken in the step. `close()` (a "close" event) cancels the training in flight — a
subprocess is killed, a thread runner stops waiting — and so does an error stop; `wait_training(timeout)` waits for a
result outside every lock.

Lock order: the part's version lock (`part=`, else the learner's or `initial`'s `part`), then the loop's — in `step` and
every control. A decision of the part that reaches the loop under its own version lock (stage 5's hook) takes them in
the same order; the loop never waits for the part's lock while it holds its own (shadow scoring, a swap). The cost:
while a step runs, the part's own decisions wait for it (with `wait=True`, for the whole training).

Eviction: past `policy.budget.max_versions_kept` versions held (served and retired, version 0 counted), the oldest
retired ones are evicted after each change of the served version — "evict" events, `Served.release()` called, state
"evicted", no longer a rollback target; their files stay for replay. The served version, the rollback target (the one
served before it) and version 0 are never evicted, so up to three can stay above a limit below 3.

Checkpoint (stage 4, `stream.checkpoint`): `save(dir, cancel=False)` writes the whole state — versions and their
files, promises and the detector (`dump`), policy, budget, rate, the label store `labels.jsonl`, the records — into a
new folder, then switches `dir/CURRENT` to it atomically; `AdapterLoop.load(dir, learner, ...)` checks the signed
manifest and every file's hash, rebuilds the loop, installs the served version and feeds the fresh learner its train
labels again. A training in flight refuses the save unless `cancel=True` (`cancel_training()`, a "cancel" control
event). A loaded loop continues with the same decisions and records as the loop it was saved from, and replays exactly.

Attach (stage 5): `AdapterLoop.attach(dispatcher, learner, ...)` (stream mode) or `AdapterLoop.attach(system, ...)`
(label mode; `mode="label"`), or `Dispatcher(..., loop=loop)` / `loop.attach_to(host)` for a loop built by hand, and
`detach()`. Stream mode: every `dispatcher.ask` is one decision of the loop, split around the dispatch — `_pre` (the
served and candidate scores, the coin, whether the loop's rule answers alone) before System 1 is asked, `_post` (the
step, with the slow path's accepted answer as the label) after; the dispatch escalates what the loop's rule does not
answer alone and checks what its coin draws; the Dispatched records `learning` (the version served, its fingerprint,
the coin, the loop's events); `Dispatcher.replay` installs each decision's version for its replay (`serving(v)`).
Neither hook runs inside the part's call, and a Dispatcher's loop decisions run one at a time under its `_learn_lock`,
taken before the part's lock and the loop's and never while either is held: the order is learn → part → loop. A label
the policy refuses (the slow path's "s2" under the default policy) is a missing label for the loop. `System.teach` and
`outcome` with a loop attached learn nothing at once (the part is the loop's): in label mode they are the loop's
labels (`label(x, y)`), in stream mode they are only stored. Label mode has no coin, no promise and no alarm: every
call is a labelled step, a version serves at its own act_guard threshold on the calib labels, and a frozen candidate
with such a threshold is promoted on the next label when `policy.promote(st, None)` says so (the default: the
pre-freeze gates decided; `DefaultPolicy(pre_freeze=False)` is refused in label mode; `status()["guarantee"]` names
the fixed-set guarantee). After a promotion a watch scores the served version and the one served before it on each
calib or held-out label and rolls back when `policy.watch(st, Watch)` says so (`DefaultPolicy`:
`policy.watch_rollback`, an anytime-valid e-process on the discordant labels and the margin; the base `LoopPolicy`
never rolls back); the target's scores are recorded (a train label's target is not scored), so the watch and its
e-process replay. Attached to a Dispatcher, labels the policy refuses are counted (`status()["labels"]["refused"]`)
and `warn_refused` of them in a row with none accepted warn once. The automatic
promotion, in both modes, needs the candidate certified, as the `promote()` control does. The candidate is scored in
shadow only while its grid is not fixed and on coin-sampled decisions. The controls take the Dispatcher's
`_learn_lock` first (`_locked`), so a control waits for the decision in flight. Refused: a question answered by a
combination or a part that is a member of one (`combinations_of`), a combination built over the part while a loop
serves it (`combine.served_by_loop`), a Dispatcher with its own `promise=`, a second loop on a System, a stream-mode
attach whose `DefaultPolicy` takes no label from the slow path (no label source).

Labels are kept in memory, inputs included (`LabelRow.x`): with dict
inputs the memory grows with the label count (the checkpoint writes them to `labels.jsonl`). `status()` takes no lock.
Unmeasured: no stand has run on this loop."""
from __future__ import annotations

import contextlib
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
from .policy import (DefaultPolicy, LabelRow, LoopPolicy, LoopState, Paired, TrainReport, VersionInfo, Watch,
                     watch_eprocess)
from .runner import RunnerError, TrainingCancelled, TrainingRefused
from .sampling import Rate, coin_message, coin_value
from .serving import BaseVersion, as_served

S2_SOURCES = ("s2", "verified")
ACTIONS = ("recertify", "ignore")
KEPT = ("served", "retired")                       # the states max_versions_kept counts
MODES = ("stream", "label")
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


def _watch_copy(w):
    """The watch's counts as a fresh dict (its e-process's log wealths "lw" a list of its own)."""
    out = dict(w)
    out["lw"] = list(w["lw"])
    return out


def _watch_of(w) -> Watch:
    """The Watch the policy reads: the counts and the e-process's log wealth (`watch_eprocess()` restored from
    "lw")."""
    mix = watch_eprocess().restore({"bet": "mixture", "lw": w["lw"], "n": w["better"] + w["worse"]})
    return Watch(target=w["target"], n=w["n"], better=w["better"], worse=w["worse"], total=w["total"],
                 log_e=mix.log_wealth)


def _num(x):
    """A score as recorded: None for no score or a non-finite one (the certificate reads both as -inf)."""
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _json(x):
    from ...core.store import plain
    return json.loads(json.dumps(x, default=plain))


def _alone(score, thr) -> bool:
    return score is not None and math.isfinite(thr) and math.isfinite(float(score)) and score >= thr


def _mirror_budget(budget, v):
    """A private `DeltaBudget` with the delta and schedule of `budget` whose next() hands out index v: the comparison
    certificate's δ_v is the candidate's own, and the loop's budget is not moved (nothing is spent from it)."""
    b = DeltaBudget(budget.delta, budget.schedule)
    b.v = int(v) - 1
    return b


class _Replayed(Exception):
    """A recorded error raised again by replay at the place it was recorded."""

    def __init__(self, why):
        super().__init__(why)
        self.why = why


class _Stale(Exception):
    """The served version changed while a training ran: its pre-freeze comparison would be against a version no longer
    served."""

    def __init__(self, at_start, now):
        super().__init__(f"the served version changed during the training (v{at_start} → v{now}): the pre-freeze "
                         "comparison is stale")
        self.at_start, self.now = at_start, now


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


def _host_system(host):
    """A Dispatcher's System 1, or the System itself."""
    from ...core.dispatch import Dispatcher
    from ...core.system import System
    if isinstance(host, Dispatcher):
        return host.system
    if isinstance(host, System):
        return host
    raise TypeError(f"attach to a Dispatcher (stream mode) or a System (label mode), not {type(host).__name__}")


def _decision_part(system, question, part=None):
    """The model-backed decision part that answers `question` (None: a rule without one). Refused: a combination, and
    a part that is a member of one (it reads the part's internals without its version lock)."""
    from ...core.deciders.combine import Combination, combinations_of
    from ...core.deciders.gate import decision_of
    found = decision_of(system.catalog, question) if part is None else part
    if isinstance(found, Combination):
        raise ValueError(f"{question!r} is answered by a combination ({type(found).__name__}): a loop serves one "
                         "decision part; attach to a part that is in no combination")
    if found is not None:
        combos = combinations_of(found)
        if combos:
            raise ValueError(f"the part {getattr(found, '__name__', found)!r} is a member of a combination "
                             f"({', '.join(sorted({type(c).__name__ for c in combos}))}), which reads it without its "
                             "version lock: a loop cannot serve it")
    return found


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
    wall clock (None: time.time); wait: True runs every training inside the step or the control (stage 3's blocking
    form); part: the part whose version lock the loop takes before its own (None: the learner's or initial's `part`) —
    from construction on, every decision of that part takes this RLock, whether or not a version is ever swapped;
    warn_refused: attached to a Dispatcher, after this many labelled decisions in a row whose label the policy refused
    (none accepted in between) a `UserWarning` says the loop learns nothing, once (None: never; the count is
    `status()["labels"]["refused"]`); trace_candidate: while a candidate is on trial, every decision's records carry
    its score and would-be answer ("candidate_score" in the stream record, "candidate_would" in the loop payload;
    the existing "shadow" is filled only where the protocol reads it) — for offline replays of promotion rules; the
    protocol never reads them (no behaviour change), it costs one candidate scoring per decision; the keys are absent
    with no candidate or with the option off (default)."""

    def __init__(self, learner, *, question, policy=None, max_risk=None, max_error=None, delta=0.05, initial=None,
                 detector=None, monitor="auto", probation=300, source="human", secret=None, seed=0, sampling=None,
                 promise_options=None, storage=None, versions_dir=None, system=None, feature="text", key=None,
                 served=None, clock=None, wait=False, part=None, mode="stream", warn_refused=200,
                 trace_candidate=False, _detector_alone=False, _batch=None):
        global _warned
        if mode not in MODES:
            raise ValueError(f"mode is 'stream' (decisions through step, a promise per version) or 'label' (labels "
                             f"only, the pre-freeze gates decide), not {mode!r}")
        if mode == "label" and (detector is not None or (monitor is not None and monitor != "auto")):
            raise ValueError("label mode has no coin samples: no detector and no score monitor (its guarantee is "
                             "act_guard's fixed-set one on the calib labels)")
        policy = DefaultPolicy() if policy is None else policy
        if not isinstance(policy, LoopPolicy):
            raise TypeError(f"policy is a LoopPolicy (stream.policy), not {type(policy).__name__}")
        if mode == "label" and isinstance(policy, DefaultPolicy) and not policy.pre_freeze:
            raise ValueError("label mode with pre_freeze=False measures nothing: the pre-freeze gates are the only "
                             "measurement before a version is served there (no stream certificate) — keep "
                             "pre_freeze=True")
        try:
            self._policy0 = pickle.dumps(policy)           # replay starts from the policy as built; it must pickle
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"the policy must pickle (the loop's replay and checkpoint keep it): {e}") from e
        self._kw = dict(question=question, max_risk=max_risk, max_error=max_error, delta=delta, initial=initial,
                        probation=probation, source=source, seed=seed, sampling=sampling,
                        promise_options=promise_options, feature=feature, key=key, served=served, wait=wait,
                        mode=mode, warn_refused=warn_refused, trace_candidate=bool(trace_candidate))
        self.trace_candidate = bool(trace_candidate)
        if warn_refused is not None and (isinstance(warn_refused, bool) or int(warn_refused) < 1):
            raise ValueError("warn_refused is a number of labelled decisions ≥ 1, or None (never warn)")
        self.warn_refused = None if warn_refused is None else int(warn_refused)
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
        self.mode = mode
        self.host = None                                   # the Dispatcher or System it is attached to (attach)
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
        p0 = None if initial is None or mode == "label" else StreamPromise(**self._pkw, budget=self.budget)
        self._vs = {0: self._row(0, "served", initial, p0, 0, 0, 0, frozenset(),
                                 None if initial is None else initial.fingerprint())}
        self._vs[0]["fixed"] = getattr(initial, "act_threshold", None)
        self._rows: list = []                              # every accepted label (LabelRow)
        self._split_of: dict = {}                          # key → the split it first got (never changes)
        self._split_n = {"train": 0, "calib": 0, "holdout": 0}
        self._split_keys = {"train": 0, "calib": 0, "holdout": 0}   # distinct labelled keys by split (as the report)
        self._labelled: set = set()                        # the keys with at least one accepted label
        self._fed_keys: set = set()                        # keys of the train labels the learner took
        self._holdout: deque = deque(maxlen=10_000)        # the served version's (alone, wrong) on held-out labels
        self._watch = None                                 # label mode after a promotion: the Watch counts (a dict)
        self._refused = {"n": 0, "run": 0, "warned": False}   # labels the policy refused on the attached path
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
        self._seen0 = learner.seen()                       # the learner as given (a checkpoint feeds it from here)
        self.error = None                                  # {"n", "where", "why"} once a hook raised: the loop stops
        self._hk, self._where, self._changing = 0, "loop", False
        self._inject = None                                # replay: {"call", "why"} of a recorded error
        self._inflight = None                              # the training being applied, for an error record
        self.wait = bool(wait)                             # True: a training runs inside the step / train_now
        self._run = None                                   # the training in flight: job, thread, result
        self.closed = False
        self.detached = False
        self._detach_run = None
        if part is None:
            part = getattr(learner, "part", None) or getattr(initial, "part", None)
        self.part = part
        if part is not None:                               # lock order: the part's version lock, then the loop's
            from ...core.deciders.part import version_lock
            self._part_lock = version_lock(part)
        else:
            self._part_lock = None

    @staticmethod
    def _row(v, state, model, promise, frozen_at, seen, n_train, keys, fp):
        return {"version": v, "state": state, "model": model, "promise": promise, "frozen_at": frozen_at,
                "seen": seen, "n_train": n_train, "keys": keys, "fingerprint": fp, "samples": [], "fed": [],
                "discarded_at": None, "served_cls": None, "cmp": None}

    def _comparison(self, p):
        """The served version's comparison certificate for a candidate whose promise is p: a `StreamPromise` with the
        loop's settings and p's (v, δ_v), on a private mirror of the budget (`_mirror_budget`) — it spends nothing from
        the loop's budget. It reads the served version's score on exactly the decisions and samples p reads (same
        grid rule, same samples, same δ), so the promotion compares the two versions certified the same way on the same
        data. It only ranks: nothing is ever answered at its threshold, so it is outside the δ accounting (see
        `_paired`)."""
        return StreamPromise(**self._pkw, budget=_mirror_budget(self.budget, p.v))

    @contextlib.contextmanager
    def _locked(self):
        """Attached to a Dispatcher: its per-decision lock (`_learn_lock`) first, so a control waits for the decision in
        flight — the slow path's call included — and never lands between the loop's read before System 1 and System
        1's call. Then the part's version lock (when the loop has a part), then the loop's: the order a decision of the
        part that reaches the loop takes them too, so no path holds the loop's lock and waits for the part's (shadow
        scoring, the report, a swap). The training thread never takes this."""
        ll = getattr(self.host, "_learn_lock", None)
        pl = self._part_lock
        with ll if ll is not None else contextlib.nullcontext():
            with pl if pl is not None else contextlib.nullcontext():
                with self._lock:
                    yield

    # --- what the hooks read
    def _now(self) -> float:
        """The wall clock: one reading per decision (the recorded one during replay)."""
        if self._fps is not None:
            return self._replay_t
        return self._step_t if self._step_t is not None else float(self._clock())

    def _threshold(self, v):
        if self.mode == "label":                           # label mode: the version's own fixed-set threshold
            t = self._vs[v].get("fixed")
            return math.inf if t is None else float(t)
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
                         keys=dict(self._split_keys),
                         versions=tuple(vs), holdout=tuple(self._holdout), alpha=self.alpha,
                         alarms=tuple(dict(a) for a in self._alarms), served_history=tuple(self._history),
                         served_since=self._served_since, trainings=tuple(self._trainings),
                         job=None if self._job is None else dict(self._job), now=self._now(), paused=self.paused,
                         mode=self.mode, promises=promises,
                         detector=None if self.detector is None else self.detector.state())

    def _paired(self, c) -> Paired:
        """The candidate's Paired (see the module docstring). A policy with `novel_only` False
        (`DefaultPolicy(novel_only=False)`, a control) gets every sample, seen ones too, in the comparison fields; the
        seen fields are counted as always.

        The symmetric comparison (1.4.0): alone_served / wrong_served are the served version's answers at the threshold
        of its comparison certificate (`_comparison`: built on exactly the candidate's post-freeze samples, the same
        grid rule and the candidate's δ_v), so both versions are certified the same way on the same data and a fresh
        candidate does not lose on certificate age (the served version's own certificate has read more samples and
        loosened further). alone_served_own / wrong_served_own are the old comparison, at the served version's own
        threshold in force now, kept for the records. The comparison is a decision rule, not a guarantee: the served
        version always answers at its own threshold, a promoted candidate at its own certificate's, and every
        certificate that ever serves holds under the loop's one δ budget whatever this rule decides. With no comparison
        certificate, or while it is not certified, the comparison falls back to the served version's own threshold
        (`_thr_served`, conservative: an uncertified comparison would put ℓ_s at 0 and the margin would pass on the
        candidate alone); Paired.thr_s records which one was used. Deterministic from the recorded samples: replay gives
        the same numbers."""
        thr_k, thr_own = self._threshold(c), self._threshold(self.served)
        thr_s, rule = self._thr_served(c)
        novel_only = bool(getattr(self.policy, "novel_only", True))   # a policy pickled before the switch: on
        n_nov = a_nov = w_nov = a_srv = w_srv = n_seen = a_seen = w_seen = a_own = w_own = 0
        for sk, ss, wk, ws, seen in self._vs[c]["samples"]:
            ak = _alone(sk, thr_k)
            if seen:
                n_seen += 1
                a_seen += ak
                w_seen += ak and wk
            if not seen or not novel_only:
                n_nov += 1
                a_nov += ak
                w_nov += ak and wk
                if _alone(ss, thr_s):
                    a_srv += 1
                    w_srv += ws
                if _alone(ss, thr_own):
                    a_own += 1
                    w_own += ws
        return Paired(n_nov, a_nov, w_nov, a_srv, n_seen=n_seen, alone_seen=a_seen, wrong_seen=w_seen,
                      wrong_served=w_srv, alone_served_own=a_own, wrong_served_own=w_own, thr_s=rule)

    def _thr_served(self, c):
        """The served version's threshold for candidate c's comparison → (threshold, rule). "cmp": its comparison
        certificate's, once certified. "own": its own threshold in force now, while the comparison is not certified or
        absent (a checkpoint saved before 1.4.0 with a candidate in trial) — the conservative fallback: an uncertified
        comparison (+inf) would read ℓ_s as 0. "no_model": the served version has no model (version 0 without one);
        its scores are None and it never answers alone, as before 1.4.0."""
        thr_own = self._threshold(self.served)
        if self._vs[self.served]["model"] is None:
            return thr_own, "no_model"
        cmp = self._vs[c].get("cmp")
        if cmp is not None and math.isfinite(cmp.threshold):
            return cmp.threshold, "cmp"
        return thr_own, "own"

    # --- the per-decision step
    def step(self, x, oracle=None, *, source=None, key=None) -> dict:
        """One input → {"n", "answer", "by", "served", "candidate", "sampled", "label", "label_source", "events"}.
        oracle: () → the label (None: missing), called only when the decision needs one (not answered alone, or
        drawn by the coin); source: its source (default the loop's); key: the input's content key (None: the loop's
        key function)."""
        with self._locked():
            self._refuse_if_stopped()
            s = self._vs[self.served]
            would_s, score_s = s["model"].score(x) if s["model"] is not None else (None, None)
            sampled = self._draw(self.n + 1, score_s, self._threshold(self.served))[2]
            would_k, score_k = self._shadow(x, sampled)
            if oracle is None:
                def oracle():
                    raise ValueError("this decision needs a label: step(x, oracle)")
            k = self._key(x) if key is None else str(key)
            return self._advance(x, k, would_s, score_s, would_k, score_k, oracle,
                                 self.source if source is None else source, watch=self._watch_scores(x, k),
                                 trial=self._trial(x, would_k, score_k))

    def _shadow(self, x, sampled):
        """The candidate's (would, score) on x when the step reads it: while its promise's grid is not fixed yet (the
        grid reads the scores) or on a coin-sampled decision (its paired sample); (None, None) otherwise — and always
        in label mode (no promise, no sample) or with no candidate."""
        c = self._vs.get(self.candidate)
        if c is None or c["model"] is None or c["promise"] is None:
            return None, None
        if c["promise"].grid is None or sampled:
            return c["model"].score(x)
        return None, None

    def _trial(self, x, would_k, score_k):
        """trace_candidate: the candidate's (would, score) on x for the records (the shadow's when the step read it,
        else scored here) — None with the option off or no candidate. Read by no rule."""
        if not self.trace_candidate or self.candidate is None:
            return None
        if score_k is not None:
            return would_k, score_k
        m = self._vs[self.candidate]["model"]
        return m.score(x) if m is not None else (None, None)

    def _watch_scores(self, x, key):
        """Label mode with a watch running: the rollback target's (would, score) on x (None: no watch). Only a calib or
        held-out key is scored (the watch counts no train label): a train key gets (None, None). The split is read as
        `_split` reads it, without recording it (the policy's split is a pure function of the key)."""
        if self.mode != "label" or self._watch is None:
            return None
        sp = self._split_of.get(key)
        if sp is None:
            sp = self.policy.split(self._state(), key)
        if sp not in ("calib", "holdout"):
            return None, None
        m = self._vs[self._watch["target"]]["model"]
        return m.score(x) if m is not None else (None, None)

    def label(self, x, y, *, source=None, key=None) -> dict:
        """Label mode: one label (x, y) → as `step` (by "oracle"). The served version is scored on it (the held-out
        window), a train label reaches the learner, and the policy may start a training, freeze or promote here. An
        attached System calls it from `teach` / `outcome`."""
        if self.mode != "label":
            raise ValueError("label() is label mode's (a System alone); in stream mode labels come with the decisions "
                             "(step, or the Dispatcher's slow path)")
        return self.step(x, lambda: y, source=source, key=key)

    def _input(self, system, state):
        """What the loop scores for an input of the System: the decision part's input text (as `System.teach` gives
        it to the part), or the input itself when the question has no decision part."""
        vals = system._state(state)[0]
        if self.part is None:
            return vals
        if not all(f in vals for f in self.part.facts):
            vals = system.facts_for(vals)
        return self.part.text_of(vals)

    def _taught(self, system, state, correct, source):
        """A label from the attached System (`teach`, `outcome`): label mode — one `label` call; stream mode — not
        taken (the loop's labels come with its decisions, after the promises read them) → the label's step or None."""
        if self.mode != "label":
            return None
        return self.label(self._input(system, state), correct, source=source)

    # --- attached to a Dispatcher (stream mode): the decision is split around the dispatch, outside the part's call
    def _pre(self, x) -> dict:
        """Before the Dispatcher decides: the served version's score, the next decision's number, coin and whether the
        loop's rule answers alone, and the candidate's score in shadow when the step reads it (`_shadow`: until its
        grid is fixed, then on coin-sampled decisions only) — read under the Dispatcher's lock, the part's, then the
        loop's, nothing changed."""
        with self._locked():
            self._refuse_if_stopped()
            if self.mode != "stream":
                raise ValueError("a label-mode loop takes no decisions")
            s = self._vs[self.served]
            would_s, score_s = s["model"].score(x) if s["model"] is not None else (None, None)
            n = self.n + 1
            prob, rho, sampled, alone = self._draw(n, score_s, self._threshold(self.served))
            would_k, score_k = self._shadow(x, sampled)
            return {"n": n, "served": self.served, "candidate": self.candidate, "hist": len(self._history),
                    "scores": (would_s, score_s, would_k, score_k), "key": self._key(x), "sampled": sampled,
                    "trial": self._trial(x, would_k, score_k),
                    "alone": alone, "probation": prob, "rho": rho, "fingerprint": s["fingerprint"]}

    def _post(self, pre, x, label, source, label_of=None) -> dict:
        """After the Dispatcher decided: the decision's step with the label it got (None: none — a decision the loop
        needed a label for and the dispatch did not label is a missing label). When a control changed the loop between
        `_pre` and here, the scores are read again (the step is the loop's record of the decision either way)."""
        with self._locked():
            self._refuse_if_stopped()
            same = ((self.n + 1, self.served, self.candidate, len(self._history))
                    == (pre["n"], pre["served"], pre["candidate"], pre["hist"]))
            if same:
                would_s, score_s, would_k, score_k = pre["scores"]
                trial = pre.get("trial")
            else:
                s = self._vs[self.served]
                would_s, score_s = s["model"].score(x) if s["model"] is not None else (None, None)
                would_k, score_k = self._shadow(x, self._draw(self.n + 1, score_s, self._threshold(self.served))[2])
                trial = self._trial(x, would_k, score_k)
            src = self.source if source is None else source
            out = self._advance(x, pre["key"], would_s, score_s, would_k, score_k,
                                lambda: Verdict(label, source, label_of), src, lenient=True, trial=trial)
            out["same"] = same
            return out

    # --- attach / detach (stage 5)
    @classmethod
    def attach(cls, host, learner=None, *, question=None, dir=None, policy=None, initial="auto", part=None,
               **kw) -> "AdapterLoop":
        """Build a loop for `question` and attach it to `host`: a Dispatcher (stream mode: every decision of
        `dispatcher.ask` goes through the loop) or a System alone (label mode: the labels of `teach` / `outcome` are the
        loop's; no coin, no stream promise). learner: None — a `LoraLearner` on the question's model-backed decision
        part (needs torch and dir=); any Learner (online or batch) otherwise. initial: version 0's model ("auto": the
        learner's `version0()` when it has one, else none). part: the decision part (None: found from the question).
        dir: the loop's folder (the default learner's and versions_dir). The other keywords are the constructor's.
        Refused: a question answered by a combination (Cascade, Vote, Route) or by a part that is a member of one, a
        rule with no decision part when no learner is given, a Dispatcher with its own stream certificate."""
        system = _host_system(host)
        mode = "stream" if system is not host else "label"
        if question is None:
            question = getattr(host, "question", None)
            if question is None:
                if len(system.questions) != 1:
                    raise ValueError(f"the System asks {sorted(system.questions)}: say which with question=")
                question = next(iter(system.questions))
        found = _decision_part(system, question, part)
        if learner is None:
            if found is None:
                raise ValueError(f"{question!r} is not answered by a model-backed decision part (a rule): give a "
                                 "learner= that serves on the System")
            if dir is None:
                raise ValueError("the default LoRA learner keeps its files in dir=")
            from .lora_learner import LoraLearner
            learner = LoraLearner(found, dir)
        if isinstance(initial, str) and initial == "auto":
            v0 = getattr(learner, "version0", None)
            initial = v0() if callable(v0) else None
        if dir is not None:
            kw.setdefault("versions_dir", dir)
        loop = cls(learner, question=question, policy=policy, initial=initial, system=system, part=found, mode=mode,
                   **kw)
        loop.attach_to(host)
        return loop

    def attach_to(self, host) -> "AdapterLoop":
        """Attach this loop to a Dispatcher (stream mode) or a System (label mode) → self. The checks of `attach`;
        also refused: a loop already attached, closed or stopped, a host that has a loop (or, a System, an update
        target), a loop that serves on another System. `Dispatcher(..., loop=)` calls it."""
        from ...core.dispatch import Dispatcher
        system = _host_system(host)
        with self._locked():
            self._refuse_if_stopped()
            if self.host is not None:
                raise ValueError("this loop is attached already: detach() first")
            if isinstance(host, Dispatcher):
                if self.mode != "stream":
                    raise ValueError("a Dispatcher's decisions go through a stream-mode loop (mode='stream')")
                if host.question != self.question:
                    raise ValueError(f"the dispatcher asks {host.question!r}, the loop learns {self.question!r}")
                if host._stream is not None:
                    raise ValueError("the dispatcher has its own stream certificate (promise=): with a loop the "
                                     "loop's per-version promises are the certificate — build it without promise=")
                if getattr(host, "loop", None) is not None:
                    raise ValueError("the dispatcher has a loop attached already")
                self._refuse_no_label_source(host)         # before any state is set: a refusal leaves nothing attached
            else:
                if self.mode != "label":
                    raise ValueError("a System alone gives labels, not decisions: a label-mode loop (mode='label'; "
                                     "AdapterLoop.attach builds one)")
                if self.question not in system.questions:
                    raise ValueError(f"no question {self.question!r} in the System")
            if getattr(system, "adapter_loop", None) is not None:
                raise ValueError("the System has a loop attached already (directly or through a Dispatcher)")
            if getattr(system, "_learning", None) is not None:
                raise ValueError("the System has an update target (LearningTarget) attached: one learner at a time")
            if self.system is not None and self.system is not system and self.system is not self.part:
                raise ValueError("the loop serves its versions on another System")
            found = _decision_part(system, self.question, self.part)
            if found is not None and self.part is not None and found is not self.part:
                raise ValueError("the loop's part is not the one that answers the question in this System")
            if found is not None and self.part is None:
                if self.n or self._run is not None or self.served != 0:
                    raise ValueError("the question has a decision part the loop has no lock for: attach before the "
                                     "first decision, or build the loop with part=")
                from ...core.deciders.part import version_lock
                self.part, self._part_lock = found, version_lock(found)
            if self.system is None:
                self.system = system
                if isinstance(self._base, BaseVersion) and self.served == 0:
                    self._base = BaseVersion(system, self.question)
            if isinstance(host, Dispatcher):
                host.loop = self
            system.adapter_loop = self                      # System.teach / outcome read it
            self.host = host
            if self.part is not None:
                from ...core.deciders.combine import serve_by_loop
                serve_by_loop(self.part, self)              # a combination built over the part from now on: refused
            return self

    def _refuse_no_label_source(self, disp):
        """Stream mode with the out-of-the-box policy: the loop's only labels are the slow path's (tickets are not
        built with a loop, `teach` is only stored). When the policy refuses them the loop learns nothing and every
        decision escalates for ever (nothing can certify a version) — refused. The slow path's labels have its
        `label_source` attribute ("human" for a trusted oracle in the process; "s2" when it has none)."""
        if not isinstance(self.policy, DefaultPolicy):
            return
        src = str(getattr(disp.slow, "label_source", None) or "s2")
        if self.policy.accept(self._state(), None, None, src):
            return
        raise ValueError(
            f"the policy refuses the slow path's labels (source {src!r}) and a Dispatcher's loop has no other label "
            "source (tickets are not built with a loop; teach is only stored): the loop would learn nothing and every "
            "decision would escalate to the slow path for ever. Give one of: DefaultPolicy(s2_labels=True) (the slow "
            "path's accepted answers count); a trusted oracle as the slow path (a SlowPath whose Thoughts carry "
            "{'oracle': {'label_source': 'human'}} and that says label_source = 'human'); or label mode "
            "(AdapterLoop.attach(system, ...) on the System alone, fed by teach / outcome)")

    def detach(self, *, by=None, why=None, timeout=30.0) -> dict | None:
        """Take the loop off its host: a training in flight is cancelled, version 0 (the part as attached) is put back
        on the System when another version serves, the host's hook is removed and the loop is closed (its records,
        `replay()`, `versions()`, `status()` and `save` stay). → the "detach" event (None when not attached). Waits up
        to timeout seconds for the training's thread after releasing every lock; with wait=True a training in progress
        holds the locks (a Dispatcher's `_learn_lock` included) until it ends, so the detach — and the Dispatcher's
        decisions — wait for it first."""
        with self._locked():
            host = self.host
            if host is None:
                return None
            e = self._detach(by=by, why=why)
            run, self._detach_run = self._detach_run, None
            if getattr(host, "loop", None) is self:
                host.loop = None
            if self.part is not None:
                from ...core.deciders.combine import serve_by_loop
                serve_by_loop(self.part, None)
            sysm = _host_system(host)
            if getattr(sysm, "adapter_loop", None) is self:
                sysm.adapter_loop = None
            self.host = None
        th = None if run is None else run.get("thread")
        if th is not None:
            th.join(timeout)
        return e

    def _detach(self, *, by=None, why=None, record=None):
        """The detach's state change and event (replay: `record` is the recorded event; nothing is installed)."""
        run = self._cancel()
        self._detach_run = run
        uninstall = None
        if record is not None:
            uninstall = record.get("uninstall")
        elif self.served != 0 and self.system is not None:
            uninstall = self._base.install(self.system)        # version 0 back: the part as attached
        e = self._control("detach", **{"from": self.served, "cancelled": None if run is None else run["job"].id,
                                       "uninstall": uninstall, "by": None if by is None else str(by), "why": why})
        self.paused, self.closed, self.detached = True, True, True
        return e

    def _served_obj(self, v):
        """Version v's Served: version 0's, the one the loop holds, or (evicted) loaded from versions_dir."""
        if v == 0:
            return self._base, False
        ver = self._vs.get(v)
        if ver is None:
            raise ValueError(f"no version {v}")
        if ver.get("served_as") is not None:
            return ver["served_as"], False
        cls = ver.get("served_cls")
        if cls is None or self.versions_dir is None:
            raise ValueError(f"version {v} is not held and has no file to load (versions_dir=)")
        kw = ({"part": self.part} if self.part is not None
              and "part" in inspect.signature(cls.load).parameters else {})
        return cls.load(self.versions_dir, v, expect_fp=ver["fingerprint"], **kw), True

    @contextlib.contextmanager
    def serving(self, v):
        """Version v installed on the System for the duration (the Dispatcher's replay of a decision made under it),
        then the served version again. Holds the part's lock and the loop's for the whole block: the part's decisions
        and the loop's controls wait."""
        with self._locked():
            if v == self.served or self.system is None:
                yield
                return
            srv, loaded = self._served_obj(v)
            cur, _ = self._served_obj(self.served)
            saved = [(o, o.__dict__.get("_replaced")) for o in (srv, cur) if hasattr(o, "__dict__")]
            srv.install(self.system)
            try:
                yield
            finally:
                cur.install(self.system)
                for o, r in saved:
                    o._replaced = r
                if loaded:
                    self._release(srv)

    def _draw(self, n, score_s, thr_s):
        """Decision n's (probation, rate, sampled, alone) from the state before it (no state change). Label mode: no
        coin and nothing alone — every call is a label."""
        if self.mode == "label":
            return False, 0.0, False, False
        prob = self.left > 0
        rho = self.rate.rate(full=math.isinf(thr_s) or prob)
        sampled = self._coin(n) < rho
        alone = not prob and score_s is not None and math.isfinite(thr_s) and score_s >= thr_s
        return prob, rho, sampled, alone

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

    def _advance(self, x, key, would_s, score_s, would_k, score_k, oracle, src, lenient=False, watch=None,
                 trial=None):
        from ...core.sources import UntrustedLabel
        n = self.n + 1
        sv, c = self.served, self.candidate
        thr_s = self._threshold(sv)
        prob, rho, sampled, alone = self._draw(n, score_s, thr_s)
        label, lsrc, lof, split = None, None, None, None
        if sampled or not alone:
            got = oracle()
            if isinstance(got, Verdict):
                label, lof = got.label, got.label_of
                src = src if got.source is None else got.source
            else:
                label = got
            if label is not None:
                try:
                    lsrc = self._check(src, x, label)      # a refused source: nothing has changed yet
                except UntrustedLabel:
                    if not lenient:
                        raise
                    label, lof = None, None                # attached: a label the policy refuses is a missing one
                    self._label_refused(src)
                else:
                    self._refused["run"] = 0
            if label is not None:
                split = self._split(key)
            else:
                lof = None
        t = self._now()
        self._step_t = t
        try:
            return self._advance_at(n, x, key, sv, c, thr_s, prob, rho, sampled, alone, label, lsrc, lof, split,
                                    would_s, score_s, would_k, score_k, t, watch, trial)
        finally:
            self._step_t = None

    def _label_refused(self, src):
        """Attached to a Dispatcher: one more label the policy refused (counted; status()["labels"]["refused"]). After
        `warn_refused` of them in a row a UserWarning, once: a loop whose every label is refused learns nothing and
        escalates every decision for ever (S2's run-time check of a slow path's declared `label_source`)."""
        r = self._refused
        r["n"] += 1
        r["run"] += 1
        if self.warn_refused is not None and r["run"] >= self.warn_refused and not r["warned"]:
            r["warned"] = True
            warnings.warn(
                f"AdapterLoop {self.question!r}: the policy refused the last {r['run']} labels (source {src!r}) and "
                "accepted none in between: the loop learns nothing and every decision escalates. The slow path's "
                "labels carry the source of its Thoughts' {'oracle': {'label_source': ...}} record ('s2' without one), "
                "not its label_source attribute. Give one of: DefaultPolicy(s2_labels=True); a trusted oracle whose "
                "Thoughts carry {'oracle': {'label_source': 'human'}}; or label mode (AdapterLoop.attach(system, ...))",
                UserWarning, stacklevel=2)

    def _advance_at(self, n, x, key, sv, c, thr_s, prob, rho, sampled, alone, label, lsrc, lof, split, would_s,
                    score_s, would_k, score_k, t, watch=None, trial=None):
        # --- from here the state changes: a hook that raises from here on stops the loop (`_stop`), the decision is
        # recorded with its "error" event and replay raises the same error at the same place
        self.n = n
        self._hk, self._where = 0, "loop"
        det = self.detector
        events, answer, by = [], None, None
        wrec = None                                        # label mode: the watch's view of this label
        try:
            if self._inject is not None and self._inject["call"] == 0:
                raise _Replayed(self._inject["why"])
            for v, score in ((sv, score_s), (c, score_k)):
                if v is not None and self._vs[v]["promise"] is not None:
                    self._vs[v]["promise"].observe(score)
            cmp = None if c is None else self._vs[c].get("cmp")
            if cmp is not None:                            # the served version, read as the candidate's promise reads
                cmp.observe(score_s)
            if sampled:
                for v, score, would in ((sv, score_s, would_s), (c, score_k, would_k)):
                    if v is None or self._vs[v]["promise"] is None:
                        continue
                    ver = self._vs[v]
                    if not n > ver["frozen_at"]:           # cannot happen: a version is frozen at the end of a step
                        raise RuntimeError(f"decision {n} would calibrate version {v} frozen at {ver['frozen_at']}")
                    ver["promise"].learn(score, would, label)
                    ver["fed"].append(n)
                if cmp is not None:
                    cmp.learn(score_s, would_s, label)
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
                if key not in self._labelled:
                    self._labelled.add(key)
                    self._split_keys[split] += 1
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
            if self.mode == "label" and self._watch is not None:
                wrec, ev_w = self._watched(n, sv, thr_s, would_s, score_s, label, split, watch)
                events += ev_w
            events += self._protocol(n)
            events += self._promise_events()
        except Exception as e:  # noqa: BLE001 — recorded, then raised again
            err = self._stop(n, e)
            events += self._promise_events() + [err]
            self._write(self._record(n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s,
                                     score_k, det, key, split, t, error={k: err[k] for k in ("where", "call", "why")},
                                     watch=wrec, trial=trial),
                        self._payload(x, would_s, would_k, label, answer, watch, c, trial), events)
            raise
        rec = self._record(n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s, score_k, det,
                           key, split, t, watch=wrec, trial=trial)
        self._write(rec, self._payload(x, would_s, would_k, label, answer, watch, c, trial), events)
        return {"n": n, "answer": answer, "by": by, "served": sv, "candidate": c, "sampled": sampled, "label": label,
                "label_source": lsrc, "events": [e["event"] for e in events]}

    def _payload(self, x, would_s, would_k, label, answer, watch, c=None, trial=None):
        out = {"input": x, "would": would_s, "shadow_would": would_k, "label": label, "answer": answer}
        if self.mode == "label":
            out["watch_would"] = None if watch is None else watch[0]
        if self.trace_candidate and c is not None:
            out["candidate_would"] = None if trial is None else trial[0]
        return out

    def _watched(self, n, sv, thr_s, would_s, score_s, label, split, watch):
        """Label mode, a watch running: a calib or held-out label scores the served version and the rollback target on
        the same input (+1 alone and right, −1 alone and wrong, 0 not alone, each at its own threshold); the counts
        grow and the policy's `watch` may roll the served version back to the target → (the record's "watch", events).
        A train label (a key a version may have trained on) and a missing label are not counted."""
        w = self._watch
        tgt = w["target"]
        if watch is None:
            raise RuntimeError(f"decision {n}: a watch runs and the target's score was not read")
        would_t, score_t = watch
        rec = {"target": tgt, "score": _num(score_t), "counted": False}
        if label is None or split not in ("calib", "holdout"):
            return rec, []

        def val(would, score, thr):
            return (1 if would == label else -1) if _alone(score, thr) else 0
        cs, ct = val(would_s, score_s, thr_s), val(would_t, score_t, self._threshold(tgt))
        w["n"] += 1
        w["better"] += ct > cs
        w["worse"] += ct < cs
        w["total"] += ct - cs
        if ct != cs:                                       # a discordant label: the e-process bets on it
            mix = watch_eprocess().restore({"bet": "mixture", "lw": w["lw"], "n": w["better"] + w["worse"] - 1})
            mix.update(1.0 if cs > ct else 0.0)
            w["lw"] = mix.vector()
        cur = _watch_of(w)
        rec.update(counted=True, served_value=cs, target_value=ct, log_e=cur.log_e)
        self._at("policy.watch")
        if not self.policy.watch(self._state(), cur):
            return rec, []
        why = (f"watch: version {sv} worse than version {tgt} on {cur.n} calib / held-out label(s) since its "
               f"promotion ({cur.better} better on {tgt}, {cur.worse} better on {sv}, margin {cur.margin:+.3f})")
        _, ev = self._rollback(tgt, by="policy", why=why, control=False)
        ev[[e["event"] for e in ev].index("rollback")]["watch"] = _watch_copy(w)
        return rec, ev

    def _record(self, n, sv, c, rho, sampled, alone, by, lsrc, lof, label, prob, score_s, thr_s, score_k, det, key,
                split, t, error=None, watch=None, trial=None):
        st = lambda v: None if v is None or self._vs[v]["promise"] is None else self._vs[v]["promise"].state()  # noqa: E731
        rec = {"n": n, "served": sv, "candidate": c, "rho": rho, "sampled": sampled, "alone": alone, "by": by,
               "label_source": lsrc, "label_of": lof, "missing": (sampled or not alone) and label is None,
               "probation": prob, "left": self.left, "score": _num(score_s),
               "threshold": None if math.isinf(thr_s) else thr_s, "shadow": _num(score_k), "served_state": st(sv),
               "candidate_state": st(c), "detector": None if det is None else det.state(), "seen": self.live.seen(),
               "key": key, "split": split, "t": t}
        if self.mode == "label":
            rec["watch"] = watch
        if self.trace_candidate and c is not None:
            rec["candidate_score"] = None if trial is None else _num(trial[1])
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
        run = self._cancel()
        inflight, self._inflight, self._job = self._inflight, None, None
        return self._event("error", self.served, n, where=self._where, call=self._hk, why=why,
                           candidate=self.candidate, train=inflight,
                           cancelled=None if run is None else run["job"].id, **extra)

    def _cancel(self):
        """The training in flight is dropped: its job's cancel event is set (the runner stops it) → its run."""
        run, self._run, self._job = self._run, None, None
        if run is not None and run["job"].cancel is not None:
            run["job"].cancel.set()
        return run

    def _refuse_if_stopped(self):
        if self.detached:
            raise RuntimeError("the loop is detached; version 0 serves again")
        if self.closed:
            raise RuntimeError("the loop is closed; the served version keeps serving")
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
            paired = None if self.mode == "label" else self._paired(c)    # label mode: the pre-freeze gates decided
            st = self._state()
            certified = math.isfinite(self._threshold(c))     # as the promote() control: never an uncertified one
            if not self.paused:
                self._at("policy.promote")
            if not self.paused and certified and self.policy.promote(st, paired):
                ev.append(self._event("promote", c, n, **{"from": self.served, "to": c, **self._paired_fields(c, paired),
                                                         "why": None}))
                ev[-1]["install"] = self._serve(c, n, promoted=True)
                ev += self._evict(n)
            else:
                self._at("policy.discard")
                why = self.policy.discard(st, paired)
                if why:
                    ev.append(self._discard(c, n, str(why)))
        if self.candidate is None and not self.paused:
            if self._run is None and self._job is None:
                self._at("policy.want_train")
                if self.policy.want_train(self._state()):
                    ev += self._start("schedule", n, inline=self.wait)[1]
            if self._run is not None and self._ready(n, "decision"):
                ev += self._apply(n, "decision")[1]
        return ev

    def _paired_fields(self, c, paired):
        """A promote event's comparison: the paired samples' numbers (stream mode); None in label mode, which has none
        (its measurement is the freeze's report)."""
        if paired is None:
            return {"l_k": None, "l_s": None, "samples": len(self._vs[c]["samples"]), "n_novel": None, "n_seen": None}
        cmp = self._vs[c].get("cmp")
        return {"l_k": paired.share_k, "l_s": paired.share_s, "l_s_own": paired.share_s_own,
                "thr_s_cmp": None if cmp is None or not math.isfinite(cmp.threshold) else cmp.threshold,
                "thr_s_own": (None if not math.isfinite(self._threshold(self.served))
                              else self._threshold(self.served)), "thr_s": paired.thr_s,
                "samples": len(self._vs[c]["samples"]),
                "n_novel": paired.n_novel, "n_seen": paired.n_seen}

    def _restart(self, v, n, why="alarm") -> list:
        """Version v's promise from scratch (the next (v, δ_v), threshold +inf, wealth 1) and, for a candidate, its
        paired samples dropped. → its "recertify" and a "restart" event."""
        ver = self._vs[v]
        ver["promise"].restart(why)
        if ver.get("cmp") is not None:                     # the comparison restarts with the candidate, at its new δ_v
            ver["cmp"].budget.v = ver["promise"].v - 1
            ver["cmp"].restart(why)
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
        ver["cmp"] = None                                  # out of trial: its comparison certificate is done
        ver["discarded_at"] = self._split_n["train"]
        self.candidate = None
        return self._event("discard", c, n, samples=len(ver["samples"]), why=why)

    def _serve(self, v, n, promoted=False):
        """Version v served from decision n (its install on the System → the install record). Label mode: a promotion
        starts the watch against the version served until now; any other change of the served version ends it."""
        old = self.served
        self._watch = ({"target": old, "n": 0, "better": 0, "worse": 0, "total": 0, "lw": watch_eprocess().vector()}
                       if promoted and self.mode == "label" and old != v else None)
        if old != v:
            self._vs[old]["state"] = "retired"
        self._vs[v]["state"] = "served"
        self._vs[v]["cmp"] = None                          # served: out of trial, its comparison certificate is done
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

    def _report_inputs(self):
        """What the pre-freeze report reads, taken when a training starts: the calib and held-out labels (latest label
        per key), the served version's model and threshold."""
        latest = {}
        for r in self._rows:
            latest[r.key] = r
        calib = [r for k, r in latest.items() if self._split_of[k] == "calib"]
        hold = [r for k, r in latest.items() if self._split_of[k] == "holdout"]
        return {"calib": calib, "hold": hold, "served": self._vs[self.served]["model"],
                "thr_s": self._threshold(self.served), "sv": self.served, "hist": len(self._history)}

    def _report(self, frozen, inp, cancel=None) -> TrainReport:
        """The trained model and the version served when the training started on the held-out labels of that moment
        (latest label per key); its threshold on the calib labels. Stops between two labels when the training is
        cancelled (TrainingCancelled) or the served version changed since the start (_Stale: the comparison would be
        stale; the apply discards the candidate). The served side is scored under the part's version lock with that
        check, so a version retired and evicted meanwhile is never scored (and so never staged) again."""
        calib, hold, served, thr_s = inp["calib"], inp["hold"], inp["served"], inp["thr_s"]

        def go_on():
            if cancel is not None and cancel.is_set():
                raise TrainingCancelled("the training was cancelled during its report")
            if len(self._history) != inp["hist"]:
                raise _Stale(inp["sv"], self.served)
        thr = getattr(frozen, "act_threshold", None)
        if thr is None and calib:
            scored = []
            for r in calib:
                go_on()
                a, s = frozen.score(r.x)
                scored.append((s, a != r.y))
            thr = calib_threshold(scored, self.alpha)
        right = wrong = s_right = s_wrong = better = worse = 0
        items = []
        pl = self._part_lock
        for r in hold:
            go_on()
            a, s = frozen.score(r.x)
            ck = (1 if a == r.y else -1) if thr is not None and _alone(s, thr) else 0
            if served is not None:
                with pl if pl is not None else contextlib.nullcontext():   # the served version cannot change here
                    go_on()
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

    # --- a training: started at one place, applied at a later one (deferred apply, design §3.2)
    def _start(self, trigger, n, *, inline):
        """Start one training (the budget checked) → (TrainJob, events). inline: it runs here, to the end (the loop's
        wait=True, train_now(wait=True), an online learner's snapshot); else in a thread of its own, and the loop goes
        on — its result is applied by `_apply` on a later decision. Replay starts nothing: the result is the recorded
        one, applied where the records say."""
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
        wait = bool(inline)                                # what was asked (recorded); an online learner is inline
        inline = wait or not self._batch                   # a snapshot is a copy: nothing to wait for
        self._job = {"id": job.id, "version": k, "trigger": trigger, "started": job.started, "eta": None,
                     "runner": None, "pid": None, "log": None, "device": None, "last": None, "last_t": None,
                     "state": "running", "started_at": n, "inline": inline}
        if self._batch:                                    # what a batch learner's runner reads (stage 3)
            latest = {}
            for r in self._rows:
                if self._split_of[r.key] == "calib":
                    latest[r.key] = r
            job.calib, job.budget, job.alpha, job.status = (tuple(latest.values()), self.policy.budget, self.alpha,
                                                            self._job)
        job.cancel = threading.Event()
        run = {"job": job, "trigger": trigger, "n": n, "inline": inline, "done": threading.Event(), "out": {},
               "thread": None, "k": k, "n_train": len(job.rows) if self._batch else None, "sv": self.served,
               "hist": len(self._history)}
        self._run = run
        extra = {"control": True, "wait": wait} if trigger == "train_now" else {}
        ev = [self._event("train_start", self.served, n, job=job.id, trigger=trigger, rows=len(job.rows),
                          hyper=job.hyper, t=job.started, inline=inline, served_at_start=self.served, **extra)]
        if self._fps is not None:                          # replay: nothing runs
            run["done"].set()
            return job, ev
        inp = self._report_inputs()
        if inline:
            self._body(run, inp)
        else:
            th = threading.Thread(target=self._body, args=(run, inp), daemon=True, name=f"solvi-loop-train-{job.id}")
            run["thread"] = th
            th.start()
        return job, ev

    def _body(self, run, inp):
        """The training itself and its pre-freeze report — in the training's thread (or inline); it reads only what
        `_start` handed it and writes only `run["out"]`."""
        out, job = run["out"], run["job"]
        t0 = time.perf_counter()
        try:
            frozen = self.live.train(job) if self._batch else self.live.snapshot()
        except BaseException as e:  # noqa: BLE001 — applied (recorded, or the loop stopped) on the decision thread;
            out.update(exc=e, where="learner.train")   # a KeyboardInterrupt in the thread is a failed training
        else:
            out.update(frozen=frozen, seconds=time.perf_counter() - t0)
            try:
                out["report"] = self._report(frozen, inp, job.cancel)
            except BaseException as e:  # noqa: BLE001
                out.update(exc=e, where="report")
        finally:
            if job.status is not None:
                job.status["state"] = "ready"
            run["done"].set()

    def _ready(self, n, where) -> bool:
        """Whether the training in flight is applied here. Live: its result is in (a training that runs in a thread is
        applied at the earliest on the decision after the one it started at). Replay: the records say so — the next
        recorded result was applied at this decision (where "decision") or in this control; no clock is read."""
        run = self._run
        if self._fps is not None:
            rec = self._replay_trains[0] if self._replay_trains else None
            return (rec is not None and rec.get("job") == run["job"].id and rec.get("decision") == n
                    and rec.get("applied") == where)
        if not run["done"].is_set():
            return False
        return run["inline"] or n > run["n"]

    def wait_training(self, timeout=None) -> bool:
        """Wait (outside every lock) until the training in flight has a result, or timeout seconds → whether it has
        one (True when none is in flight). It is applied on the next decision."""
        run = self._run
        return True if run is None else run["done"].wait(timeout)

    def _apply(self, n, where):
        """The result of the training in flight, applied here (where: "decision" or "train_now") → (TrainJob,
        events): a "train" event at this decision — state "done" (then a freeze or a discard), "failed" or "refused"
        — or the loop stops (a RunnerError: the runner failed; an exception of the report). The paths are stage 3's."""
        run = self._run
        job, trigger, k, out = run["job"], run["trigger"], run["k"], run["out"]
        self._job, self._run = None, None                  # applied here: no longer in flight (nothing to cancel)
        self._inflight = {"job": job.id, "decision": n, "applied": where, "state": None, "fingerprint": None,
                          "report": None, "extra_keys": []}   # what an error record keeps of this training
        self._at("learner.train")                          # a runner that died stops the loop here (RunnerError)
        rec = None
        if self._fps is not None:                          # replay: the recorded result
            rec = self._replay_trains.pop(0) if self._replay_trains else {}
            if rec.get("job") not in (None, job.id):
                raise RuntimeError(f"replay: training {job.id} applied here, the records hold {rec.get('job')}")
        frozen, error, refused = None, None, None
        if rec is not None:
            if rec.get("state") == "failed":
                error = rec.get("why")
            elif rec.get("state") == "refused":
                refused = rec.get("why")
            else:
                frozen = _Recorded(rec.get("fingerprint"))
        elif "frozen" not in out and "exc" not in out:  # cannot happen: _body always leaves one of them
            error = "the training's thread ended without a result"
        elif out.get("where") == "learner.train":
            e = out["exc"]
            if isinstance(e, RunnerError):
                raise e                                # the runner itself failed: the loop stops (status "error")
            if isinstance(e, TrainingRefused):         # the time estimate is above max_train_s: not started
                refused = str(e)
            else:                                      # a failed training is recorded; the served one serves on
                error = f"{type(e).__name__}: {e}"
        else:
            frozen = out["frozen"]
        n_train = run["n_train"] if self._batch else self._split_n["train"]
        base = {"job": job.id, "trigger": trigger, "rows": len(job.rows), "hyper": job.hyper, "t": job.started,
                "started_at": run["n"], "applied": where, "served_at_start": run["sv"]}
        if refused is not None:
            job.state, job.why = "refused", refused
            return job, [self._event("train", self.served, n, state="refused", why=refused, fingerprint=None,
                                     report=None, frozen=False, **base)]
        if error is not None:
            job.state, job.why = "failed", error
            return job, [self._event("train", self.served, n, state="failed", why=error, fingerprint=None,
                                     report=None, frozen=False, **base)]
        if len(self._history) != run["hist"]:              # S1: the served version changed during the training
            return job, self._stale(job, run, n, frozen, k, n_train, base)
        self._at("report")
        if rec is not None:
            r = dict(rec["report"])
            r["items"] = tuple(tuple(i) for i in r.get("items") or ())
            report = TrainReport(**r)
        else:
            if out.get("where") == "report":
                raise out["exc"]
            report = dataclasses.replace(out["report"], seconds=out["seconds"], n_train=n_train)
        self._inflight.update(state="done", report=dataclasses.asdict(report))
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
            self._vs[k]["fixed"] = report.threshold        # label mode serves at it (act_guard's on the calib labels)
        else:
            self._release(frozen)                          # never served: unstaged at once
            row = self._row(k, "discarded", None, None, n, self.live.seen(), n_train, keys, fp)
            row["discarded_at"] = self._split_n["train"]
            self._vs[k] = row
            self._ev_seen[k] = 0
            ev.append(self._event("discard", k, n, samples=0, why=f"freeze refused: {why or 'the policy said no'}"))
        self._inflight = None
        return job, ev

    def _stale(self, job, run, n, frozen, k, n_train, base) -> list:
        """The served version changed while the training ran (a rollback): its pre-freeze report compares the candidate
        with a version no longer served, so it is not read — the candidate is discarded at birth, released at once, and
        the "train" event (state "stale") and the "discard" event say which version it was compared with. Chosen over
        recomputing the report on the decision thread, which would hold the part's lock for a whole held-out pass."""
        why = str(_Stale(run["sv"], self.served))
        self._at("fingerprint")
        fp = frozen.fingerprint()
        self._inflight.update(state="stale", fingerprint=fp)
        job.state, job.why, job.fingerprint, job.frozen = "stale", why, fp, False
        self._release(frozen)                              # never served: unstaged at once
        keys = frozenset(r.key for r in job.rows) if self._batch else frozenset(self._fed_keys)
        row = self._row(k, "discarded", None, None, n, self.live.seen(), n_train, keys, fp)
        row["discarded_at"] = self._split_n["train"]
        self._vs[k] = row
        self._ev_seen[k] = 0
        self._inflight = None
        return [self._event("train", self.served, n, state="stale", why=why, fingerprint=fp, report=None,
                            frozen=False, **base),
                self._event("discard", k, n, samples=0, why=f"stale: {why}")]

    # --- eviction (Budget.max_versions_kept)
    def _rollback_target(self):
        """The version a rollback would most likely return to: the last one served before the served one."""
        for v in reversed(self._history):
            if v != self.served:
                return v
        return None

    def _evict(self, n) -> list:
        """Past policy.budget.max_versions_kept served and retired versions (version 0 counted), the oldest retired
        ones are evicted: released (`Served.release`), no longer a rollback target; their files stay (replay reads
        them). Never the served version, the rollback target or version 0. → the "evict" events."""
        keep = self.policy.budget.max_versions_kept
        held = [v for v, ver in sorted(self._vs.items()) if ver["state"] in KEPT]
        over = len(held) - int(keep)
        ev = []
        if over <= 0:
            return ev
        protect = {self.served, self._rollback_target(), 0}
        for v in held:
            if over <= 0:
                break
            if v in protect:
                continue
            ver = self._vs[v]
            self._at("release")
            self._release(ver.get("served_as"), ver["model"])
            ver["state"], ver["model"], ver["served_as"] = "evicted", None, None
            over -= 1
            ev.append(self._event("evict", v, n, kept=int(keep), why="max_versions_kept"))
        return ev

    def _freeze(self, k, n, model, keys, n_train):
        if self._fps is not None:
            model = _Recorded(self._fps.get(k))
        fp = model.fingerprint()
        self._at("served")
        srv = self._served_of(model) if self._fps is None else None
        self._at("save")
        if self.versions_dir is not None and srv is not None:
            srv.save(self.versions_dir, k)
        p = None if self.mode == "label" else StreamPromise(**self._pkw, budget=self.budget)
        self._ev_seen[k] = 0
        row = self._row(k, "candidate", model, p, n, self.live.seen(), n_train, keys, fp)
        row["served_as"], row["served_cls"] = srv, None if srv is None else type(srv)
        row["cmp"] = (None if p is None or self._vs[self.served]["model"] is None   # no model served: nothing to rank
                      else self._comparison(p))
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
        happens (its δ_v stays spent), as do alarms and rollbacks. A training in flight goes on; its result waits for
        resume() (applied on the first decision after it)."""
        with self._locked():
            self._refuse_if_stopped()
            if self.paused:
                return None
            self.paused = True
            return self._control("pause", by=None if by is None else str(by), why=why)

    def resume(self, *, by=None, why=None) -> dict | None:
        with self._locked():
            self._refuse_if_stopped()
            if not self.paused:
                return None
            self.paused = False
            return self._control("resume", by=None if by is None else str(by), why=why)

    def train_now(self, *, wait=None) -> TrainJob:
        """One training now, whatever want_train says — within the budget, never while paused, with a candidate in
        trial or with a training in flight. → the TrainJob. wait (None: the loop's `wait`): False — a batch learner's
        training starts and the call returns at once (state "running"; its result is applied on a later decision, see
        `step`); an online learner's snapshot is a copy, always taken here (state "done"; the "train_start" event
        records the wait asked and inline=True); True — the call returns when the training is over and its result
        applied (state "done", "failed", "refused" or "stale" with why). A RunnerError stops the loop where the result
        is applied. A train_now refused (paused, a candidate in trial, a training in flight, the budget) writes its
        control event, a "train" with state "refused"."""
        wait = self.wait if wait is None else bool(wait)
        with self._locked():
            self._refuse_if_stopped()
            k = max(self._vs) + 1
            why = ("paused" if self.paused else f"version {self.candidate} is in trial" if self.candidate is not None
                   else "a training is in flight" if self._run is not None else None)
            ev = []
            if why:
                job = TrainJob(id=f"{k}.-", version=k, trigger="train_now", started=self._now(), state="refused",
                               why=why)
            else:
                job, ev = self._guarded("train_now", {"wait": wait}, lambda: self._train_now(wait))
                ev += self._promise_events()
                if ev:
                    self._write(None, None, ev)
            if job.state == "refused" and not any(e.get("event") == "train" for e in ev):
                self._control("train", state="refused", why=job.why)   # every control writes an event, a refused one
            return job                                                 # too (refused by the estimate: its own "train")

    def _train_now(self, wait):
        job, ev = self._start("train_now", self.n, inline=wait)
        if job.state == "refused" or self._run is None:
            return job, ev
        if self._run["inline"]:
            job, more = self._apply(self.n, "train_now")
            ev += more
        return job, ev

    def close(self, *, timeout=30.0) -> dict | None:
        """Stop the loop: a training in flight is cancelled (its runner stops it: a subprocess is killed, a thread
        runner stops waiting for its thread), `step` and every control refuse from here on; the served version stays
        installed. With `wait=True` a training runs inside a step or a control under the loop's locks, so a training in
        progress there ends first (a Dispatcher's decisions wait for it too: it holds `_learn_lock`). Waits up to
        timeout seconds for the training's thread, outside every lock. → the "close" event
        (None when already closed). Also `with AdapterLoop(...) as loop:`."""
        with self._locked():
            if self.closed:
                return None
            run = self._cancel()
            self.paused = True
            e = self._control("close", cancelled=None if run is None else run["job"].id)
            self.closed = True
        th = None if run is None else run.get("thread")
        if th is not None:
            th.join(timeout)
        return e

    def cancel_training(self, *, by=None, why=None) -> dict | None:
        """Drop the training in flight (running, or its result waiting to be applied): its runner stops it, nothing is
        applied, nothing restarts it here; it stays counted against the budget. → the "cancel" event (None when no
        training is in flight). The policy may start another one on a later decision."""
        with self._locked():
            self._refuse_if_stopped()
            if self._run is None:
                return None
            run = self._cancel()
            return self._control("cancel", cancelled=run["job"].id, by=None if by is None else str(by), why=why)

    # --- the checkpoint (stage 4; stream.checkpoint)
    def save(self, dir, *, cancel=False) -> dict:
        """Save the loop's whole state to `dir` (see `stream.checkpoint`): written to a temporary folder, then switched
        in atomically, so a crash in the middle leaves the previous checkpoint usable. A training in flight refuses the
        save (RuntimeError) unless cancel=True, which cancels it first (`cancel_training`, a "cancel" event). → {"path",
        "manifest", "n", "files"}. Writes no record otherwise. The save holds the part's version lock and the loop's lock
        for all its writes (every version's files, each fsynced, included): the part's decisions and the loop's controls
        wait for it; unmeasured."""
        from .checkpoint import save_loop
        return save_loop(self, dir, cancel=cancel)

    @classmethod
    def load(cls, dir, learner, **kw) -> "AdapterLoop":
        """The loop saved in `dir`, continuing where it was saved (see `stream.checkpoint.load_loop` for the keywords:
        system, storage, versions_dir, part, initial, key, served, clock, secret, groups)."""
        from .checkpoint import load_loop
        return load_loop(cls, dir, learner, **kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def promote(self, v=None, *, force=False, by=None, why=None) -> dict:
        """Serve the candidate now: it must be certified; without force the policy's promote must agree (force skips
        the promotion rule, never the certificate); refused while paused. Attached to a Dispatcher, every control waits
        for the decision in flight (its slow path's call included). → the "promote" event."""
        with self._locked():
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
        paired = None if self.mode == "label" else self._paired(c)
        if not force:
            self._at("policy.promote")
        if not force and not self.policy.promote(self._state(), paired):
            raise ValueError(f"the policy does not promote version {c} now (force=True skips its rule)")
        self._changing = True
        e = self._event("promote", c, self.n, **{"from": self.served, "to": c, **self._paired_fields(c, paired),
                                                "why": why, "by": None if by is None else str(by),
                                                "forced": bool(force), "control": True, "t": self._now()})
        e["install"] = self._serve(c, self.n, promoted=True)
        ev = self._evict(self.n)
        self._write(None, None, [e] + ev + self._promise_events())
        return e

    def rollback(self, v, *, by=None, why=None) -> dict:
        """Serve an earlier version again (its certificate and threshold as it left them, unless an alarm came after it
        retired: then its promise restarts → "restart"; version 0 without a model: nothing alone). A candidate in trial
        is discarded (its δ_v stays spent); waits for a Dispatcher's decision in flight → the "rollback" event."""
        with self._locked():
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
        ev += self._evict(self.n)
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
        fresh._part_lock = None                            # the replay scores nothing on the part
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
                    fresh.train_now(wait=a.get("wait"))
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
                    "train": lambda d: fresh.train_now(), "error": again,
                    "train_start": lambda d: fresh.train_now(wait=d.get("wait")), "close": lambda d: fresh.close(),
                    "cancel": lambda d: fresh.cancel_training(by=d.get("by"), why=d.get("why")),
                    "detach": lambda d: fresh._detach(by=d.get("by"), why=d.get("why"), record=d)}
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
                wr = st.get("watch")
                fresh._advance(lp.get("input"), st.get("key"), lp.get("would"), st.get("score"),
                               lp.get("shadow_would"), st.get("shadow"),
                               lambda: Verdict(label, st.get("label_source") or fresh.source, st.get("label_of")),
                               fresh.source, watch=None if wr is None else (lp.get("watch_would"), wr.get("score")),
                               trial=(lp.get("candidate_would"), st["candidate_score"]) if "candidate_score" in st
                               else None)
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
            if ("candidate_score" in st) != ("candidate_score" in got):    # trace_candidate: on every trial decision
                bad.append((st["n"], "candidate_score", "recorded " + ("with" if "candidate_score" in st else
                                                                       "without") + " the candidate's trace"))
        rebuilt_ev = [e for kk, e in fresh.log if kk == "stream"]
        if rebuilt_ev != stored_ev:
            bad.append((None, "stream records", f"{len(stored_ev)} stored, {len(rebuilt_ev)} rebuilt, or they differ"))
        if self.versions_dir is not None:
            bad += self._check_files(fps, items, rescore)
        return {"ok": not bad, "mismatches": bad, "decisions": k, "versions": len(fps)}

    def _check_files(self, fps, items, rescore):
        bad, models = [], {}
        with self._lock:                                   # a background apply may add a version meanwhile
            vs = {v: dict(ver) for v, ver in self._vs.items()}
        for v, fp in fps.items():
            cls = (vs.get(v) or {}).get("served_cls")
            if cls is None:
                bad.append((None, "version file", f"version {v}: no Served to load it with"))
                continue
            srv = (vs.get(v) or {}).get("served_as")
            part = getattr(srv, "part", None) or self.part      # an evicted version: the loop's part
            kw = ({"part": part} if part is not None
                  and "part" in inspect.signature(cls.load).parameters else {})   # a LoRA version scores on its part
            try:
                models[v] = cls.load(self.versions_dir, v, expect_fp=fp, **kw)
            except FileNotFoundError as e:
                bad.append((None, "version file", f"version {v}: {e.filename} is missing"))
            except Exception as e:  # noqa: BLE001 — a tampered or foreign file
                bad.append((None, "version file", f"version {v}: {e}"))
        if vs[0]["model"] is not None:
            models[0] = vs[0]["model"]                     # version 0: the model given as initial=
        if rescore:
            for kind, d in items:
                if kind != "loop":
                    continue
                st, lp = d["stream"], d.get("loop") or {}
                for v, key in ((st.get("served"), "score"), (st.get("candidate"), "shadow"),
                               (st.get("candidate"), "candidate_score")):
                    if key != "score" and st.get(key) is None:       # not scored in shadow on that decision
                        continue
                    if v in models:
                        got = models[v].score(lp.get("input"))[1]
                        if _num(got) != st.get(key):
                            bad.append((st["n"], "rescore", f"version {v}: recorded {st.get(key)}, scores {got}"))
        live = {id(x) for ver in vs.values() for x in (ver.get("model"), ver.get("served_as")) if x is not None}
        live_fp = {ver["model"].fingerprint() for ver in vs.values() if ver.get("model") is not None}
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
            thr = self._threshold(v)                       # label mode: the version's fixed-set threshold
            out.append({"version": v, "state": ver["state"], "certified": math.isfinite(thr),
                        "threshold": None if math.isinf(thr) else thr,
                        "n": None if p is None else p.n, "v_index": None if p is None else p.v,
                        "delta_v": None if p is None else p.delta_v, "frozen_at": ver["frozen_at"],
                        "seen": ver["seen"], "n_train": ver["n_train"], "train_keys": len(ver["keys"]),
                        "samples": len(ver["samples"]), "discarded_at": ver["discarded_at"],
                        "fingerprint": ver["fingerprint"],
                        "paired": dataclasses.asdict(self._paired(v)) if v == self.candidate else None})
        return out

    def status(self) -> dict:
        """{"state", "error", "served", "candidate", "n", "labels": {"train", "calib", "holdout", "refused"},
        "trainings",
        "budget": {...}, "job", "paused", "alarms", "mode", "attached", "guarantee", "watch", "delta_spent",
        "versions_kept"}. labels: the accepted labels by split and "refused", the labels the policy refused on the
        attached path (each a missing label). watch: label mode after a promotion, the watch's counts {"target", "n",
        "better", "worse", "total", "lw"} ("lw": its e-process's log wealths; None otherwise).
        attached: "dispatcher", "system" or None. state: "detached", "closed", "error" (a hook raised: the loop stopped; error: {"n", "where", "why"}), "training", "paused" or
        "running". job: the training in flight — id, version, trigger, started (wall clock), started_at (its decision),
        inline, state ("running"; "ready": its result waits for the next decision), and what the runner fills in
        (runner, pid, log, device, eta, last, last_t: a heartbeat line's age says whether it is alive).

        Takes no lock: it answers while a step (or an inline training inside it) runs on another thread. Its fields
        are read one by one, so two of them may straddle a decision."""
        b, job, err = self.policy.budget, self._job, self.error
        state = ("detached" if self.detached else "closed" if self.closed else "error" if err is not None
                 else "training" if job is not None else "paused" if self.paused else "running")
        host = self.host
        attached = (None if host is None else "dispatcher" if hasattr(host, "slow") else "system")
        return {"state": state, "error": None if err is None else dict(err), "served": self.served,
                "candidate": self.candidate, "n": self.n, "labels": {**self._split_n, "refused": self._refused["n"]},
                "trainings": len(self._trainings),
                "budget": {"max_trainings": b.max_trainings, "per_day": b.per_day,
                           "refuse": b.refuse(list(self._trainings), self._now())},
                "job": None if job is None else dict(job), "paused": self.paused,
                "alarms": len(self._alarms), "mode": self.mode, "attached": attached,
                "guarantee": ("anytime stream certificate per version (StreamPromise on post-freeze coin samples)"
                              if self.mode == "stream" else
                              "fixed-set (act_guard on calib labels); no anytime certificate"),
                "watch": None if self._watch is None else _watch_copy(self._watch),
                "delta_spent": self.budget.spent,
                "versions_kept": sum(1 for ver in list(self._vs.values()) if ver["state"] in KEPT)}

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
