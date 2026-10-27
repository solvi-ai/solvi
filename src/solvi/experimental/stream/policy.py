"""The policy of the adapter loop: the abstract class a developer controls the loop with, the state its hooks read, the
budget, and the out-of-the-box policy (experimental; the LoRA learning loop, stage 1; `stream.loop.AdapterLoop` runs
it, stage 2).

    from solvi.experimental.stream import Budget, DefaultPolicy, LoopPolicy, LoopState

    policy = DefaultPolicy(budget=Budget(per_day=6, max_train_s=900))
    policy.split(st, key)                 # "train" | "calib" | "holdout": by the content key, fixed forever
    policy.want_train(st)                 # growth, the improvement signal, the budget; why_not_train(st) says why not
    policy.freeze(st, report)             # the pre-freeze gates on held-out labels (freeze_gates(st, report): each one)
    policy.promote(st, paired)            # certified, ≥ 100 novel samples, min_share, margin, the memorisation probe,
                                          # the novel-error bound

`LoopPolicy` is the abstract class: every hook reads a `LoopState` (a frozen snapshot the loop rebuilds per call:
decisions so far, the served version and the candidate, label counts by split, the versions table, the served
version's latest held-out outcomes, alarms, trainings started, the wall clock, paused, the mode) and returns a plain
decision. A hook that raises stops the loop (status "error") and the served version keeps serving. A policy keeps its
own counters in `self` and must pickle (the loop's checkpoint saves it).

`DefaultPolicy` (each number a keyword):

- accept — a label from the trusted sources ("human", "outcome", "rule", "audit"); "s2" / "verified" only with
  s2_labels=True.
- split — `stream.gates.split`: sha256 of the content key → 70 / 15 / 15 train / calib / holdout.
- want_train — no candidate in trial, no training in progress, not paused; train labels ≥ `first` (300) and ≥
  `growth` (1.5) × the largest n_train of any version so far (discarded ones included: a discarded candidate is not
  retrained on nearly the same labels; with `refreeze` given instead, the lane LearningLoop's rule: ≥ growth × the
  served version's n_train and ≥ refreeze × the train labels the loop had at the latest discard); the budget allows
  one now; and the improvement signal
  (`gates.incumbent_short`): on its last `window` (200) held-out labels the served version answers nothing alone, or
  errs alone above the promised α, or answers alone on less than `target_share` (0.5; None: off) of them. Why the
  share: a certified incumbent keeps its error alone ≤ α, so with the error alone as the only signal a weak but
  certified incumbent — short in coverage, not in error — is never retrained (one smoke run, its script not in the
  repository, solvi-base on Banking77: error alone 0.000, share alone 0.090 from about decision 960 to 2 000, no
  training). The share is read on held-out labels,
  a selection-biased sample (labels come mostly from deferred decisions), so it reads low (0.09 there against 0.30 on
  the whole stream in that run), and a strong incumbent will usually be retrained too; the promotion gates (the
  novel-only paired comparison, the memorisation probe, the novel-risk bar) are what keep a worse version from being
  served. Unmeasured: the stand will measure the default. target_share=None restores the error-only signal. The
  signal is read only once the window holds min(min_holdout, window) outcomes of the served version: fewer (the window restarts at a promotion or a rollback) is "no signal yet", not "short", so a new
  served version is not retrained before it was measured (version 0 without a model fills the window with outcomes
  answered by nobody, so it is short as soon as the window is full; with holdout=0 there is no window and the signal
  reads short). With the pre-freeze gates on, a training also waits for the labels those gates need — ≥ min_holdout
  held-out labels and ≥ the calib labels the calib threshold needs (`min_calib`; `gates.calib_min`, 73 at α = 0.05)
  — since a training with fewer is refused at birth (with the default shares the first training comes at about 470
  train labels, not 300).
- select — one row per content key (the latest label wins), then a class-balanced sample of ≤ budget.max_examples
  of them, half from the latest 2 × the served version's n_train rows; seeded by the next version number, so the
  same rows give the same selection.
- hyper — r 8, lr 3e-4, updates = clamp(⌈2 n / 8⌉, 200, 3000) (n: the train labels, ≤ max_examples), seed = the next
  version number, warm_start False (every version is trained from the base checkpoint).
- freeze — on the held-out labels: ≥ min_holdout (100) of them and (alone & right − alone & wrong) / n ≥ the served
  version's + min_gain (0.01); act_guard found a threshold; no honesty number worse by more than tolerance (0.02);
  the training within budget.max_train_s; in label mode also the paired held-out gain's Wilson lower bound > 0.
  pre_freeze=False freezes every training (the lane LearningLoop's protocol: the stream certificate alone decides);
  refused in label mode, where the gates are the only measurement before a version is served.
- promote — the candidate certified, ≥ min_novel (100) post-freeze samples on keys it never trained on, on those
  share_k ≥ min_share (0.05) and share_k ≥ share_s + margin (0.01) (share_s: the served version certified the same
  way on the same samples, `Paired`), and the memorisation probe passes (error alone on
  novel − on seen ≤ gap, 0.10). The candidate must have ≥ probe_min (30) alone answers on novel samples: a thin
  novel side is the memoriser's signature (it answers the items it trained on and abstains on the rest), so an
  unmeasured probe does not promote. The errors are compared once the seen side has ≥ probe_min alone answers too
  (fewer alone answers on seen keys is not memorising). And the candidate's error on novel samples is within the
  promise: the Wilson upper bound (at `wilson`, 95 %; `core.store.sysreport.wilson`) of wrong_novel / n_novel ≤ α
  (st.alpha; the joint risk on keys it never trained on — a memoriser is valid on the pool because its trained items
  dilute the risk; author's decisions 2026-10-23; novel_risk=False drops it). A point estimate looked at on every
  decision passes a candidate above α by chance (at a true novel risk of 6.4 % in more than half of simulated
  1 000-sample trials); the bound holds that near the rate the confidence names (about 5 % of candidates at 5.5 %,
  1 % at 6.4 %) and costs some candidates just under α (at 3 % about 6 % are not promoted within 1 000 novel
  samples). Two limits: the count is at the thresholds in force now, which loosen during a trial, and it is looked at
  on every decision, so the bound is a stand-in for a fixed-sample test, not an anytime guarantee; the rate of false
  promotions it allows is what the shuffled-label control of a stand reads.
  probe=False drops the probe (promotion and discard). Label
  mode (no paired samples): the pre-freeze gates decided, promote. The novel-risk bar is a joint ratio (wrong alone
  over every novel sample, alone or not): it never asks whether the alone answers are right. Additive mode (decision
  D1): five checks — certified, novel, novel_risk, probe and, with the loop's usefulness certificate on (default),
  useful: the candidate's selective certificate at β (`AdapterLoop(useful_error=0.5)`) holds at its certified
  threshold, so a layer that is wrong on most of the few answers it gives alone (a layer trained on noise, whose
  scores collapse) never starts serving; no min_share and no margin. A certified candidate that is not useful by
  `trial` decisions after its freeze is discarded with "useful" among the failed checks.
- discard — (probe on) the probe shows the gap beyond noise (Wilson lower bound of the novel error − Wilson upper bound of the
  seen error > gap, at `wilson` confidence; a point-estimate gap at n = 30 does not discard); not certified
  `stale` (3000) decisions after its freeze; certified and not promotable `trial` (3000) decisions after its freeze
  (`LoopState.since_freeze`). Both clocks count decisions, not coin samples (author's decision 14, 2026-10-24): at a
  rate ρ ≈ 0.08–0.17, 1 500 samples were 9 000–18 000 decisions, so a certified candidate that never cleared the
  margin stayed in trial for a whole 12 000-decision stream and no later training started. 3 000, not 1 500: the
  grid opens 200 decisions after the freeze and a 0–1 %-risk point certifies after about 120–280 samples (δ_v of the
  2nd–4th version) at 6–12 decisions per sample, i.e. 900–3 600 decisions after the freeze; 1 500 would discard a
  zero-risk candidate at 12 decisions per sample (≈ 1 700), 3 000 lets a 1 %-risk point of the first candidate
  certify at any of these rates (≤ 2 840) and the 2nd–4th up to 10 decisions per sample. The clock runs from the
  freeze: an alarm's restart drops the paired samples but does not restart it.
- maturity gate (`mature_patience`, "auto": 150 in additive mode, off in replacement mode; an int: on in both; None:
  off) — no training while the gated certificate (the deepest layer in additive mode, the served version otherwise) is
  still loosening: a "certified" / "loosened" event within its last `mature_patience` coin samples, or its first
  uncertified grid point's log-wealth set a new running maximum within them (it is still climbing). Freezing a
  threshold that would still loosen gives up coverage the version would have earned (in additive mode the attach
  freezes the deepest layer's routing threshold Λ; its answer threshold keeps loosening, so what an early attach
  changes is who answers, not the coverage — kept as pre-registered). Overrides: an alarm on that version since the
  latest freeze (drift response unchanged: alarm → restart → train), or ≥ `backlog_max` (None: 2 × `first`, 600) accepted labels since the
  latest freeze (the labels are paid for). The label floors and the budget still apply after it. The gate's inputs
  and the reason it opened (`maturity_gate(st)["trigger"]`: "mature" / "alarm" / "backlog") are in why_not_train and
  in the loop's "train_start" / "train" events ("gate"). On by default in additive mode only: in replacement mode it
  changes when trainings start, and replacement records stay as before unless asked (DESIGN_DELTA_CADENCE §3).
- alarm — "recertify"; a second alarm on the current serving of the served version (alarms at n ≥ served_since)
  within `probation` (300) decisions → "rollback:<v>", v the highest retired version below the served one with no
  alarm on record (the versions rolled back from were alarmed); no such version → "recertify".
- watch — label mode only, from a promotion until the served version changes again: on each calib or held-out
  label the served version and the version served before it are scored on the same input (+1 alone and right, −1
  alone and wrong, 0 not alone, each at its own threshold); after ≥ `watch_min` (100) such labels the served version
  is rolled back when an anytime-valid e-process on the discordant labels (`Mixture(0.5)` fed 1 when the served
  version wins one, 0 when the target does) reaches wealth 1 / (1 − `wilson`), the target won more of them, and the
  margin per label is ≥ `min_gain` (`watch_rollback`; watch=False: off). P(a false rollback) ≤ 1 − `wilson` per
  promotion at any horizon (Ville); simulated, not a stand: 0.8–2.1 % false rollbacks within 3 000 watched labels, a
  served version truly worse (p = 0.6, d = 0.1) caught in 70 % of runs by then. The base `LoopPolicy.watch` returns
  False (opt-in).

Unmeasured: no stand has run on this policy; its numbers are the design's, not tuned."""
from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass, field
from typing import Any

from ...core.sources import TRUSTED_SOURCES
from . import gates

SLOW_SOURCES = ("s2", "verified")
STATES = ("served", "candidate", "retired", "discarded", "evicted", "layer", "detached")   # layer, detached: additive


# ───────────────────────────────────────────────────────────── the budget
@dataclass
class Budget:
    """What trainings may cost. max_trainings: over the loop's life (None: no cap); per_day: trainings per 24 h;
    min_interval_s: between two trainings; max_train_s: one training (estimated before it starts; refused if above; the
    runner stops a training at its estimate + the runner's grace_s, which must cover what the estimate leaves out:
    loading the checkpoint, calibration and saving);
    cpu_threads: torch threads of the runner on a CPU; device: "auto" (cuda when present and ≥ free_gpu_mb free, else
    cpu), "cpu", "cuda" or "cuda:<i>"; max_examples: rows per training; max_versions_kept: versions held in memory
    (served and retired, version 0 counted): past it the loop evicts the oldest retired ones (released, no longer a
    rollback target; files on disk are never deleted) — never the served version, the rollback target or version 0. per_day=0 or max_trainings=0 means "never train"; max_trainings
    counts failed trainings too."""
    max_trainings: int | None = None
    per_day: int = 24
    min_interval_s: float = 600.0
    max_train_s: float = 1800.0
    cpu_threads: int = 4
    device: str = "auto"
    free_gpu_mb: int = 3000
    max_examples: int = 20_000
    max_versions_kept: int = 10

    def __post_init__(self):
        def count(name, v, low):
            if isinstance(v, bool) or not isinstance(v, int) or v < low:
                raise ValueError(f"Budget {name} must be an integer ≥ {low}, not {v!r}")
        if self.max_trainings is not None:
            count("max_trainings", self.max_trainings, 0)
        for name, low in (("per_day", 0), ("cpu_threads", 1), ("free_gpu_mb", 0), ("max_examples", 1),
                          ("max_versions_kept", 1)):
            count(name, getattr(self, name), low)
        for name in ("min_interval_s", "max_train_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0 or math.isinf(v):
                raise ValueError(f"Budget {name} must be a finite number of seconds ≥ 0, not {v!r}")
        d = self.device
        if not isinstance(d, str) or not (d in ("auto", "cpu", "cuda") or (d.startswith("cuda:")
                                                                            and d[5:].isdigit())):
            raise ValueError(f"Budget device must be 'auto', 'cpu', 'cuda' or 'cuda:<i>', not {d!r}")

    def refuse(self, started, now: float, estimate_s: float | None = None) -> str | None:
        """Why a training may not start now, or None. started: the wall-clock start times of the trainings so far;
        estimate_s: the estimated seconds of this one (None: not estimated yet). per_day is a rolling 24 h window
        (now − t < 86 400 s), no calendar day, no time zone. A start time after `now` (a clock set back, a checkpoint
        restored on another machine) counts towards max_trainings but not the window or the interval."""
        if self.max_trainings is not None and len(started) >= self.max_trainings:
            return f"budget: {len(started)} training(s) of max_trainings {self.max_trainings}"
        started = sorted(t for t in started if t <= now)
        day = sum(1 for t in started if now - t < 86_400)
        if day >= self.per_day:
            return f"budget: {day} training(s) in the last 24 h of per_day {self.per_day}"
        if started and now - started[-1] < self.min_interval_s:
            return f"budget: {now - started[-1]:.0f} s since the last training < min_interval_s {self.min_interval_s:g}"
        if estimate_s is not None and estimate_s > self.max_train_s:
            return f"budget: estimated {estimate_s:.0f} s > max_train_s {self.max_train_s:g}"
        return None


# ───────────────────────────────────────────────────────────── what the hooks read
@dataclass(frozen=True)
class VersionInfo:
    """One row of the versions table. state: "served", "candidate", "retired", "discarded" or "evicted" (a retired
    version past Budget.max_versions_kept: released, no longer a rollback target); n_train: the train
    labels it was trained on (version 0: 0 unless stated); samples: post-freeze samples it has read; certified and
    threshold: its promise now; frozen_at: the decision it was frozen at (None for version 0); train_keys: the content
    keys it trained on, a set snapshotted at the freeze (not a digest: novel / seen is membership, `key in
    train_keys`, for every post-freeze sample; the learner's key list grows after the freeze and must not be read
    live). A list or set given is stored as a frozenset. discarded_at: the train labels the loop had when this
    version was discarded (None: not discarded) — the lane LearningLoop's refreeze floor reads it. Additive mode with
    the usefulness certificate on (`AdapterLoop(useful_error=)`): certified / threshold are the version's own
    certificate's (λ_J), useful whether the usefulness certificate holds at λ_J, useful_thr the threshold it answers
    at (composed; +inf: no point certified by both)."""
    version: int
    state: str
    n_train: int = 0
    samples: int = 0
    certified: bool = False
    threshold: float = math.inf
    fingerprint: str | None = None
    frozen_at: int | None = None
    train_keys: frozenset = frozenset()
    discarded_at: int | None = None
    layer: int | None = None                           # additive mode: index in the chain (None: not a layer)
    routing_thr: float | None = None                   # additive mode: its frozen routing threshold Λ (None: none)
    learning: bool | None = None                       # additive mode: its promise learns — True for every layer and
    #                                                    the candidate since decision 21 (None: replacement mode)
    useful: bool | None = None                         # additive, usefulness on: its usefulness certificate holds at
    #                                                    `threshold` (its certificate's λ_J); None: no such certificate
    useful_thr: float | None = None                    # additive, usefulness on: the composed answer threshold (the
    #                                                    loosest point of the certified prefix that is also useful)

    def __post_init__(self):
        if not isinstance(self.train_keys, frozenset):
            object.__setattr__(self, "train_keys", frozenset(self.train_keys))


@dataclass(frozen=True)
class LoopState:
    """What a hook sees, rebuilt by the loop for every call (changing it changes nothing in the loop). n: decisions
    so far; served / candidate: version numbers (candidate None: none in trial; `since_freeze`: the decisions since
    the candidate's freeze); labels: label counts by split;
    versions: the versions table; holdout: the served version's [(alone, wrong)] on its latest held-out labels, oldest
    first; alpha: the promised level; alarms: [{"n", "who", "version"}] so far; served_history: the versions served,
    in order (a rollback appends); served_since: the decision at which the current serving of `served` began (a
    promotion or a rollback sets it); trainings: the wall-clock start times of the trainings so far; job: the training
    in flight ({"id", "started", "eta"}) or None; now: the wall clock; paused; mode: "stream" (a Dispatcher, the stream
    promise) or "label" (a System alone: the pre-freeze gates only); promises, detector: their state dumps, for a
    policy that reads them; keys: the distinct content keys with a label, by split (a relabelled key counts once, as
    the pre-freeze report's latest label per key; None: a state built without them, `labels` stands in). maturity:
    the inputs of `DefaultPolicy.maturity_gate` — version (the deepest layer in additive mode, else the served
    version), samples (its promise's samples read), changed_at (the samples at its latest "certified" / "loosened"
    event since its latest restart; None: none), peak_at (`StreamPromise.peak_at`), last_freeze (the decision of the
    latest training's freeze or birth discard, 0: none), labels_since_freeze (accepted labels after it), alarm (an
    alarm on that version after it); None in label mode or when that version has no promise.

    `stream.loop.AdapterLoop` passes copies of the dict fields and resets `holdout` at a promotion or a rollback (the
    new served version does not inherit the old one's errors). The state has dict fields, so `hash(st)` raises: do not
    use states as keys."""
    n: int = 0
    served: int = 0
    candidate: int | None = None
    labels: dict = field(default_factory=lambda: {s: 0 for s in gates.SPLITS})
    versions: tuple = (VersionInfo(0, "served"),)
    holdout: tuple = ()
    alpha: float = 0.05
    alarms: tuple = ()
    served_history: tuple = (0,)
    served_since: int = 0
    trainings: tuple = ()
    job: dict | None = None
    now: float = 0.0
    paused: bool = False
    mode: str = "stream"
    promises: dict = field(default_factory=dict)
    detector: dict | None = None
    keys: dict | None = None
    layers: tuple | None = None                        # additive mode: the chain's versions, L0 first (None: off)
    deepest: int | None = None                         # additive mode: the deepest layer
    deepest_certified: bool | None = None              # additive mode: the deepest layer's promise is certified
    zone_share: dict | None = None                     # additive mode: layer version → P̂(Z_v), its zone's share
    maturity: dict | None = None                       # the maturity gate's inputs (`DefaultPolicy.maturity_gate`)
    useful_error: float | None = None                  # additive mode: β of the usefulness certificate (None: off)

    def __post_init__(self):
        if self.mode not in ("stream", "label"):
            raise ValueError(f"mode is 'stream' or 'label', not {self.mode!r}")

    @property
    def since_freeze(self) -> int | None:
        """Decisions since the candidate was frozen: n − its frozen_at (None: no candidate, or a candidate row
        built without frozen_at). DefaultPolicy's `stale` / `trial` clocks read it."""
        c = self.candidate_info
        return None if c is None or c.frozen_at is None else self.n - c.frozen_at

    def version(self, v) -> VersionInfo:
        for row in self.versions:
            if row.version == v:
                return row
        raise KeyError(f"no version {v!r} in the versions table")

    @property
    def served_info(self) -> VersionInfo:
        return self.version(self.served)

    @property
    def candidate_info(self) -> VersionInfo | None:
        return None if self.candidate is None else self.version(self.candidate)

    @property
    def next_version(self) -> int:
        return 1 + max((r.version for r in self.versions), default=-1)


@dataclass(frozen=True)
class LabelRow:
    """One stored label: the input's content key, the input, the label, its source, the decision it came at."""
    key: str
    x: Any
    y: Any
    source: str
    t: int


@dataclass(frozen=True)
class TrainReport:
    """What a training produced, scored on labels the candidate never trained on. n_holdout: held-out labels asked;
    right / wrong: the candidate's answers alone and right / wrong on them, served_right / served_wrong the served
    version's; better / worse: held-out labels where the candidate's item score (+1 alone and right, −1 alone and
    wrong, 0 escalated) beats / trails the served one's; threshold: act_guard's on the calib split (None: none found
    at the promised risk; in stage 2 a fixed-set stand-in, while the served version answers at its stream threshold,
    so the held-out comparison sets two kinds of threshold side by side); honesty_before / honesty_after: the served version's and the candidate's
    `solvi.testing.honesty` numbers on the held-out labels; seconds: the training's wall time; n_train: rows trained;
    items: optional per-label scores ((key, candidate score, served score), each +1 / −1 / 0) for a policy that runs
    its own paired test (McNemar, bootstrap); empty when the loop does not provide them."""
    n_holdout: int
    right: int
    wrong: int
    served_right: int
    served_wrong: int
    better: int = 0
    worse: int = 0
    threshold: float | None = None
    honesty_before: dict = field(default_factory=dict)
    honesty_after: dict = field(default_factory=dict)
    seconds: float = 0.0
    n_train: int = 0
    items: tuple = ()


@dataclass(frozen=True)
class Paired:
    """The candidate against the served version on the candidate's post-freeze samples (stream mode). Novel: samples
    whose key is not in the candidate's train_keys — n_novel of them, alone_novel / wrong_novel the candidate's
    answers alone / alone and wrong there, alone_served / wrong_served the served version's on the same novel samples
    at the threshold of its comparison certificate (1.4.0: a `StreamPromise` built on exactly the candidate's samples
    with the same grid rule and the candidate's δ_v; it only ranks, it never serves) once that certificate is
    certified, and at the served version's own threshold while it is not (thr_s: "cmp", "own" or "no_model" — a served
    version without a model never answers alone), alone_served_own / wrong_served_own at the served version's own
    threshold in force now (the comparison before 1.4.0, kept for the records; None when not counted). Seen: the samples whose key it trained on — n_seen,
    the candidate's alone_seen / wrong_seen — read only by the memorisation probe. A candidate with no sample yet is Paired(0, 0, 0, 0), not None.

    Exact-key identity: a paraphrase of a trained item counts as novel, and calib keys (the threshold was fitted on
    them) recur as novel — a small optimism on share_k."""
    n_novel: int
    alone_novel: int
    wrong_novel: int
    alone_served: int
    n_seen: int = 0
    alone_seen: int = 0
    wrong_seen: int = 0
    wrong_served: int = 0
    alone_served_own: int | None = None
    wrong_served_own: int | None = None
    thr_s: str | None = None

    @property
    def share_k(self) -> float:
        return self.alone_novel / self.n_novel if self.n_novel else 0.0

    @property
    def share_s(self) -> float:
        return self.alone_served / self.n_novel if self.n_novel else 0.0

    @property
    def share_s_own(self) -> float | None:
        """The served version's share alone at its own threshold (the old comparison); None when not counted."""
        if self.alone_served_own is None:
            return None
        return self.alone_served_own / self.n_novel if self.n_novel else 0.0

    @property
    def err_k_novel(self) -> float | None:
        return self.wrong_novel / self.alone_novel if self.alone_novel else None

    @property
    def err_s_novel(self) -> float | None:
        return self.wrong_served / self.alone_served if self.alone_served else None

    @property
    def err_k_seen(self) -> float | None:
        return self.wrong_seen / self.alone_seen if self.alone_seen else None


@dataclass(frozen=True)
class Watch:
    """Label mode, after a promotion: the served version against the version served before it (`target`, the
    rollback target) on the same labels since the promotion — calib and held-out labels only (keys no version trained
    on). Each label scores each version +1 (answered alone and right at its own threshold), −1 (alone and wrong) or 0
    (not alone). n: the labels compared; better: those where the target scored higher; worse: those where the served
    version scored higher; margin: Σ (target − served) / n (0.0 with n = 0). log_e: the log wealth of the watch's
    e-process (`watch_eprocess()`, a `Mixture(0.5)` fed loss 1 when the served version wins a discordant label and
    0 when the target wins), 0.0 before the first discordant label."""
    target: int
    n: int = 0
    better: int = 0
    worse: int = 0
    total: int = 0

    log_e: float = 0.0

    @property
    def margin(self) -> float:
        return self.total / self.n if self.n else 0.0


# ───────────────────────────────────────────────────────────── the abstract policy
class LoopPolicy(abc.ABC):
    """See the module docstring. Subclass and give every hook; `budget` (a Budget) is read by the loop too."""

    def __init__(self, budget: Budget | None = None):
        self.budget = budget

    @property
    def budget(self) -> Budget:
        """The Budget (the defaults when none was set, also for a subclass whose __init__ skips super().__init__)."""
        b = self.__dict__.get("_budget")
        return b if b is not None else Budget()

    @budget.setter
    def budget(self, budget):
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError(f"budget is a Budget, not {type(budget).__name__}")
        self._budget = budget if budget is not None else Budget()

    # --- collecting
    @abc.abstractmethod
    def accept(self, st: LoopState, x, y, source: str) -> bool:
        """Take this label at all."""

    @abc.abstractmethod
    def split(self, st: LoopState, key: str) -> str:
        """"train" | "calib" | "holdout" for a content key; the same key must always get the same split."""

    # --- training
    @abc.abstractmethod
    def want_train(self, st: LoopState) -> bool:
        """Ask for a training now."""

    @abc.abstractmethod
    def select(self, st: LoopState, rows) -> list:
        """The train rows (LabelRow) that go into this training."""

    @abc.abstractmethod
    def hyper(self, st: LoopState) -> dict:
        """The training's settings: r, lr, updates, seed, ..."""

    # --- freezing
    @abc.abstractmethod
    def freeze(self, st: LoopState, report: TrainReport) -> bool:
        """Freeze the trained model as the candidate (the pre-freeze gates on labels it never trained on). In label mode
        this is the only measurement before a version is served (the candidate is promoted on the next label, and only
        the watch after it can roll it back): a freeze that measures nothing serves every training."""

    # --- promotion / rollback
    @abc.abstractmethod
    def promote(self, st: LoopState, paired: Paired | None) -> bool:
        """Serve the candidate now (paired None in label mode; in stream mode a candidate with no sample yet gets
        Paired(0, 0, 0, 0), not None)."""

    @abc.abstractmethod
    def discard(self, st: LoopState, paired: Paired | None) -> str | None:
        """Why the candidate is discarded now, or None to keep trialling (paired as for promote)."""

    @abc.abstractmethod
    def alarm(self, st: LoopState, who: str, report: dict) -> str:
        """An alarm on the served version (who: "detector" or "monitor") → "recertify", "rollback:<v>" or "ignore"."""

    # --- label mode: the watch after a promotion (not abstract, opt-in: a policy without it never rolls back)
    def watch(self, st: LoopState, w: Watch) -> bool:
        """Label mode: roll the served version back to `w.target` now? Called on each calib or held-out label while a
        watch runs (from a promotion to the next change of the served version). This default returns False: a custom
        policy has no automatic rollback unless it defines this hook (e.g. `return watch_rollback(w)`, the
        `DefaultPolicy` rule). The loop keeps the watch's counts either way (`status()["watch"])."""
        return False


# ───────────────────────────────────────────────────────────── the out-of-the-box policy
class DefaultPolicy(LoopPolicy):
    """See the module docstring for each hook. budget: a Budget (None: the defaults); s2_labels: take slow-path labels;
    calib / holdout: the split's shares; first, growth: the train labels before the first / a next training (absolute,
    × the served version's n_train); window, target_share: the improvement signal (target_share 0.5: also short when
    it answers alone on less than half its held-out labels; None: the error-only signal); min_gain, min_holdout, tolerance:
    the pre-freeze gates; wilson: the confidence of the Wilson bounds (label mode's paired gain, the probe's discard
    test); min_novel, min_share, margin: the
    promotion rule; gap, probe_min: the memorisation probe; stale, trial: the decisions after its freeze at which a
    candidate not certified / certified and not promotable is discarded (decisions, not coin samples); probation:
    two alarms closer than this roll back (0: never); r, lr, min_updates, max_updates: the training's settings;
    refreeze: the lane LearningLoop's retrain floor instead of the largest n_train (None: off); probe: the memorisation
    probe on (False: off, promotion and discard); pre_freeze: the pre-freeze gates on (False: every training is
    frozen); min_calib: the calib labels a training waits for (None: the fewest the calib threshold needs at α, 73 at
    α = 0.05; 0 for a learner with its own threshold); novel_risk: promotion needs the Wilson upper bound (at
    `wilson`) of wrong_novel / n_novel ≤ α (False: off); watch, watch_min: label mode's watch after a promotion (False:
    off; the labels it needs before it may roll back); novel_only: the promotion's paired comparison counts only samples
    on keys the candidate never trained on (False: every sample, its train keys too — for controls; weakens the
    guarantee: a memoriser's trained items then count for it). In label mode pre_freeze=False is refused by the loop:
    there the pre-freeze gates are the only measurement before a version is served. mature_patience, backlog_max: the
    maturity gate (see the module docstring)."""

    def __init__(self, budget: Budget | None = None, *, s2_labels: bool = False, calib: float = 0.15,
                 holdout: float = 0.15, first: int = 300, growth: float = 1.5, window: int = 200, target_share=0.5,
                 min_gain: float = 0.01, min_holdout: int = 100, tolerance: float = 0.02, wilson: float = 0.95,
                 min_novel: int = 100, min_share: float = 0.05, margin: float = 0.01, gap: float = 0.10,
                 probe_min: int = 30, stale: int = 3000, trial: int = 3000, probation: int = 300, r: int = 8,
                 lr: float = 3e-4, min_updates: int = 200, max_updates: int = 3000, refreeze=None, probe: bool = True,
                 pre_freeze: bool = True, novel_risk: bool = True, min_calib: int | None = None, watch: bool = True,
                 watch_min: int = 100, novel_only: bool = True, mature_patience="auto", backlog_max: int | None = None):
        super().__init__(budget)
        gates.split("", calib=calib, holdout=holdout)                     # the shares, checked once
        for name, v in (("first", first), ("window", window), ("min_holdout", min_holdout), ("min_novel", min_novel),
                        ("probe_min", probe_min), ("stale", stale), ("trial", trial), ("probation", probation),
                        ("r", r), ("min_updates", min_updates), ("max_updates", max_updates),
                        ("watch_min", watch_min)):
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if name == "probation" else 1):
                raise ValueError(f"{name} must be an integer ≥ {0 if name == 'probation' else 1}, not {v!r}")
        if min_updates > max_updates:
            raise ValueError(f"min_updates ({min_updates}) > max_updates ({max_updates})")
        for name, v in (("growth", growth), ("refreeze", 1 if refreeze is None else refreeze)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v < math.inf:
                raise ValueError(f"{name} must be a number ≥ 1, not {v!r}")
        for name, v in (("min_gain", min_gain), ("tolerance", tolerance), ("min_share", min_share),
                        ("margin", margin), ("gap", gap)):
            gates._share(name, v, closed=True)
        if target_share is not None:
            gates._share("target_share", target_share, closed=True)
        if not 0 < wilson < 1:
            raise ValueError(f"wilson is a confidence in (0, 1), not {wilson!r}")
        if not lr > 0 or math.isinf(lr):
            raise ValueError(f"lr is a positive step, not {lr!r}")
        self.s2_labels = bool(s2_labels)
        self.calib, self.holdout = float(calib), float(holdout)
        self.first, self.growth, self.window, self.target_share = first, float(growth), window, target_share
        self.min_gain, self.min_holdout, self.tolerance, self.wilson = min_gain, min_holdout, tolerance, wilson
        self.min_novel, self.min_share, self.margin = min_novel, min_share, margin
        self.gap, self.probe_min, self.stale, self.trial, self.probation = gap, probe_min, stale, trial, probation
        self.r, self.lr, self.min_updates, self.max_updates = r, float(lr), min_updates, max_updates
        self.refreeze = None if refreeze is None else float(refreeze)
        self.probe, self.pre_freeze, self.novel_risk = bool(probe), bool(pre_freeze), bool(novel_risk)
        if min_calib is not None and (isinstance(min_calib, bool) or not isinstance(min_calib, int) or min_calib < 0):
            raise ValueError(f"min_calib must be an integer ≥ 0 or None, not {min_calib!r}")
        self.min_calib = min_calib
        self.watch_on, self.watch_min = bool(watch), watch_min
        self.novel_only = bool(novel_only)
        mp = mature_patience
        if not (mp is None or mp == "auto" or (isinstance(mp, int) and not isinstance(mp, bool) and mp >= 1)):
            raise ValueError(f"mature_patience is 'auto', None or a number of coin samples ≥ 1, not {mp!r}")
        if backlog_max is not None and (isinstance(backlog_max, bool) or not isinstance(backlog_max, int)
                                        or backlog_max < 1):
            raise ValueError(f"backlog_max is a number of labels ≥ 1 or None, not {backlog_max!r}")
        self.mature_patience, self.backlog_max = mature_patience, backlog_max

    # --- collecting
    def accept(self, st, x, y, source) -> bool:
        return source in TRUSTED_SOURCES or (self.s2_labels and source in SLOW_SOURCES)

    def split(self, st, key) -> str:
        return gates.split(key, calib=self.calib, holdout=self.holdout)

    # --- training
    def why_not_train(self, st) -> str | None:
        """Why want_train says no, or None when it says yes."""
        if st.paused:
            return "paused"
        if st.job is not None:
            return "training in progress"
        if st.candidate is not None:
            return f"version {st.candidate} is in trial"
        if st.deepest_certified is False:                     # additive: a candidate attaches below a certified layer
            return f"the deepest layer (version {st.deepest}) is not certified: a candidate cannot attach below it"
        gate = self.maturity_gate(st)
        if gate is not None and gate["trigger"] is None:
            return (f"maturity gate: version {gate['version']} is not mature (samples {gate['samples']}, last "
                    f"certified/loosened at {gate['changed_at']}, first uncertified point's peak at {gate['peak_at']}, "
                    f"patience {gate['patience']}); no alarm since the freeze at decision {gate['last_freeze']}; "
                    f"{gate['backlog']} label(s) since it < {gate['backlog_max']}")
        have = st.labels.get("train", 0)
        if self.refreeze is None:
            top = max((r.n_train for r in st.versions), default=0)      # discarded versions too: a retrain floor
            need = max(self.first, math.ceil(self.growth * top))
            why = f"first {self.first}, growth {self.growth:g} × the largest n_train of any version, {top}"
        else:
            last = max((r.discarded_at for r in st.versions if r.discarded_at is not None), default=0)
            n_s = st.served_info.n_train
            need = max(self.first, math.ceil(self.growth * n_s), math.ceil(self.refreeze * last))
            why = (f"first {self.first}, growth {self.growth:g} × the served n_train {n_s}, refreeze "
                   f"{self.refreeze:g} × {last} at the latest discard")
        if have < need:
            return f"{have} train label(s) < {need} ({why})"
        if self.pre_freeze:                                    # the gates' own floors: a freeze that cannot pass
            # distinct keys, as the report counts them (latest label per key): a repeated label is not a new one
            distinct = st.labels if st.keys is None else st.keys
            hold = distinct.get("holdout", 0)
            if hold < self.min_holdout:
                return f"{hold} held-out label(s) < {self.min_holdout}: a freeze cannot pass yet"
            calib, need_c = distinct.get("calib", 0), self.calib_floor(st.alpha)
            if calib < need_c:
                return f"{calib} calib label(s) < {need_c}: act_guard cannot find a threshold yet"
        refused = self.budget.refuse(st.trainings, st.now)
        if refused:
            return refused
        outcomes = list(st.holdout)[-self.window:]
        need_o = min(self.min_holdout, self.window)
        if self.holdout > 0 and len(outcomes) < need_o:
            return (f"no improvement signal yet: {len(outcomes)} held-out outcome(s) of the served version < "
                    f"{need_o}")
        sig = gates.incumbent_short(outcomes, st.alpha, self.target_share, min_n=need_o)
        if not sig["short"]:
            return f"no improvement signal: {sig['why']}"
        return None

    def maturity_gate(self, st) -> dict | None:
        """The maturity gate on `st` (see the module docstring): None when off (mature_patience None, "auto" in
        replacement mode, label mode, no gated promise), else its inputs and `trigger` — "mature", "alarm" (override
        A), "backlog" (override B) or None (closed: no training)."""
        patience = getattr(self, "mature_patience", None)           # a policy pickled before the gate: off
        if patience == "auto":
            patience = 150 if st.layers is not None else None
        m = st.maturity
        if patience is None or m is None or st.mode == "label":
            return None
        backlog_max = getattr(self, "backlog_max", None) or 2 * self.first
        samples, changed, peak = m["samples"], m["changed_at"], m["peak_at"]
        loosening = changed is not None and samples - changed < patience
        climbing = peak is not None and samples - peak < patience
        backlog = int(m["labels_since_freeze"])
        if not (loosening or climbing):
            trigger = "mature"
        elif m["alarm"]:
            trigger = "alarm"
        elif backlog >= backlog_max:
            trigger = "backlog"
        else:
            trigger = None
        return {"version": m["version"], "patience": patience, "samples": samples, "changed_at": changed,
                "peak_at": peak, "loosening": loosening, "climbing": climbing, "alarm": bool(m["alarm"]),
                "last_freeze": m["last_freeze"], "backlog": backlog, "backlog_max": backlog_max, "trigger": trigger}

    def calib_floor(self, alpha) -> int:
        """The calib labels a freeze needs: min_calib, or (None) `gates.calib_min(alpha, wilson)`, the fewest with
        which the calib threshold can exist (0: the learner brings its own threshold)."""
        return gates.calib_min(alpha, confidence=self.wilson) if self.min_calib is None else self.min_calib

    def want_train(self, st) -> bool:
        return self.why_not_train(st) is None

    def select(self, st, rows) -> list:
        """One row per content key — a relabelled input keeps only its latest label (by t; on a tie the later row) —
        then the class-balanced, half-recent sample (see the module docstring)."""
        latest: dict = {}
        for r in sorted(rows, key=lambda r: r.t):
            latest.pop(r.key, None)
            latest[r.key] = r
        rows = list(latest.values())
        k = self.budget.max_examples
        if len(rows) <= k:
            return rows
        rng = random.Random(f"select|{st.next_version}")
        recent_n = 2 * st.served_info.n_train
        recent, older = (rows[-recent_n:], rows[:-recent_n]) if recent_n else ([], rows)
        take_recent = min(len(recent), k // 2) if older else min(len(recent), k)
        picked = _balanced(recent, take_recent, rng)
        picked += _balanced(older, k - len(picked), rng)
        if len(picked) < k:                                    # older ran out: fill from the recent rows left
            chosen = {id(r) for r in picked}
            picked += _balanced([r for r in recent if id(r) not in chosen], k - len(picked), rng)
        return sorted(picked, key=lambda r: r.t)

    def hyper(self, st) -> dict:
        """updates from the train label count capped at max_examples — an approximation of the rows `select` returns
        (it does not see them, and a relabelled key counts once there). Seeded by the next version number: a retry
        after a failed training is the same training (deterministic on purpose)."""
        n = min(st.labels.get("train", 0), self.budget.max_examples)
        updates = min(self.max_updates, max(self.min_updates, math.ceil(2 * n / 8)))
        return {"r": self.r, "lr": self.lr, "updates": updates, "seed": st.next_version, "warm_start": False}

    # --- freezing
    def freeze_gates(self, st, report: TrainReport) -> dict:
        """Each pre-freeze gate's result → {"heldout", "act_guard", "honesty", "time"[, "paired_gain"]}, each with
        "ok" and "why"."""
        out = {"heldout": gates.heldout_gate(report.right, report.wrong, report.served_right, report.served_wrong,
                                             report.n_holdout, min_gain=self.min_gain, min_holdout=self.min_holdout)}
        thr = report.threshold
        out["act_guard"] = {"ok": thr is not None and math.isfinite(thr), "threshold": thr,
                            "why": "act_guard found a threshold on the calib labels" if thr is not None
                            and math.isfinite(thr) else "act_guard found no threshold at the promised risk"}
        out["honesty"] = gates.honesty_gate(report.honesty_before, report.honesty_after, self.tolerance)
        out["time"] = {"ok": report.seconds <= self.budget.max_train_s, "seconds": report.seconds,
                       "why": f"trained in {report.seconds:.0f} s (max_train_s {self.budget.max_train_s:g})"}
        if st.mode == "label":
            low = gates.paired_gain_low(report.better, report.worse, self.wilson)
            out["paired_gain"] = {"ok": low is not None and low > 0, "low": low,
                                  "why": f"paired held-out gain {report.better} better / {report.worse} worse, Wilson "
                                         f"lower bound {'none' if low is None else f'{low:+.3f}'} (> 0 needed)"}
        return out

    def freeze(self, st, report) -> bool:
        if not self.pre_freeze:
            return True
        return all(g["ok"] for g in self.freeze_gates(st, report).values())

    # --- promotion / rollback
    def promote_checks(self, st, paired: Paired) -> dict:
        """Each promotion condition → {"certified", "novel", "min_share", "margin", "novel_risk", "probe"}, each with
        "ok" (novel_risk / probe only when on). The probe
        passes only with ≥ probe_min alone answers on novel samples (an unmeasured novel side does not promote).
        Additive mode (st.layers set): the additive gate (decision D1) — certified, novel, novel_risk, probe; no
        min_share and no margin (nothing to compare with: a layer serves only the zone the layer above escalates);
        with the loop's usefulness certificate on (st.useful_error set) a fifth check, "useful": the candidate's
        usefulness certificate (selective error among its alone answers ≤ β, anytime-valid on its own δ budget) holds
        at its certificate's threshold in force λ_J — a layer whose alone answers are not shown to be right more
        often than β allows never starts serving."""
        c = st.candidate_info
        out = {"certified": {"ok": bool(c is not None and c.certified)},
               "novel": {"ok": paired.n_novel >= self.min_novel, "n_novel": paired.n_novel, "min": self.min_novel}}
        if st.layers is None:
            out["min_share"] = {"ok": paired.share_k >= self.min_share, "share_k": paired.share_k}
            out["margin"] = {"ok": paired.share_k >= paired.share_s + self.margin, "share_k": paired.share_k,
                             "share_s": paired.share_s, "share_s_own": paired.share_s_own, "thr_s": paired.thr_s}
        if self.novel_risk:
            risk = paired.wrong_novel / paired.n_novel if paired.n_novel else None
            ub = gates._wilson(paired.wrong_novel, paired.n_novel, self.wilson)[1] if paired.n_novel else None
            out["novel_risk"] = {"ok": ub is not None and ub <= st.alpha, "risk": risk, "ub": ub,
                                 "n_novel": paired.n_novel, "alpha": st.alpha,
                                 "why": "no novel sample yet" if risk is None else
                                 f"wrong alone on novel samples {paired.wrong_novel} / {paired.n_novel} = {risk:.4f}, "
                                 f"Wilson {self.wilson:.0%} upper bound {ub:.4f} (≤ α {st.alpha:g} needed)"}
        if self.probe:
            out["probe"] = self._probe_promote(paired)
        beta = getattr(st, "useful_error", None)
        if st.layers is not None and beta is not None:
            ok = bool(c is not None and c.useful)
            thr = None if c is None or c.useful_thr is None or math.isinf(c.useful_thr) else c.useful_thr
            out["useful"] = {"ok": ok, "beta": beta, "thr": thr,
                             "why": (f"the usefulness certificate (error among the alone answers ≤ {beta:g}) holds at "
                                     "the certified threshold" if ok else
                                     f"the usefulness certificate (error among the alone answers ≤ {beta:g}) does not "
                                     "hold at the certified threshold yet")}
        return out

    def _probe(self, paired):
        return gates.probe(paired.alone_seen, paired.wrong_seen, paired.alone_novel, paired.wrong_novel, gap=self.gap,
                           min_n=self.probe_min, confidence=self.wilson)

    def _probe_promote(self, paired):
        pr = self._probe(paired)
        if paired.alone_novel >= self.probe_min:
            return {**pr, "novel_measured": True}
        return {**pr, "ok": False, "novel_measured": False,
                "why": f"{paired.alone_novel} alone answer(s) on novel samples < probe_min {self.probe_min}: the probe "
                       "cannot clear the candidate yet"}

    def promote(self, st, paired) -> bool:
        if st.candidate is None:
            return False
        if paired is None:
            return st.mode == "label"                  # label mode: the pre-freeze gates decided
        return all(g["ok"] for g in self.promote_checks(st, paired).values())

    def discard(self, st, paired) -> str | None:
        c = st.candidate_info
        if c is None or (paired is None and st.mode == "label"):
            return None
        if paired is None:                                        # stream mode, no sample yet
            paired = Paired(0, 0, 0, 0)
        pr = self._probe(paired) if self.probe else {"shown": False}
        if pr["shown"]:
            return f"memorisation probe: {pr['why']}"
        age = st.since_freeze                                     # decisions, not coin samples (decision 14)
        if age is None:
            return None
        if not c.certified and age >= self.stale:
            return f"not certified {age} decisions after its freeze ({c.samples} samples)"
        if c.certified and age >= self.trial and not self.promote(st, paired):
            if st.layers is not None:                             # additive: the gate, no comparison
                failed = [k for k, g in self.promote_checks(st, paired).items() if not g["ok"]]
                return (f"certified, the additive gate did not pass {age} decisions after its freeze ({c.samples} "
                        f"zone samples, {paired.n_novel} novel; failed: {', '.join(failed)})")
            own = "" if paired.share_s_own is None else f", share_s_own {paired.share_s_own:.4g}"
            own += "" if paired.thr_s is None else f", thr_s {paired.thr_s}"
            return (f"certified, not better {age} decisions after its freeze ({c.samples} samples; share_k "
                    f"{paired.share_k:.4g}, share_s {paired.share_s:.4g}{own}, {paired.n_novel} novel)")
        return None

    def watch(self, st, w) -> bool:
        """Label mode: `watch_rollback(w, min_n=watch_min, confidence=wilson, min_gain=min_gain)` (the e-process at
        level 1 − wilson); never with watch=False."""
        if not getattr(self, "watch_on", True):          # a policy pickled before the watch existed: on
            return False
        return watch_rollback(w, min_n=getattr(self, "watch_min", 100), confidence=self.wilson,
                              min_gain=self.min_gain)

    def rollback_target(self, st) -> int | None:
        """The highest retired version below the served one with no alarm on record, or None. Conservative: any alarm
        on record excludes a version, also one on an earlier serving that was recertified and then served well."""
        alarmed = {a.get("version") for a in st.alarms}
        ok = [r.version for r in st.versions if r.state == "retired" and r.version < st.served
              and r.version not in alarmed]
        return max(ok, default=None)

    def alarm(self, st, who, report) -> str:
        if st.layers is not None:                                 # additive: per layer
            v, j = report.get("version"), report.get("layer")
            last = max((a["n"] for a in st.alarms if a.get("version") == v), default=None)
            if j and last is not None and st.n - last <= self.probation:
                return f"rollback:{j}"                            # two alarms close on layer j ≥ 1: detach it
            return "recertify"                                    # L0 (or a first alarm): its promise restarts
        last = max((a["n"] for a in st.alarms if a.get("version") == st.served and a["n"] >= st.served_since),
                   default=None)
        if last is not None and st.n - last <= self.probation:
            target = self.rollback_target(st)
            if target is not None:
                return f"rollback:{target}"
        return "recertify"


def watch_eprocess():
    """The watch's e-process: `Mixture(0.5)` (stream._eprocess), the bet against "the served version wins at least half
    of the discordant labels". The loop feeds it loss = 1 when the served version scores higher on a calib or held-out
    label, 0 when the target does (concordant labels are not fed); its wealth is a test supermartingale while the
    served version is not worse."""
    from ._eprocess import Mixture
    return Mixture(0.5)


def watch_rollback(w: Watch, *, min_n: int = 100, confidence: float = 0.95, min_gain: float = 0.01) -> bool:
    """The watch's rule: the served version is shown worse than the target on the labels since its promotion — ≥ min_n
    labels compared, the e-process's wealth ≥ 1 / (1 − confidence) (`w.log_e`, `watch_eprocess()`: 20 at 0.95),
    better > worse, and the margin (target − served, per label) ≥ min_gain. Anytime-valid: the e-process is a test
    supermartingale while the served version wins at least half of the discordant labels, so by Ville's inequality
    P(a rollback ever, at any horizon) ≤ 1 − confidence per promotion when the served version is not worse on the
    discordant labels; the other conditions only make a rollback rarer.

    Simulation, not a stand (2 000 runs per cell; a watched label discordant with probability d ∈ {0.1, 0.3, 0.5}, a
    discordant one won by the target with probability p, steps 1 or 2): false rollbacks with both versions equally good
    (p = 0.5) 0.4–1.4 % within 300 watched labels, 0.5–1.9 % within 1 000, 0.8–2.1 % within 3 000; a served version
    truly worse (p = 0.6, d = 0.1) rolled back in 10 / 28 / 70 % of runs by 300 / 1 000 / 3 000 watched labels."""
    if w.n < min_n:
        return False
    return w.log_e >= math.log(1.0 / (1.0 - confidence)) and w.better > w.worse and w.margin >= min_gain


def _balanced(rows, k, rng) -> list:
    """≤ k rows, as even over the labels as the rows allow: each label's rows shuffled, then taken in turn."""
    if k <= 0 or not rows:
        return []
    by: dict = {}
    for r in rows:
        by.setdefault(repr(r.y), []).append(r)
    pools = [by[c] for c in sorted(by)]
    for p in pools:
        rng.shuffle(p)
    out, i = [], 0
    while len(out) < k and any(pools):
        p = pools[i % len(pools)]
        if p:
            out.append(p.pop())
        i += 1
    return out


__all__ = ["Budget", "DefaultPolicy", "LabelRow", "LoopPolicy", "LoopState", "Paired", "STATES", "SLOW_SOURCES",
           "TrainReport", "VersionInfo", "Watch", "watch_eprocess", "watch_rollback"]
