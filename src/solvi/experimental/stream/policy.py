"""The policy of the adapter loop: the abstract class a developer controls the loop with, the state its hooks read, the
budget, and the out-of-the-box policy (experimental; the LoRA learning loop, stage 1 — not wired into a loop yet).

    from solvi.experimental.stream import Budget, DefaultPolicy, LoopPolicy, LoopState

    policy = DefaultPolicy(budget=Budget(per_day=6, max_train_s=900))
    policy.split(st, key)                 # "train" | "calib" | "holdout": by the content key, fixed forever
    policy.want_train(st)                 # growth, the improvement signal, the budget; why_not_train(st) says why not
    policy.freeze(st, report)             # the pre-freeze gates on held-out labels (freeze_gates(st, report): each one)
    policy.promote(st, paired)            # certified, ≥ 100 novel samples, min_share, margin, the memorisation probe

`LoopPolicy` is the abstract class: every hook reads a `LoopState` (a frozen snapshot the loop rebuilds per call:
decisions so far, the served version and the candidate, label counts by split, the versions table, the served
version's latest held-out outcomes, alarms, trainings started, the wall clock, paused, the mode) and returns a plain
decision. A hook that raises stops the loop (status "error") and the served version keeps serving. A policy keeps its
own counters in `self` and must pickle (the loop's checkpoint saves it).

`DefaultPolicy` (each number a keyword):

- accept — a label from the trusted sources ("human", "outcome", "rule", "audit"); "s2" / "verified" only with
  s2_labels=True.
- split — `stream.gates.split`: sha256 of the content key → 70 / 15 / 15 train / calib / holdout.
- want_train — no candidate in trial, not paused; train labels ≥ `first` (300) and ≥ `growth` (1.5) × the served
  version's n_train; the budget allows one now; and the improvement signal (`gates.incumbent_short`): on its last
  `window` (200) held-out labels the served version answers nothing alone, or errs alone above the promised α, or
  (target_share given) answers alone less than that. A strong incumbent is not retrained for nothing.
- select — a class-balanced sample of ≤ budget.max_examples train rows, half of them from the latest 2 × the served
  version's n_train rows; seeded by the next version number, so the same rows give the same selection.
- hyper — r 8, lr 3e-4, updates = clamp(⌈2 n / 8⌉, 200, 3000) (n: the train labels, ≤ max_examples), seed = the next
  version number, warm_start False (every version is trained from the base checkpoint).
- freeze — on the held-out labels: ≥ min_holdout (100) of them and (alone & right − alone & wrong) / n ≥ the served
  version's + min_gain (0.01); act_guard found a threshold; no honesty number worse by more than tolerance (0.02);
  the training within budget.max_train_s; in label mode also the paired held-out gain's Wilson lower bound > 0.
- promote — the candidate certified, ≥ min_novel (100) post-freeze samples on keys it never trained on, on those
  share_k ≥ min_share (0.05) and share_k ≥ share_s + margin (0.01), and the memorisation probe passes (error alone on
  novel − on seen ≤ gap, 0.10; measured once both sides have ≥ probe_min, 30, alone answers). Label mode (no
  paired samples): the pre-freeze gates decided, promote.
- discard — the probe failed; not certified after `stale` (1500) samples; certified and not promotable after
  `trial` (1500).
- alarm — "recertify"; a second alarm on the same served version within `probation` (300) decisions →
  "rollback:<the version served before it>".

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
STATES = ("served", "candidate", "retired", "discarded")


# ───────────────────────────────────────────────────────────── the budget
@dataclass
class Budget:
    """What trainings may cost. max_trainings: over the loop's life (None: no cap); per_day: trainings per 24 h;
    min_interval_s: between two trainings; max_train_s: one training (estimated before it starts; refused if above);
    cpu_threads: torch threads of the runner on a CPU; device: "auto" (cuda when present and ≥ free_gpu_mb free, else
    cpu), "cpu", "cuda" or "cuda:<i>"; max_examples: rows per training; max_versions_kept: adapter files kept on disk
    (served and retired ones are never deleted)."""
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
        estimate_s: the estimated seconds of this one (None: not estimated yet)."""
        started = sorted(started)
        if self.max_trainings is not None and len(started) >= self.max_trainings:
            return f"budget: {len(started)} training(s) of max_trainings {self.max_trainings}"
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
    """One row of the versions table. state: "served", "candidate", "retired" or "discarded"; n_train: the train
    labels it was trained on (version 0: 0 unless stated); samples: post-freeze samples it has read; certified and
    threshold: its promise now; frozen_at: the decision it was frozen at (None for version 0)."""
    version: int
    state: str
    n_train: int = 0
    samples: int = 0
    certified: bool = False
    threshold: float = math.inf
    fingerprint: str | None = None
    frozen_at: int | None = None


@dataclass(frozen=True)
class LoopState:
    """What a hook sees, rebuilt by the loop for every call (changing it changes nothing in the loop). n: decisions
    so far; served / candidate: version numbers (candidate None: none in trial); labels: label counts by split;
    versions: the versions table; holdout: the served version's [(alone, wrong)] on its latest held-out labels, oldest
    first; alpha: the promised level; alarms: [{"n", "who", "version"}] so far; served_history: the versions served,
    in order (a rollback appends); trainings: the wall-clock start times of the trainings so far; now: the wall clock;
    paused; mode: "stream" (a Dispatcher, the stream promise) or "label" (a System alone: the pre-freeze gates only);
    promises, detector: their state dumps, for a policy that reads them."""
    n: int = 0
    served: int = 0
    candidate: int | None = None
    labels: dict = field(default_factory=lambda: {s: 0 for s in gates.SPLITS})
    versions: tuple = (VersionInfo(0, "served"),)
    holdout: tuple = ()
    alpha: float = 0.05
    alarms: tuple = ()
    served_history: tuple = (0,)
    trainings: tuple = ()
    now: float = 0.0
    paused: bool = False
    mode: str = "stream"
    promises: dict = field(default_factory=dict)
    detector: dict | None = None

    def __post_init__(self):
        if self.mode not in ("stream", "label"):
            raise ValueError(f"mode is 'stream' or 'label', not {self.mode!r}")

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
    at the promised risk); honesty_before / honesty_after: the served version's and the candidate's
    `solvi.testing.honesty` numbers on the held-out labels; seconds: the training's wall time; n_train: rows trained."""
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


@dataclass(frozen=True)
class Paired:
    """The candidate against the served version on the candidate's post-freeze samples (stream mode). Novel: samples
    whose key the candidate never trained on — n_novel of them, alone_k / wrong_k the candidate's answers alone / alone
    and wrong there, alone_s the served version's answers alone on the same samples at its threshold now. Seen: the
    samples whose key it trained on — n_seen, alone_seen, wrong_seen — read only by the memorisation probe."""
    n_novel: int
    alone_k: int
    wrong_k: int
    alone_s: int
    n_seen: int = 0
    alone_seen: int = 0
    wrong_seen: int = 0

    @property
    def share_k(self) -> float:
        return self.alone_k / self.n_novel if self.n_novel else 0.0

    @property
    def share_s(self) -> float:
        return self.alone_s / self.n_novel if self.n_novel else 0.0

    @property
    def err_k_novel(self) -> float | None:
        return self.wrong_k / self.alone_k if self.alone_k else None

    @property
    def err_k_seen(self) -> float | None:
        return self.wrong_seen / self.alone_seen if self.alone_seen else None


# ───────────────────────────────────────────────────────────── the abstract policy
class LoopPolicy(abc.ABC):
    """See the module docstring. Subclass and give every hook; `budget` (a Budget) is read by the loop too."""

    def __init__(self, budget: Budget | None = None):
        if budget is not None and not isinstance(budget, Budget):
            raise TypeError(f"budget is a Budget, not {type(budget).__name__}")
        self.budget = budget if budget is not None else Budget()

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
        """Freeze the trained model as the candidate (the pre-freeze gates on labels it never trained on)."""

    # --- promotion / rollback
    @abc.abstractmethod
    def promote(self, st: LoopState, paired: Paired | None) -> bool:
        """Serve the candidate now (paired None in label mode)."""

    @abc.abstractmethod
    def discard(self, st: LoopState, paired: Paired | None) -> str | None:
        """Why the candidate is discarded now, or None to keep trialling."""

    @abc.abstractmethod
    def alarm(self, st: LoopState, who: str, report: dict) -> str:
        """An alarm on the served version (who: "detector" or "monitor") → "recertify", "rollback:<v>" or "ignore"."""


# ───────────────────────────────────────────────────────────── the out-of-the-box policy
class DefaultPolicy(LoopPolicy):
    """See the module docstring for each hook. budget: a Budget (None: the defaults); s2_labels: take slow-path labels;
    calib / holdout: the split's shares; first, growth: the train labels before the first / a next training (absolute,
    × the served version's n_train); window, target_share: the improvement signal; min_gain, min_holdout, tolerance:
    the pre-freeze gates; wilson: the confidence of label mode's paired gain bound; min_novel, min_share, margin: the
    promotion rule; gap, probe_min: the memorisation probe; stale, trial: when a candidate is discarded; probation:
    two alarms closer than this roll back; r, lr, min_updates, max_updates: the training's settings."""

    def __init__(self, budget: Budget | None = None, *, s2_labels: bool = False, calib: float = 0.15,
                 holdout: float = 0.15, first: int = 300, growth: float = 1.5, window: int = 200, target_share=None,
                 min_gain: float = 0.01, min_holdout: int = 100, tolerance: float = 0.02, wilson: float = 0.95,
                 min_novel: int = 100, min_share: float = 0.05, margin: float = 0.01, gap: float = 0.10,
                 probe_min: int = 30, stale: int = 1500, trial: int = 1500, probation: int = 300, r: int = 8,
                 lr: float = 3e-4, min_updates: int = 200, max_updates: int = 3000):
        super().__init__(budget)
        gates.split("", calib=calib, holdout=holdout)                     # the shares, checked once
        for name, v in (("first", first), ("window", window), ("min_holdout", min_holdout), ("min_novel", min_novel),
                        ("probe_min", probe_min), ("stale", stale), ("trial", trial), ("probation", probation),
                        ("r", r), ("min_updates", min_updates), ("max_updates", max_updates)):
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if name == "probation" else 1):
                raise ValueError(f"{name} must be an integer ≥ {0 if name == 'probation' else 1}, not {v!r}")
        if min_updates > max_updates:
            raise ValueError(f"min_updates ({min_updates}) > max_updates ({max_updates})")
        if isinstance(growth, bool) or not isinstance(growth, (int, float)) or growth < 1:
            raise ValueError(f"growth must be a number ≥ 1, not {growth!r}")
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
        if st.candidate is not None:
            return f"version {st.candidate} is in trial"
        have, served_n = st.labels.get("train", 0), st.served_info.n_train
        need = max(self.first, math.ceil(self.growth * served_n))
        if have < need:
            return (f"{have} train label(s) < {need} (first {self.first}, growth {self.growth:g} × the served "
                    f"version's {served_n})")
        refused = self.budget.refuse(st.trainings, st.now)
        if refused:
            return refused
        sig = gates.incumbent_short(list(st.holdout)[-self.window:], st.alpha, self.target_share)
        if not sig["short"]:
            return f"no improvement signal: {sig['why']}"
        return None

    def want_train(self, st) -> bool:
        return self.why_not_train(st) is None

    def select(self, st, rows) -> list:
        rows = sorted(rows, key=lambda r: r.t)
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
        return all(g["ok"] for g in self.freeze_gates(st, report).values())

    # --- promotion / rollback
    def promote_checks(self, st, paired: Paired) -> dict:
        """Each promotion condition → {"certified", "novel", "min_share", "margin", "probe"}, each with "ok"."""
        c = st.candidate_info
        return {"certified": {"ok": bool(c is not None and c.certified)},
                "novel": {"ok": paired.n_novel >= self.min_novel, "n_novel": paired.n_novel, "min": self.min_novel},
                "min_share": {"ok": paired.share_k >= self.min_share, "share_k": paired.share_k},
                "margin": {"ok": paired.share_k >= paired.share_s + self.margin, "share_k": paired.share_k,
                           "share_s": paired.share_s},
                "probe": self._probe(paired)}

    def _probe(self, paired):
        return gates.probe(paired.alone_seen, paired.wrong_seen, paired.alone_k, paired.wrong_k, gap=self.gap,
                           min_n=self.probe_min)

    def promote(self, st, paired) -> bool:
        if st.candidate is None:
            return False
        if paired is None:
            return st.mode == "label"                  # label mode: the pre-freeze gates decided
        return all(g["ok"] for g in self.promote_checks(st, paired).values())

    def discard(self, st, paired) -> str | None:
        c = st.candidate_info
        if c is None or paired is None:
            return None
        pr = self._probe(paired)
        if not pr["ok"]:
            return f"memorisation probe: {pr['why']}"
        if not c.certified and c.samples >= self.stale:
            return f"not certified after {c.samples} samples"
        if c.certified and c.samples >= self.trial and not self.promote(st, paired):
            return (f"certified, not better after {c.samples} samples (share_k {paired.share_k:.4g}, share_s "
                    f"{paired.share_s:.4g}, {paired.n_novel} novel)")
        return None

    def alarm(self, st, who, report) -> str:
        last = max((a["n"] for a in st.alarms if a.get("version") == st.served), default=None)
        hist = list(st.served_history)
        if last is not None and st.n - last <= self.probation and len(hist) >= 2 and hist[-1] == st.served:
            return f"rollback:{hist[-2]}"
        return "recertify"


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
           "TrainReport", "VersionInfo"]
