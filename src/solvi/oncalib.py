"""EXPERIMENTAL — calibration on the fly from the outcomes the agent sees. Importing it warns (ExperimentalWarning); its
API may change or it may be removed, and nothing stable in solvi imports it.

    from solvi.oncalib import OnTheFly                  # ExperimentalWarning
    live = OnTheFly(system, "move", max_risk=0.05, every=50, window=500)
    res = system.ask(state)
    ...                                                 # later, the environment shows what the decision led to
    live.outcome(res, "west", note="Link lost a heart going east")   # stored as an "outcome" label (System.outcome);
                                                                     # every 50 labels: recalibrated on the last 500
    live.drifted()                                      # a drift flag: recalibrate now, on the labels since the flag

What it does: every outcome is stored as a label of its decision through `System.outcome` (source "outcome", the
same source check as every label). After every `every` new labels of the question — and when `drifted()` is called —
the question's guarantee is set again (`System.guarantee`) on the last `window` outcome labels (after a drift flag, only
those recorded since it), as long as there are at least `min_labels`. Each recalibration is kept in `history` with its
report and the ids of the labels it used.

RISK — why this is not in the stable path. A guarantee's promise ("P(answered alone and wrong) ≤ 5%") holds for
decisions drawn like its calibration examples, calibrated once. solvi's own measurements found the promise broken when
a guarantee was recalibrated continuously, which is why the stable path does not do it. Why:
  - the outcomes an agent sees are not a random sample of its decisions (it sees the consequences of the moves it made,
    more often of the bad ones; a decision that abstained has no outcome at all), so the calibration set is biased;
  - the threshold is moved again and again on overlapping windows, each time to the edge of the promise on the latest
    labels: the promise of one calibration does not hold for a threshold chosen by many — the error drifts above the
    stated level without any single step looking wrong;
  - a window short enough to follow a change is too short for a strict method (the guarantee abstains on everything,
    or weak="raise" refuses to calibrate).
Use it to explore, never to make a promise you report. For a promise: calibrate explicitly, on labels drawn at random
(or all the outcomes of a period), on a schedule or after a drift flag — `System.guarantee(question, examples,
corrections=True)` — and say on what it was calibrated."""
from __future__ import annotations

import warnings

from .core import ExperimentalWarning

warnings.warn("solvi.oncalib is experimental: calibration on the fly from outcomes does not keep a guarantee's promise "
              "(see its module docs); its API may change", ExperimentalWarning, stacklevel=2)


class OnTheFly:
    """Recalibrate a question's guarantee from the outcomes the agent sees (EXPERIMENTAL; see the module docs and its
    risk note).

    system: a System with storage; question: the question whose guarantee is recalibrated. every: recalibrate after this
    many new outcome labels; window: on the last this many (None: all); min_labels: fewer → not recalibrated (the labels
    are still stored). The other keyword arguments go to System.guarantee (max_risk=, max_error=, method=, delta=,
    signal=, ...)."""

    def __init__(self, system, question, *, every=50, window=500, min_labels=30, **guarantee):
        if system.storage is None:
            raise ValueError("OnTheFly keeps the outcomes as labels in the system's storage: System(storage=...)")
        if question not in system.questions:
            raise KeyError(f"no question {question!r} in the system")
        if int(every) < 1 or (window is not None and int(window) < 1) or int(min_labels) < 1:
            raise ValueError("every, window and min_labels are positive")
        if "examples" in guarantee or "corrections" in guarantee:
            raise ValueError("OnTheFly calibrates on the outcome labels it reads from the storage: no examples= / "
                             "corrections=")
        self.system, self.question = system, question
        self.every, self.window, self.min_labels = int(every), None if window is None else int(window), int(min_labels)
        self.guarantee = dict(guarantee)
        self.new = 0                      # outcome labels since the last recalibration
        self.since = None                 # after drifted(): only labels stored after this time
        self.history = []                 # [{"time", "why", "labels": [ids], "report"}]

    def outcome(self, decision, value, *, note=None, by=None):
        """Store the outcome as a label (System.outcome) → its stored id; recalibrate when `every` new labels came."""
        cid = self.system.outcome(decision, value, question=self.question, note=note, by=by)
        self.new += 1
        if self.new >= self.every:
            self.recalibrate(f"{self.new} new outcome label(s)")
        return cid

    def drifted(self, why="drift flag"):
        """A drift flag (a DriftMonitor, an open-set gate): from now on only the labels stored after it count, and the
        guarantee is recalibrated at once when there are enough of them."""
        self.since = self.system.storage.clock()
        return self.recalibrate(why)

    def labels(self):
        """The outcome labels of the question the next recalibration reads → [correction dicts], oldest first."""
        got = [c for c in self.system.storage.corrections()
               if c["question"] == self.question and c["source"] == "outcome"
               and (self.since is None or c["time"] >= self.since)]
        return got if self.window is None else got[-self.window:]

    def recalibrate(self, why="asked"):
        """Set the question's guarantee again on labels() → its calibration report, or None when there are fewer than
        min_labels (the guarantee in force stays)."""
        labs = self.labels()
        if len(labs) < self.min_labels:
            return None
        rep = self.system.guarantee(self.question, [(c["init"], c["answer"]) for c in labs], **self.guarantee)
        self.new = 0
        self.history.append({"time": self.system.storage.clock(), "why": why, "labels": [c["id"] for c in labs],
                             "report": rep})
        return rep


__all__ = ["OnTheFly"]
