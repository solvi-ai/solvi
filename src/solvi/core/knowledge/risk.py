"""Knowledge as protection (the default) and justified risk (an option): what to do with a Prediction.

    from solvi.core.knowledge import Protect, RiskBudget
    policy = Protect()                                   # the verdict is final: refuse → avoid, unknown → System 2
    policy = RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0, min_support=3)
    d = policy.decide("descend", prediction, gain=0.3)  # → RiskDecision(choice="take" | "avoid" | "ask_s2", ...)
    policy.new_episode()                                 # the per-episode budget starts again

The RiskPolicy protocol: decide(action, prediction, gain=0.0, *, key=None) → RiskDecision; new_episode(). Every
decision records its rationale: the risk estimate, its support, the expected gain, the budget left.

Protect: "accept" → take, "refuse" → avoid, "unknown" → ask System 2. Knowledge used only as hard gates.
RiskBudget: a refused action is taken when its expected gain is at least `min_gain_ratio` × its estimated risk and the
risk fits what is left of the episode's budget; a refusal resting on fewer than `min_support` cases (and not on a written
spec's rate, support -1) is ambiguous → System 2; a hard prediction (an instant-harm or policy rule) is never taken in
any mode. A `key` is charged once per episode (the same fight, step after step).

Why both exist: knowledge used only as gates removes the failures it targets and can lower a metric that rewards risk
(in a dungeon game: fewer deaths of the targeted kinds, less depth reached, heroes starving instead). Justified risk
won back part of it there, not all. So protection stays the default; RiskBudget is an option to measure on your own
metric. benchmarks/knowledge/risk_dungeon.py shows the mechanism on a toy dungeon.

LearnedGate: a gate learned from failures is bounded. A depth gate learned as "the median death depth at this level
− 1" tightens itself: deaths happen where the gate lets the hero go, so each death lowers it further
(risk_dungeon.py, arm protect_unbounded: 6.70 → 2.99 levels from the first to the last third of the streams). Here a
learned limit reads only the last `expiry` episodes,
never falls below `floor`, and is reopened by evidence: a level whose recorded arrivals show a low failure rate is
predicted with that rate and support, which a RiskBudget can take; Protect still refuses it. A bounded gate does not
make the decline go away by itself: with it, Protect still fell 6.74 → 5.02 there, as learned refusals accumulated."""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .actions import Prediction

CHOICES = ("take", "avoid", "ask_s2")


@dataclass
class RiskDecision:
    """What a risk policy decided about one action, and why: the choice, the reason, the prediction's risk and support,
    the expected gain, and the episode's budget left after it."""
    choice: str
    reason: str
    risk: float = 0.0
    support: int = 0
    gain: float = 0.0
    budget_left: float | None = None
    action: str | None = None

    def to_dict(self):
        return {"choice": self.choice, "reason": self.reason, "risk": round(float(self.risk), 6),
                "support": int(self.support), "gain": self.gain, "budget_left": self.budget_left,
                "action": self.action}


@runtime_checkable
class RiskPolicy(Protocol):
    """What to do with a prediction.

    You implement: `decide(action, prediction, gain=0.0, *, key=None) → RiskDecision` (take / avoid / ask_s2 with its
    reason, the risk, support, gain and budget left) and `new_episode()`.

    You get for free: every decision on a refused or unknown action recorded with its rationale; a hard prediction is
    yours to never take (Protect and RiskBudget never do).

    Stability: stable."""

    def decide(self, action, prediction, gain=0.0, *, key=None) -> RiskDecision: ...

    def new_episode(self) -> None: ...


class Protect:
    """The verdict is final (the default): accept → take, refuse → avoid, unknown → ask System 2."""

    def __init__(self):
        self.log = []

    def decide(self, action, prediction, gain=0.0, *, key=None):
        p = prediction
        if p.verdict == "accept":
            d = RiskDecision("take", "predicted accept", p.risk, p.support, gain, None, action)
        elif p.verdict == "refuse":
            d = RiskDecision("avoid", f"predicted refuse: {p.reason}" + (" (hard)" if p.hard else ""), p.risk, p.support,
                             gain, None, action)
        else:
            d = RiskDecision("ask_s2", f"unknown: {p.reason}", p.risk, p.support, gain, None, action)
        self.log.append(d)
        return d

    def new_episode(self):
        pass

    def fingerprint(self):
        return "Protect"


class RiskBudget:
    """Justified risk within a per-episode budget (see the module docstring). max_risk_per_episode: the sum of the risks
    taken in one episode (in the predictions' unit, e.g. death-equivalents); min_gain_ratio: take only when gain ≥ ratio ×
    risk; min_support: fewer cases behind a refusal → ask System 2."""

    def __init__(self, max_risk_per_episode=1.0, min_gain_ratio=1.0, min_support=3):
        if max_risk_per_episode < 0 or min_gain_ratio < 0 or min_support < 0:
            raise ValueError("max_risk_per_episode, min_gain_ratio and min_support are ≥ 0")
        self.max_risk_per_episode = float(max_risk_per_episode)
        self.min_gain_ratio = float(min_gain_ratio)
        self.min_support = int(min_support)
        self.left = self.max_risk_per_episode
        self.taken = set()
        self.log = []

    def new_episode(self):
        self.left = self.max_risk_per_episode
        self.taken = set()

    def decide(self, action, prediction, gain=0.0, *, key=None):
        p = prediction
        gain = float(gain)

        def out(choice, why):
            d = RiskDecision(choice, why, p.risk, p.support, gain, round(self.left, 6), action)
            self.log.append(d)
            return d

        if p.verdict == "accept":
            return out("take", "predicted accept")
        if p.verdict == "unknown":
            return out("ask_s2", f"unknown: {p.reason}")
        if p.hard:
            return out("avoid", f"hard rule, never traded: {p.reason}")
        if key is not None and key in self.taken:
            return out("take", "already charged this episode")
        if 0 <= p.support < self.min_support:
            return out("ask_s2", f"ambiguous: {p.support} case(s) < {self.min_support}")
        if gain < self.min_gain_ratio * p.risk:
            return out("avoid", f"gain {gain:g} < {self.min_gain_ratio:g} × risk {p.risk:.4g}")
        if p.risk > self.left:
            return out("avoid", f"risk {p.risk:.4g} > budget left {self.left:.4g}")
        self.left -= p.risk
        if key is not None:
            self.taken.add(key)
        return out("take", f"gain {gain:g} ≥ {self.min_gain_ratio:g} × risk {p.risk:.4g}; within budget")

    def fingerprint(self):
        return f"RiskBudget({self.max_risk_per_episode}, {self.min_gain_ratio}, {self.min_support})"


class LearnedGate:
    """A limit on a level (a depth, an amount, a distance) per context (a number: an experience level, a stage), learned
    from failures and bounded.

        gate = LearnedGate("depth", expiry=10, floor=lambda xl: xl + 1, near=1, min_support=3, margin=1)
        gate.failed(context=xl, level=depth)          # a failure (a death) at this level
        gate.arrived(context=xl, level=depth, failed=False)   # evidence: reached this level; failed there or not
        gate.end_episode()
        gate.limit(xl)                                 # max(floor(xl), median failure level nearby − margin) or None
        gate.predict(xl, depth)                        # Prediction: accept below the limit, else refuse with the rate

    expiry: only failures of the last `expiry` episodes count (a gate cannot be tightened by old deaths forever); floor:
    the limit never falls below floor(context) (a number or a function of the context); near: failures at contexts
    within ±near count; min_support: failures needed to set a limit at all, and arrivals needed before their rate
    replaces the prior; prior: the risk predicted past the limit without enough arrivals."""

    def __init__(self, name, *, expiry=10, floor=0, near=1, min_support=3, margin=1, prior=0.5):
        if expiry < 1:
            raise ValueError("expiry is at least one episode: a learned gate without an expiry tightens without bound")
        self.name, self.expiry, self.floor, self.near = name, int(expiry), floor, near
        self.min_support, self.margin, self.prior = int(min_support), margin, float(prior)
        self.episodes = deque(maxlen=self.expiry)          # per episode: [(context, level)] failures
        self.arrivals = deque(maxlen=3 * self.expiry)      # per episode: [(context, level, failed)]
        self._fails, self._arr = [], []

    def failed(self, context, level):
        self._fails.append((context, level))

    def arrived(self, context, level, failed=False):
        self._arr.append((context, level, bool(failed)))

    def end_episode(self):
        self.episodes.append(self._fails)
        self.arrivals.append(self._arr)
        self._fails, self._arr = [], []

    def _floor(self, context):
        return self.floor(context) if callable(self.floor) else self.floor

    def limit(self, context):
        """The highest allowed level at this context, or None (no limit learned: not enough recent failures)."""
        lv = [lev for ep in self.episodes for c, lev in ep if abs(c - context) <= self.near]
        if len(lv) < self.min_support:
            return None
        return max(self._floor(context), statistics.median(lv) - self.margin)

    def evidence(self, context, level):
        """(failure rate (Laplace), number of recorded arrivals) at this level and a context nearby."""
        n = k = 0
        for ep in self.arrivals:
            for c, lev, f in ep:
                if lev == level and abs(c - context) <= self.near:
                    n += 1
                    k += f
        return (k + 1) / (n + 2), n

    def predict(self, context, level):
        """Prediction for going to `level` at `context`: accept within the limit; past it refuse, with the measured
        arrival failure rate and its support when there is enough evidence (else the prior, support = arrivals)."""
        lim = self.limit(context)
        if lim is None or level <= lim:
            return Prediction("accept", 0.0, 0, f"{self.name}: within the learned limit ({lim})", action=self.name)
        rate, n = self.evidence(context, level)
        risk = rate if n >= self.min_support else self.prior
        return Prediction("refuse", risk, n, f"{self.name}: {level} past the learned limit {lim:g} (floor "
                          f"{self._floor(context):g}, last {len(self.episodes)} episode(s))", action=self.name)


__all__ = ["CHOICES", "LearnedGate", "Protect", "RiskBudget", "RiskDecision", "RiskPolicy"]
