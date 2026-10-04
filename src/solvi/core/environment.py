"""The Environment protocol: a world an agent acts in, one step at a time — what the environment agent of solvi 1.0
(`solvi.Agent`, the high level) runs on, and what an action model learns from.

    from solvi.core import Environment, Outcome

    class Corridor:                              # the smallest environment: walk right to the door
        def reset(self, seed=None):
            self.pos = 0
            return {"pos": 0}
        def actions(self, state):
            return ["left", "right"]
        def step(self, action):
            self.pos = max(0, self.pos + (1 if action == "right" else -1))
            return Outcome({"pos": self.pos}, accepted=True, effect={"pos": self.pos}, done=self.pos == 3)

`solvi.testing.conformance.check_environment` checks one (the same seed and the same actions give the same
outcomes; every step returns an Outcome; a refused action leaves the state as it was)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Outcome:
    """What one step did. state: the state after it (what `actions` and the next decision read); accepted: did the
    environment take the action (False: refused — a wall, a rule, a tool that said no — and the state did not move);
    effect: what changed, as data the environment itself reports (an action model learns from it and is checked
    against it); done: the episode is over; info: anything else (a score, a message), not read by solvi."""
    state: Any
    accepted: bool = True
    effect: Any = None
    done: bool = False
    info: dict = field(default_factory=dict)


@runtime_checkable
class Environment(Protocol):
    """A world acted in step by step.

    You implement: `reset(seed=None)` → the first state (the same seed gives the same episode); `actions(state)` →
    the actions possible in that state (a list; each one a plain value: a name, a tuple, a dict); `step(action)` →
    an `Outcome(state, accepted, effect, done)`.

    You get for free (with `solvi.Agent`, the environment agent of the 1.0 high level): System 1 = the action model's
    prediction plus the open goals of the agenda, hard checks = the agenda's gates and the action model's refusals
    (no action past a gate, no action predicted to be refused), System 2 (a search) when System 1 is unsure, a
    recorded and replayable step per decision, knowledge carried across episodes.

    Stability: stable (`solvi.Agent` calls `reset(seed)`, `actions(state)` and `step(action)` only)."""

    def reset(self, seed: int | None = None) -> Any: ...

    def actions(self, state: Any) -> list: ...

    def step(self, action: Any) -> Outcome: ...


__all__ = ["Environment", "Outcome"]
