"""The agenda: goals with done checks written in code, gates that block actions or goals, and the order between goals —
kept in the knowledge store's journal.

    from solvi.core.knowledge import Agenda, KnowledgeStore
    ag = Agenda(KnowledgeStore("agent.jsonl"))
    ag.goal("authenticated", done=lambda s: s.get("user_id") is not None)
    ag.goal("order_changed", done=lambda s: s.get("order_status") == "exchange requested", requires=["authenticated"])
    ag.gate("auth_first", lambda s: s.get("user_id") is not None, blocks=["modify_*", "exchange_*", "cancel_*"])

    ag.update(state)                    # runs the done checks; every change of a goal's state is journaled
    ag.open(state), ag.blocked(state), ag.done()
    ok, why = ag.allows(state, "exchange_delivered_order_items")    # no action past a gate
    ag.dry_run(recorded_successes)      # how often each gate would have blocked actions that worked

A goal is done only when its code check says so over the state, never because a model claims it. A gate is a hard
check: `allows` refuses an action it blocks while its check is False, and a goal it blocks is not open; a person may
override a gate with a recorded reason. `requires` orders goals (a tech tree): a goal is open when every goal it
requires is done and no gate blocks it. Goals and gates are rule items of the knowledge store (source "spec"), the
goals' states are fact items (source "outcome": what the done check observed), so a decision can say which goal it
served and which gates held, and the journal shows the progress.

Validate gates before making them hard. A gate written from policy text can block correct calls: on a support-agent
stream, a gate taken from the policy's "remind the customer to confirm they listed all items" blocked mostly calls the
store would accept and the customer wanted, and some right calls were never made. `dry_run(records)` replays recorded
successful (state, action) pairs through the gates and reports, per gate, how often it would have blocked them — run it
on your logs before you install a gate. Where the gates are an environment's written rules, "no action past a gate"
and "no wrong block" hold by construction (benchmarks/knowledge/toy_crafting.py: the agenda arms fail no attempt)."""
from __future__ import annotations

import fnmatch

from ..provenance import code_fingerprint, digest

GOAL_STATES = ("done", "open", "blocked")


class Agenda:
    """See the module docstring. store: a KnowledgeStore the goals, gates and state changes are written to (None: an
    in-memory list, `history`); scope: the scope of those items (default {"agenda": "agenda"}); sticky: a done goal
    stays done (achievements) — False re-checks it on every update (an order whose state can change back)."""

    def __init__(self, store=None, *, scope=None, sticky=True):
        self.store = store
        self.scope = dict(scope or {"agenda": "agenda"})
        self.sticky = sticky
        self.goals = {}            # name → {"done", "requires", "gates", "item"}
        self.gates = {}            # name → {"check", "blocks", "item"}
        self.overrides = {}        # gate → {"by", "why"}
        self.states = {}           # goal → last journaled state
        self._done = set()
        self.history = []          # [(goal, old, new)]

    # --- declaring
    def goal(self, name, *, done, requires=(), gates=()):
        """A goal: done(state) → bool is its check in code; requires: goals done first; gates: gates that must pass
        for it to be open (a gate may also name it in `blocks`)."""
        if not callable(done):
            raise TypeError(f"goal {name!r}: done is a function of the state (code), not {type(done).__name__}")
        unknown = [g for g in requires if g not in self.goals and g != name]
        if unknown:
            raise ValueError(f"goal {name!r} requires goals not declared yet: {unknown} (declare them first)")
        self.goals[name] = {"done": done, "requires": tuple(requires), "gates": tuple(gates), "item": None}
        if self.store is not None:
            self.goals[name]["item"] = self.store.add(
                "rule", {"s": f"goal:{name}", "r": "goal",
                         "o": {"requires": list(requires), "gates": list(gates), "done": code_fingerprint(done)}},
                self.scope, source="spec", by="agenda")
        return self

    def gate(self, name, check, *, blocks):
        """A gate: check(state) → True when actions / goals it blocks may go ahead (False or a Fail: blocked, with
        the reason). blocks: action names (glob patterns like "modify_*"), goal names, or a function(action) → bool."""
        if not callable(check):
            raise TypeError(f"gate {name!r}: check is a function of the state")
        if isinstance(blocks, str):
            blocks = [blocks]
        self.gates[name] = {"check": check, "blocks": blocks, "item": None}
        if self.store is not None:
            shown = [b for b in blocks] if not callable(blocks) else [f"<function {getattr(blocks, '__name__', '?')}>"]
            self.gates[name]["item"] = self.store.add(
                "rule", {"s": f"gate:{name}", "r": "gate", "o": {"blocks": shown, "check": code_fingerprint(check)}},
                self.scope, source="spec", by="agenda")
        return self

    def override(self, gate, *, by, why):
        """A person lets a gate's blocks through (until clear_override), with a recorded reason."""
        if gate not in self.gates:
            raise KeyError(f"no gate {gate!r}")
        self.overrides[gate] = {"by": str(by), "why": str(why)}
        if self.store is not None:
            self.store.add("fact", {"s": f"gate:{gate}", "r": "override", "o": str(why)}, self.scope, source="person",
                           by=by)

    def clear_override(self, gate):
        self.overrides.pop(gate, None)
        if self.store is not None:
            self.store.note(agenda="override cleared", gate=gate)

    # --- reading the state
    def _passes(self, gate, state):
        if gate in self.overrides:
            return True, f"overridden by {self.overrides[gate]['by']}: {self.overrides[gate]['why']}"
        v = self.gates[gate]["check"](state)
        reasons = (getattr(v, "extra", None) or {}).get("reasons") if v is not None else None
        return bool(v), "; ".join(reasons) if reasons else ""

    def _blocks(self, gate, target):
        b = self.gates[gate]["blocks"]
        if callable(b):
            return bool(b(target))
        return any(target == x or fnmatch.fnmatchcase(str(target), str(x)) for x in b)

    def gates_on(self, target):
        """The gates that apply to an action or a goal."""
        out = [g for g in self.gates if self._blocks(g, target)]
        if target in self.goals:
            out += [g for g in self.goals[target]["gates"] if g not in out]
        return out

    def allows(self, state, action):
        """May `action` be taken in `state`? → (True, []) or (False, ["gate: reason", ...]) — every gate that blocks it
        and whose check is False."""
        why = []
        for g in self.gates_on(action):
            ok, reason = self._passes(g, state)
            if not ok:
                why.append(f"{g}: {reason}" if reason else g)
        return not why, why

    def _state_of(self, name, state):
        if name in self._done:
            return "done", []
        why = [f"requires {r}" for r in self.goals[name]["requires"] if r not in self._done]
        ok, gw = self.allows(state, name) if state is not None else (True, [])
        return ("blocked" if why or not ok else "open"), why + gw

    def update(self, state):
        """Run every done check on `state`; journal each goal whose state changed. → {goal: (old, new)}."""
        for name, g in self.goals.items():
            if name in self._done and self.sticky:
                continue
            if bool(g["done"](state)):
                self._done.add(name)
            elif not self.sticky:
                self._done.discard(name)
        changes = {}
        for name in self.goals:
            new, _ = self._state_of(name, state)
            old = self.states.get(name)
            if new != old:
                changes[name] = (old, new)
                self.states[name] = new
                self.history.append((name, old, new))
                if self.store is not None:
                    item = self.goals[name]["item"]
                    self.store.add("fact", {"s": f"goal:{name}", "r": "state", "o": new}, self.scope, source="outcome",
                                   by="done check", derived_from=[item] if item else ())
        return changes

    def done(self):
        """The goals whose done check said so (as of the last update)."""
        return set(self._done)

    def open(self, state=None):
        """Goals that can be worked on now: not done, every required goal done, no gate blocking them (on `state`)."""
        return [n for n in self.goals if self._state_of(n, state)[0] == "open"]

    def blocked(self, state=None):
        """{goal: [reasons]} for the goals that are not done and cannot be worked on now."""
        out = {}
        for n in self.goals:
            s, why = self._state_of(n, state)
            if s == "blocked":
                out[n] = why
        return out

    def snapshot(self, state=None):
        """The agenda as a plain fact for a decision: done, open, blocked (with reasons)."""
        return {"done": sorted(self._done), "open": self.open(state), "blocked": self.blocked(state),
                "overrides": dict(sorted(self.overrides.items())), "fp": self.fingerprint()}

    def fingerprint(self):
        return digest("Agenda", {n: [code_fingerprint(g["done"]), list(g["requires"]), list(g["gates"])]
                                 for n, g in sorted(self.goals.items())},
                      {n: [code_fingerprint(g["check"]), g["blocks"] if not callable(g["blocks"]) else
                           code_fingerprint(g["blocks"])] for n, g in sorted(self.gates.items())})

    # --- validating gates
    def dry_run(self, records, *, examples=3):
        """Replay recorded successful actions — an iterable of (state, action) that worked — through the gates, without
        changing anything. → {"records": n, "blocked_any": k, "gates": {gate: {"applies", "would_block", "share",
        "examples": [record index, ...]}}}: a gate that would have blocked many actions that worked is not ready to be
        hard (see the module docstring)."""
        rep = {g: {"applies": 0, "would_block": 0, "share": 0.0, "examples": []} for g in self.gates}
        n = blocked_any = 0
        for idx, (state, action) in enumerate(records):
            n += 1
            hit = False
            for g in self.gates_on(action):
                r = rep[g]
                r["applies"] += 1
                if not self._passes(g, state)[0]:
                    r["would_block"] += 1
                    hit = True
                    if len(r["examples"]) < examples:
                        r["examples"].append(idx)
            blocked_any += hit
        for r in rep.values():
            r["share"] = r["would_block"] / r["applies"] if r["applies"] else 0.0
        return {"records": n, "blocked_any": blocked_any, "gates": rep}


__all__ = ["Agenda", "GOAL_STATES"]
