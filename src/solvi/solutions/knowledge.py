"""Knowledge: what a system learned, from whom, and what rests on it — one object a decision system, a guard and an
environment agent share.

    import solvi

    km = solvi.Knowledge("knowledge.jsonl", vocabulary={"here": lambda s, a: s["here"]})
    km.tell({"s": "c17", "r": "tier", "o": "vip"}, source="person", by="crm")      # a fact, with its source
    km.goal("wood", done=lambda s: s["wood"] > 0)                                 # an agenda goal: done is code
    km.agenda.gate("tool_first", lambda s: s["pickaxe"], blocks=["collect_stone"])  # a gate: a hard check
    km.observe(state, "collect_wood", outcome)        # an environment step: action model, map facts, agenda, skills
    km.retract(item_id, why="wrong label")            # with everything derived from it
    km.report()

A facade over the low level (solvi.core.knowledge): `store` is the KnowledgeStore (the hash-chained journal: facts,
rules, skills, actions, episodes, with provenance, disputes to a person and the exact retraction cascade), `agenda` the
Agenda (goals with done checks in code, gates, order — in the same journal), `actions` the action model
(ConservativeActionModel over `vocabulary`, or one of your own with `actions=`, or None), `failures` the failure memory
(FailureMemory, with `failures=True` or its settings as a dict; None by default).

Who reads it: `solvi.build(..., knowledge=km)` gives every decision `km.snapshot()` as the fact "knowledge" (a rule
reads it with `Knowledge.value(knowledge, s, r)`); `solvi.Guard(..., knowledge=km)` turns the action model's refusals
and the agenda's gates into hard checks on tool calls; `solvi.Agent(env, knowledge=km)` acts on it in an environment
and writes what it observes back. Knowledge never comes from the system's own unverified answers: the store's source
check refuses them (sources "person", "outcome", "spec", or "verified" with the stored decision it comes from).

Map facts. `observe(..., at=, to=, map=)` writes what an environment showed as facts scoped to one map: "leads_to
<action>" (from the key `at`, the action led to the key `to`) and "accepts <action>" (taken or refused there). They are
carried to the next episode on the same map as they are — and dropped by the first contradiction: an arrival that
contradicts a fact learned in an earlier episode raises a flag on the map's scope (KnowledgeStore.flag), so every
older map fact is a hint, not a fact, until an observation confirms it again. Rules (the action model, the skills) are
carried across maps. A skill is a confirmed route of a goal: the action after which the goal's done check turned true,
written as a "skill" item (source "outcome")."""
from __future__ import annotations

import json

DEFAULT_FAILURES = {"window": 10, "unit": "steps", "max_blocked": 8, "min_open": 1}


def action_key(action):
    """An environment's action as the knowledge names it: a string as it is, anything else as its JSON form."""
    return action if isinstance(action, str) else json.dumps(action, sort_keys=True, ensure_ascii=False, default=str)


def split_action(action, args=None):
    """An action → (name, args) for an action model and the agenda's gates: "name" → (name, {}); ("name", x, ...) →
    (name, {"args": [x, ...]}); {"name": n, ...} → (n, the other keys); anything else → (its JSON form, {})."""
    if args is not None:
        return (action if isinstance(action, str) else action_key(action)), dict(args)
    if isinstance(action, str):
        return action, {}
    if isinstance(action, (tuple, list)) and action and isinstance(action[0], str):
        return action[0], ({"args": list(action[1:])} if len(action) > 1 else {})
    if isinstance(action, dict) and isinstance(action.get("name"), str):
        return action["name"], {k: v for k, v in action.items() if k != "name"}
    return action_key(action), {}


class Knowledge:
    """See the module docstring.

    path: where the journal is kept — None (in memory), a path (.jsonl, .db, ...) or a TraceStorage (the decisions'
    store, so knowledge and decisions share one hash chain); an existing journal is replayed (the action model too, when
    it is built from a vocabulary). vocabulary: a Vocabulary or {name: predicate(state, args)} — the conditions the
    action model may learn from (None: no action model unless `actions=` gives one). write_gate: a WriteGate (or a list)
    run after the built-in source and consistency gates. actions: an ActionModel of your own (instead of vocabulary=).
    failures: True or FailureMemory settings ({"window", "unit", "max_blocked", "min_open"}) — a failure memory whose
    blocks are hard checks of the agent (None: off). severity: the action model's harm of a refused action (a number or
    {action: number})."""

    def __init__(self, path=None, *, vocabulary=None, write_gate=None, actions=None, failures=None, severity=1.0):
        from ..core.knowledge import Agenda, ConservativeActionModel, FailureMemory, KnowledgeStore, Vocabulary
        from ..core.knowledge.gates import default_gates
        if vocabulary is not None and actions is not None:
            raise ValueError("give vocabulary= (the built-in ConservativeActionModel learns over it) or actions= (an "
                             "action model of your own), not both")
        gates = None
        if write_gate is not None:
            extra = list(write_gate) if isinstance(write_gate, (list, tuple)) else [write_gate]
            gates = default_gates() + extra
        self.store = KnowledgeStore(path, gates=gates)
        self.agenda = Agenda(self.store)
        if vocabulary is not None:
            if isinstance(vocabulary, dict):
                vocabulary = Vocabulary(vocabulary)
            self.actions = ConservativeActionModel.replay(self.store, vocabulary, severity=severity)
            self.actions.store = self.store
        else:
            if actions is not None and not all(callable(getattr(actions, m, None)) for m in ("observe", "predict",
                                                                                              "fingerprint")):
                raise TypeError("actions= takes an ActionModel: observe(state, action, args, accepted, effect), "
                                "predict(state, action, args) → Prediction, fingerprint()")
            self.actions = actions
        self.failures = None
        if failures:
            cfg = dict(DEFAULT_FAILURES, **(failures if isinstance(failures, dict) else {}))
            self.failures = FailureMemory(cfg.pop("window"), store=self.store, **cfg)
        self._edges, self._accepts, self._skills = {}, {}, {}      # indexes over the store's map facts and skills
        self._episode = {"at": 0, "flagged": set(), "n": 0}
        self._index()

    # ------------------------------------------------------------------------------------------------ writing
    def tell(self, fact, *, source, by=None, of=None, scope=None, derived_from=(), evidence=None, valid=None):
        """Write a fact — {"s", "r", "o"} or (s, r, o) — from a person, an outcome, a written spec, or a verified System 2
        answer (`of=` its stored decision). It goes through the source check and the write gates; a contradiction of
        equal rank becomes a person question (`store.disputes()`). → the item's id, or None when a gate refused it (the
        journal says why)."""
        if isinstance(fact, (tuple, list)) and len(fact) == 3:
            fact = {"s": fact[0], "r": fact[1], "o": fact[2]}
        if not (isinstance(fact, dict) and {"s", "r", "o"} <= set(fact)):
            raise ValueError("a fact is {'s': subject, 'r': relation, 'o': value} or (subject, relation, value)")
        return self.store.add("fact", fact, scope, source=source, by=by, of=of, derived_from=derived_from,
                              evidence=evidence, valid=valid)

    def retract(self, item, why=None, *, by=None, storage=None, decide=None):
        """Take an item back with everything derived from it (nothing is deleted: the journal keeps it). → {"status":
        {id: (before, after)}, "answer_changes", "justification_only"} — the last two (with `storage=`, the decisions'
        TraceStorage, and `decide=`, a System or a function init → answers) list the stored decisions that rested on
        it, re-run on the snapshot rebuilt without it: the answer changes (for a reviewer) or only the justification."""
        status = self.store.retract(item, by=by, why=why)
        out = {"status": status, "answer_changes": None, "justification_only": None}
        if storage is not None and decide is not None:
            from ..core.store import open_storage
            changes, same = self.store.redecide(open_storage(storage), set(status) | {item}, decide)
            out["answer_changes"], out["justification_only"] = changes, same
        return out

    def goal(self, name, done, requires=(), gates=()):
        """An agenda goal: done(state) → bool is its check in code (never a model's claim); requires: goals done first;
        gates: gates (declared with `km.agenda.gate(name, check, blocks=)`) that must pass for it to be open. → self."""
        self.agenda.goal(name, done=done, requires=requires, gates=gates)
        return self

    def observe(self, state, action, outcome, *, args=None, at=None, to=None, map=None):
        """One environment step: `action` taken in `state` gave `outcome` (an Outcome, or True / False: accepted or
        refused). Updates the action model (and journals its prediction against what happened), the failure memory,
        the agenda (done checks on the new state) and the skills (a goal done right after this action); with `at`
        (the key of `state`), `to` (the key reached) and `map` (the map's name), the map facts — the first
        contradiction of a fact carried from an earlier episode flags the map (see the module docstring).
        → {"prediction": the action model's prediction before the step (dict) or None, "surprise", "drift": the flag's
        id or None, "goals_done": goals whose done check turned true, "skills": skill item ids written}."""
        from ..core.environment import Outcome
        out = outcome if isinstance(outcome, Outcome) else Outcome(state, accepted=bool(outcome))
        name, a_args = split_action(action, args)
        akey = action_key(action)
        before = surprise = None
        if self.actions is not None:
            before = self.actions.observe(state, name, a_args, bool(out.accepted), out.effect)
            surprise = before is not None and ((before.verdict == "accept" and not out.accepted)
                                               or (before.verdict == "refuse" and out.accepted))
        if self.failures is not None:
            plan = [at, akey] if at is not None else akey
            if out.accepted:
                self.failures.succeeded(plan)
            else:
                self.failures.failed(plan, why="refused by the environment")
            self.failures.step()
        drift = None
        if at is not None:
            drift = self._map_facts(map, at, akey, bool(out.accepted), to if out.accepted else at)
        changes = self.agenda.update(out.state)
        done = sorted(g for g, (old, new) in changes.items() if new == "done" and old is not None)
        skills = []
        if out.accepted:
            for g in done:
                iid = self.store.add("skill", {"s": f"goal:{g}", "r": f"via {akey}", "o": True}, {"skills": "agent"},
                                     source="outcome", by="environment", evidence=[{"at": at, "map": map}])
                if iid is not None:
                    self._skills.setdefault(g, {})[akey] = iid
                    skills.append(iid)
        return {"prediction": before.to_dict() if before is not None else None, "surprise": bool(surprise),
                "drift": drift, "goals_done": done, "skills": skills}

    def _map_facts(self, map_id, at, akey, accepted, to):
        scope = {"map": str(map_id)}
        drift = None
        if accepted:
            old = self._edges.get(scope["map"], {}).get(at, {}).get(akey)
            if old is not None and old[1] != to and self.store.status(old[0]) == "active" \
                    and (self.store.state.get(old[0]) or {}).get("learned_at", 0) < self._episode["at"] \
                    and scope["map"] not in self._episode["flagged"]:
                drift = self.store.flag(scope=scope, by="environment",
                                        why=f"an arrival contradicted the carried map: {akey} from {at} led to {to}, "
                                            f"not {old[1]}")
                self._episode["flagged"].add(scope["map"])
            iid = self.store.add("fact", {"s": at, "r": f"leads_to {akey}", "o": to}, scope, source="outcome",
                                 by="environment")
            if iid is not None:
                self._edges.setdefault(scope["map"], {}).setdefault(at, {})[akey] = (iid, to)
        iid = self.store.add("fact", {"s": at, "r": f"accepts {akey}", "o": accepted}, scope, source="outcome",
                             by="environment")
        if iid is not None:
            self._accepts.setdefault(scope["map"], {}).setdefault(at, {})[akey] = (iid, accepted)
        return drift

    def _index(self):
        """The map facts and skills of a replayed journal, indexed (latest assertion wins)."""
        order = sorted(self.store.items.values(), key=lambda it: (it.asserts[-1]["t"] if it.asserts else -1, it.id))
        for it in order:
            b, sc = it.body if isinstance(it.body, dict) else {}, it.scope or {}
            r = str(b.get("r", ""))
            if it.kind == "fact" and "map" in sc and r.startswith("leads_to "):
                self._edges.setdefault(sc["map"], {}).setdefault(b["s"], {})[r[9:]] = (it.id, b["o"])
            elif it.kind == "fact" and "map" in sc and r.startswith("accepts "):
                self._accepts.setdefault(sc["map"], {}).setdefault(b["s"], {})[r[8:]] = (it.id, b["o"])
            elif it.kind == "skill" and str(b.get("s", "")).startswith("goal:") and r.startswith("via "):
                self._skills.setdefault(b["s"][5:], {})[r[4:]] = it.id

    # ------------------------------------------------------------------------------------------------ episodes
    def _begin(self, map_id=None):
        """A new episode: the agenda's goals start open again (their states are journaled anew), the failure memory's
        episode clock moves, and map facts learned before now count as carried (the first contradiction flags them)."""
        ag = self.agenda
        ag._done.clear()                     # achievements are per episode; the goals and gates stay declared
        ag.states.clear()
        if self.failures is not None:
            self.failures.new_episode()
        self._episode = {"at": len(self.store.journal), "flagged": set(), "n": self._episode["n"] + 1}
        self.store.note(episode="begin", n=self._episode["n"], map=None if map_id is None else str(map_id))

    def _end(self):
        """The episode ended: the action model's learned rules are committed as "action" items (newer versions refute
        older ones)."""
        if self.actions is not None and callable(getattr(self.actions, "commit", None)) \
                and getattr(self.actions, "store", None) is self.store and self.actions.actions():
            self.actions.commit()

    # ------------------------------------------------------------------------------------------------ reading
    def usable(self, iid):
        return iid is not None and self.store.usable(iid)

    def edges(self, map_id, at):
        """The usable map facts from key `at`: {action key: key reached}."""
        out = {}
        for a, (iid, to) in self._edges.get(str(map_id), {}).get(at, {}).items():
            if self.store.usable(iid):
                out[a] = to
        return out

    def accepted_at(self, map_id, at, akey):
        """What the map says about taking `akey` at `at`: True / False (a usable fact), or None."""
        hit = self._accepts.get(str(map_id), {}).get(at, {}).get(akey)
        return hit[1] if hit is not None and self.store.usable(hit[0]) else None

    def skills(self, goal):
        """The actions after which `goal`'s done check turned true (usable skill items)."""
        return [a for a, iid in sorted(self._skills.get(goal, {}).items()) if self.store.usable(iid)]

    def where(self, map_id, akey):
        """The keys on a map where `akey` was accepted (usable facts)."""
        return sorted(at for at, acts in self._accepts.get(str(map_id), {}).items()
                      if (h := acts.get(akey)) is not None and h[1] is True and self.store.usable(h[0]))

    def snapshot(self, **query):
        """What a decision is given: KnowledgeStore.snapshot(**query) (default: the facts and rules) — the active items
        as facts, the hints with why, the ids they rest on, the store's fingerprint and journal position."""
        query.setdefault("kinds", ("fact", "rule"))
        return self.store.snapshot(**query)

    @staticmethod
    def value(snapshot, s, r, default=None):
        """The value of the active fact (s, r) in a snapshot (the fact "knowledge" a rule reads), or `default`."""
        for row in (snapshot or {}).get("facts") or ():
            b = row.get("body") or {}
            if isinstance(b, dict) and b.get("s") == s and b.get("r") == r:
                return b.get("o")
        return default

    def report(self):
        """What was learned and from whom (the store's report), the agenda (done, open, blocked), the action model
        (per action: allowed values, refusal signatures, transitions — for the built-in model), the skills, the failure
        memory's blocks, and the map facts per map."""
        rep = {"store": self.store.report(),
               "agenda": {"goals": list(self.agenda.goals), "gates": list(self.agenda.gates),
                          "done": sorted(self.agenda.done())},
               "skills": {g: self.skills(g) for g in sorted(self._skills)},
               "maps": {m: sum(1 for at in e for iid, _ in e[at].values() if self.store.usable(iid))
                        for m, e in sorted(self._edges.items())}}
        am = self.actions
        if am is not None:
            summ = getattr(am, "summary", None)
            rep["actions"] = {a: summ(a) for a in am.actions()} if callable(summ) and callable(
                getattr(am, "actions", None)) else {"fingerprint": am.fingerprint()}
        if self.failures is not None:
            rep["failures"] = self.failures.blocked()
        return rep

    def __repr__(self):
        r = self.store.report()
        return (f"Knowledge({r['items']} items, {len(self.agenda.goals)} goals, {len(self.agenda.gates)} gates, "
                f"action model {'on' if self.actions is not None else 'off'})")


__all__ = ["Knowledge", "action_key", "split_action"]
