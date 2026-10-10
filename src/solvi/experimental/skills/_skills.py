"""Skills compiled from verified plans and the skill set (see the package docstring)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ._certificate import Certificate, EProcess
from ._model import Plan, _plain, state_key

MAX_COUNTEREXAMPLES = 32


@dataclass(eq=False)
class Skill:
    """System 1's rule "in this situation, for this goal, do this sequence". condition: a conjunction of atom names (all
    must hold; without a grammar, the start state's key: a case); actions: the sequence; goal: the goal's name; source:
    "verified" (a plan the code verified) or "unverified" (a control); support: the verified plans it was compiled
    from; outcomes: counts of its uses — ok / fail (every use), alone_ok / alone_fail (System 1 alone), supervised_ok /
    supervised_fail (re-verified on the exact model before a use), source_ok / source_fail (the run of the plan it was
    compiled from); certificate: its own EProcess (None without a certificate). status: "active" or "retired" (its
    alarm fired: not used alone until its own process certifies again). start / simulation: the first plan's start key
    and simulation hash; starts: the start keys of every plan merged into it; counterexamples: the atoms of states
    where it failed (a generalisation never covers one)."""
    condition: tuple
    actions: tuple
    goal: str
    source: str = "verified"
    support: int = 1
    outcomes: dict = field(default_factory=dict)
    certificate: EProcess | None = None
    id: str = ""
    status: str = "active"
    why: str | None = None
    start: str = ""
    simulation: str = ""
    starts: list = field(default_factory=list)
    counterexamples: list = field(default_factory=list)

    def __post_init__(self):
        self.condition = tuple(sorted(self.condition))
        self.actions = tuple(self.actions)
        for k in ("ok", "fail", "alone_ok", "alone_fail", "supervised_ok", "supervised_fail", "source_ok",
                  "source_fail"):
            self.outcomes.setdefault(k, 0)

    def to_dict(self) -> dict:
        return {"id": self.id, "goal": self.goal, "if": list(self.condition), "do": [_plain(a) for a in self.actions],
                "source": self.source, "support": self.support, "status": self.status, "why": self.why,
                "outcomes": dict(self.outcomes), "start": self.start, "simulation": self.simulation,
                "starts": len(self.starts), "counterexamples": len(self.counterexamples),
                "certificate": self.certificate.to_dict() if self.certificate is not None else None}

    def __repr__(self):
        return (f"Skill({self.id!r}, goal={self.goal!r}, if={list(self.condition)}, do={list(self.actions)}, "
                f"{self.status})")


class SkillSet:
    """The skills System 1 may run, their certificates and their export into the knowledge store.

    grammar: a solvi.experimental.laws Grammar (its atoms are read with action None; the conjunction size `k` does not
    bound a skill's condition) or None (a skill is a case: its start state's key). certificate: a Certificate (default
    Certificate()), or None — every skill acts alone at once (a control, not a guarantee). generalize: merge a new
    verified plan into an existing skill of the same goal and actions (below); key: the state key (canonical JSON).

    The condition rule (conservative): a new skill's condition is every atom of the grammar true at its plan's start.
    A later verified plan of the same goal with the same actions from another start widens it to the atoms both
    starts share (anti-unification over atoms: an atom that differed between two verified starts was shown not to be
    needed for this sequence) — unless the widened condition would hold on a state where the skill is recorded to
    have failed; then the plan becomes a skill of its own. Nothing else widens a condition. Widening restarts the
    skill's own certificate (its evidence was gathered on a narrower condition)."""

    def __init__(self, grammar=None, certificate: Certificate | None | bool = True, *, generalize: bool = True,
                 key=None):
        if certificate is True:
            certificate = Certificate()
        if certificate is False:
            certificate = None
        if certificate is not None and not isinstance(certificate, Certificate):
            raise TypeError("certificate= takes a Certificate (or None: no certificate, a control)")
        self.grammar, self.certificate, self.generalize = grammar, certificate, bool(generalize)
        self.key = key or state_key
        self.skills: list[Skill] = []
        self.pools: dict[str, EProcess] = {}
        self._items: dict[str, str] = {}             # skill id → its current knowledge item id
        self._n = 0

    # --- atoms and matching
    def atoms(self, state) -> frozenset:
        """The atoms true in `state` (without a grammar: its key, as one atom)."""
        if self.grammar is None:
            return frozenset({"key=" + self.key(state)})
        return frozenset(self.grammar.true(state, None))

    def candidates(self, state, goal, *, atoms=None) -> list[Skill]:
        """The skills of `goal` whose condition holds in `state` (active and retired), most specific first (more atoms;
        ties: the earlier skill)."""
        at = self.atoms(state) if atoms is None else atoms
        out = [s for s in self.skills if s.goal == goal and at.issuperset(s.condition)]
        return sorted(out, key=lambda s: -len(s.condition))

    def pool(self, skill) -> EProcess | None:
        """The process whose certification a skill acts on (its own under scope "skill")."""
        if self.certificate is None:
            return None
        if self.certificate.scope == "skill":
            return skill.certificate
        k = skill.goal if self.certificate.scope == "goal" else "*"
        if k not in self.pools:
            self.pools[k] = self.certificate.process()
        return self.pools[k]

    def usable(self, skill) -> bool:
        """May System 1 run it alone: active, and its certificate (its own, or its pool's) holds — always, without a
        certificate."""
        if self.certificate is None:
            return skill.status == "active"
        return skill.status == "active" and self.pool(skill).certified

    def match(self, state, goal, *, atoms=None) -> Skill | None:
        """The most specific skill of `goal` whose condition holds in `state` and whose certificate holds, or None."""
        for s in self.candidates(state, goal, atoms=atoms):
            if self.usable(s):
                return s
        return None

    def retired(self) -> list[Skill]:
        return [s for s in self.skills if s.status == "retired"]

    # --- compiling
    def add(self, skill: Skill) -> Skill:
        """Add a skill built by hand (it gets an id and a certificate)."""
        self._n += 1
        skill.id = skill.id or f"{skill.goal}#{self._n}"
        if skill.certificate is None and self.certificate is not None:
            skill.certificate = self.certificate.process()
        self.skills.append(skill)
        return skill

    def compile(self, goal: str, plan: Plan, state, *, verified: bool = True) -> Skill:
        """A plan found (and, with verified=True, verified) from `state` → the skill it becomes: an existing one widened
        by the condition rule, or a new one with the atoms true at `state`."""
        source = "verified" if verified else "unverified"
        at = self.atoms(state)
        acts = tuple(plan.actions)
        k = self.key(state)
        if self.generalize and self.grammar is not None:
            for s in self.skills:
                if s.goal != goal or s.actions != acts or s.source != source or s.status != "active":
                    continue
                new = tuple(sorted(at.intersection(s.condition)))
                if any(set(cx).issuperset(new) for cx in s.counterexamples):
                    continue                          # it would cover a state where this skill failed
                if new != s.condition:
                    s.condition = new
                    if s.certificate is not None and self.certificate is not None and self.certificate.scope == "skill":
                        s.certificate.restart()
                s.support += 1
                s.starts.append(k)
                return s
        s = Skill(tuple(sorted(at)), acts, goal, source=source, start=plan.start or k, simulation=plan.simulation,
                  starts=[k])
        return self.add(s)

    # --- outcomes
    def observe(self, skill: Skill, ok: bool, *, alone: bool = True, state=None, kind: str | None = None) -> dict:
        """One use of `skill`: did it reach its goal (from the real episode when it ran alone; from the exact model
        when System 2 re-verified it first: alone=False). Updates its outcomes and its certificate(s); a failure from
        `state` records that state's atoms as a counterexample. A skill whose own alarm fires is retired; a retired
        skill whose own process certifies again is back ("recertified").
        → {"certified", "alarm", "retired", "recertified", "pool_alarm"}."""
        kind = kind or ("alone" if alone else "supervised")
        skill.outcomes["ok" if ok else "fail"] += 1
        skill.outcomes[f"{kind}_{'ok' if ok else 'fail'}"] += 1
        if not ok and state is not None and len(skill.counterexamples) < MAX_COUNTEREXAMPLES:
            skill.counterexamples.append(tuple(sorted(self.atoms(state))))
        ev = {"certified": False, "alarm": False, "retired": False, "recertified": False, "pool_alarm": False}
        if self.certificate is None or skill.certificate is None:
            return ev
        own = skill.certificate.update(ok, alone)
        ev["certified"], ev["alarm"] = own["certified"], own["alarm"]
        if self.certificate.scope != "skill":
            p = self.pool(skill).update(ok, alone)
            ev["pool_alarm"], ev["certified"] = p["alarm"], p["certified"]
        if own["alarm"] and skill.status == "active":
            skill.status, ev["retired"] = "retired", True
            e = skill.certificate
            skill.why = (f"the alarm on its alone outcomes ({skill.outcomes['alone_fail']} failed of "
                         f"{skill.outcomes['alone_ok'] + skill.outcomes['alone_fail']}, promise "
                         f"{self.certificate.promise:g}); alarm {e.alarms}")
        elif own["certified"] and skill.status == "retired":
            skill.status, skill.why, ev["recertified"] = "active", None, True
        return ev

    # --- reading and export
    def report(self) -> dict:
        usable = [s for s in self.skills if self.usable(s)]
        return {"count": len(self.skills), "usable": len(usable), "retired": len(self.retired()),
                "verified": sum(1 for s in self.skills if s.source == "verified"),
                "certificate": self.certificate.to_dict() if self.certificate is not None else None,
                "pools": {k: p.to_dict() for k, p in self.pools.items()},
                "skills": [s.to_dict() for s in self.skills]}

    def to_knowledge(self, store, scope: dict | None = None) -> dict[str, str]:
        """Write the verified skills into a KnowledgeStore: each a "skill" item with body {"kind": "plan", "goal",
        "if", "do"}, source "outcome" (the code world answered), by "skills", evidence = the plan, its start key, the
        verifying simulation's hash, the starts merged, the outcomes and the certificate's state. A retired skill is
        retire()d with its reason (and asserted again when it is re-certified); a widened skill is a new item and the
        narrower version is retired. Unverified skills are never written. Idempotent. → skill id → item id."""
        from ...core.knowledge.store import item_id, plain_json
        out = {}
        sc = plain_json(scope or {})
        for s in self.skills:
            if s.source != "verified":
                continue
            body = {"kind": "plan", "goal": s.goal, "if": list(s.condition), "do": [_plain(a) for a in s.actions]}
            iid = item_id("skill", plain_json(body), sc)
            prev = self._items.get(s.id)
            if prev is not None and prev != iid and store.status(prev) == "active":
                store.retire(prev, why=f"widened into {iid}")
            st = store.status(iid)
            if st is None or (s.status == "active" and st != "active"):
                ev = {"skill": s.id, "plan": [_plain(a) for a in s.actions], "start": s.start,
                      "simulation": s.simulation, "starts": len(s.starts), "support": s.support,
                      "outcomes": dict(s.outcomes),
                      "certificate": s.certificate.to_dict() if s.certificate is not None else None}
                got = store.add("skill", body, sc, source="outcome", by="skills", evidence=ev)
                if got is None:
                    continue
                st = store.status(iid)
            if s.status == "retired" and st != "expired":
                store.retire(iid, why=s.why or "retired")
            self._items[s.id] = iid
            out[s.id] = iid
        return out


__all__ = ["Skill", "SkillSet"]
