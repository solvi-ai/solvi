"""One change pipeline for four sources of change (experimental): a document version, verified experience, a person's
correction, a search proposal — every one a candidate that passes the same pluggable stages, is journaled, promoted
into what already holds that kind of knowledge, and can be rolled back.

    from solvi.experimental.changes import Change, Pipeline, Static, Consistency, CleanCases, FreshEnvironments, hazard_body

    ch = Change("outcome", payload={"kind": "rule", "body": hazard_body("lava", {"next": "lava"}, ["step"])},
                proposer="model:x", reason="died on lava 3 times", evidence=["dec-17", "dec-42"],
                expected={"failures": "-1 per 10 episodes"}, change_type="rule")
    pipe = Pipeline([KnownFailures(), Static(), Consistency(), CleanCases(successes), FreshEnvironments(...)],
                    storage="changes.jsonl", store=ks, approval={("rule", "outcome"): "gate"})
    res = pipe.propose(ch)              # → Result: "promoted" (with a version) | "rejected" (the stage) | "held"
    pipe.rule(res.change_id, True, by="ann")    # a person's ruling for a held change (or a compile Ruling keep / drop)
    pipe.versions(); pipe.current; pipe.rollback(0); pipe.lineage(res.change_id); pipe.observe(res.change_id, {...})

What it is: a generalisation of solvi.experimental.learning.Learning (a candidate, gates with recorded numbers, promote /
rollback, versions, a changelog) from one source (labels) to four, with the stages as objects like the knowledge
store's write gates. Nothing here stores knowledge: an admitted change goes into what already holds it (_targets.py) —
a KnowledgeStore item ("item": fact, rule, skill, action, with its sub-kind in body["kind"], e.g. "hazard", and the change
type in body["change"]), a compiled catalog version ("compiled": solvi.experimental.compile.Versions), or a learning update
("update": LearningTarget calls Learning.run() — decision: the loop's own gates (consistency, heldout, honesty,
act_guard, size) are the stages for that artifact, and the pipeline does not re-implement them).

The candidate. `Change(source, artifact="item", payload=...)`: source is "document", "outcome", "person" or "search";
it maps onto the knowledge store's SOURCES as document → "spec", outcome → "outcome", person → "person", and search →
"outcome" with by = "search:<proposer>", and only with evidence (Static refuses a search proposal without it) that
the confirmation stage confirmed: a search proposal is promotable only when Confirmation.on_evidence ran and admitted
it (else rejected: "search proposal: confirmation on its evidence did not run"), so the store's SourceGate keeps
refusing the system's own unverified answers. Also: the proposer and the reason, the
evidence, the proposer's expected effect (recorded; `observe` later records what happened next to it), the change
type (fact / rule / meaning) when known, the parent version, rule code to sandbox-check (`code`), and the rule's
executable form (`check`, not journaled; its code fingerprint is).

The stages (_stages.py; each `run(change, ctx) → StageVerdict` admit / refuse / hold with measurements): KnownFailures (a
candidate rejected before is refused — new), Static (wraps SourceGate, KnowledgeStore.proposal and the compiler's sandbox
check), Consistency (wraps ConsistencyGate; records key conflicts and narrows / widens), Confirmation (wraps a
compilation's acceptance or a replay on the evidence), CleanCases (wraps Agenda.dry_run), Impact (wraps decision_diff /
solvi.core.store.diff.diff), FreshEnvironments (ported: paired selection, deterministic replay, confirmation with a
bootstrap lower bound > 0, rules never crash, progress not lower), Approval (new: a table (kind, source) → "gate" |
"gate+person" | "person"), Canaries (wraps solvi.testing.honesty; a suspiciously perfect score holds).

Decision on approval "person": the automated stages still run and are journaled, but their refusals are advisory
(recorded with "advisory": true; the person decides) — except the static stage's, which stays final (a malformed
change cannot be promoted by anyone).

Hazard rules: the stored artifact is a `rule` item with body["kind"] == "hazard" (hazard_body); no second item is
written through Agenda.gate. gates_of(store, scope=None, parse=...) is a read-only view of the active hazard items as
(check, blocks) gate checks, by the same code path the dry run uses (gate_of), so an agent installs them live. The
condition language is supplied by the caller through parse= (lawlang + a macro catalog is the decided form);
equality of state keys (hazard_gate) is the placeholder.

The journal (_journal.py): records of kind "change" in a TraceStorage — "propose" (the change, its fingerprint, parent),
one "verdict" per stage, then "promoted" (version, target reference), "rejected" (the stage and reason), "held",
"ruling", "rollback" and "observed"; promoted / rejected / held carry the approval level ("level"). The storage's hash chain covers them (a tampered record fails verify()); with
the decisions' store they share one chain. Versions: version 0 is the state before the first change; each promotion is
the next number with the version in force as its parent; rollback(v) undoes every version after v through its target
(KnowledgeStore.rollback for items) and is itself a record. EvaluatorLock (ported): held-out seeds pinned by SHA-256,
the evaluator's code pinned, an access log, lock_out() for proposer processes, leak_scan.

What is checked: tests/test_changes.py — the chain verifies and a tampered record is caught; a hazard rule from
"outcome" passes static → consistency → clean cases → fresh environments (a toy corridor) → approval and is an active
rule item, listed in versions and undone by rollback; a rejected candidate is recorded with its stage and refused when
proposed again; approval "person" holds until a ruling; a search proposal needs confirmation on its evidence;
gates_of reads the active hazard items without writing; Static refuses a candidate catalog with a part that reads a
sensitive field (Static(sensitive=), Catalog(sensitive=)) or a citation not in the text (Static(text=, spec=)); Impact
refuses an exception whose diff moves decisions outside expected["only"] and admits the exact one.

Measured (2026-10, two stands; the stands are outside this repository, so no script in benchmarks/ yet): on a credit
policy the pipeline carried a document version (144 of 144 changed decisions traced to their clause), admitted 12 of 12
exact exceptions, held every widening one for a person and refused 8 of 8 exceptions that touched other decisions, 0
extraneous changes, 14 versions replayed; on a bank-request stream with a person's labels every one of 22 candidates
passed every stage (`nogate` = `gate`), so the stages added nothing there. The module is journal, impact analysis,
approval levels and rollback — infrastructure, not a claim that its gates admit fewer harmful changes.

What is not done:
- no rewind over an Environment (Confirmation.on_evidence takes the replay as a callback);
- no narrows / widens classifier (Consistency takes it as a callback; Z3 over lawlang is planned);
- no regression-scenario stage (`solvi test` cases.json);
- no scope flag on a world change for rules from experience;
- no validity periods in compile Versions (nor the change-type classifier in the compiler);
- the research harness's untrusted-code process sandbox is not ported (Static uses the compiler's sandbox check);
- a held change lives only in this process's memory until it is ruled on (its journal records survive, the Change
  object does not)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.runtime import vhash
from .. import warn_on_import
from ._journal import ChangeJournal, EvalLocked, EvaluatorLock
from ._paired import paired_bootstrap, run_episode
from ._stages import (DEFAULT_APPROVAL, LEVELS, Approval, Canaries, CleanCases, catalog_of, Confirmation, Consistency, Context,
                     FreshEnvironments, Impact, KnownFailures, SOURCES, Stage, Static, StageVerdict, allowed, gate_of,
                     gates_of, hazard_body, hazard_gate)
from ._targets import KnowledgeTarget, LearningTarget, Refused, VersionsTarget

warn_on_import(__name__)

STORE_SOURCE = {"document": "spec", "outcome": "outcome", "person": "person", "search": "outcome"}
ARTIFACTS = ("item", "compiled", "update")
CHANGE_TYPES = ("fact", "rule", "meaning")


@dataclass
class Change:
    """A candidate change (see the module docstring). payload: for "item", {"kind", "body", and optionally "scope",
    "derived_from", "supersedes", "valid", "reconfirm_after"}; for "compiled", a solvi.experimental.compile.Compiled;
    for "update", None (the learning loop reads its own labels) or the updated state, whose `fingerprint` (an attribute or
    a dict key) identifies the candidate (two updates to different states are different candidates)."""
    source: str
    artifact: str = "item"
    payload: Any = None
    proposer: str = ""
    reason: str = ""
    evidence: list = field(default_factory=list)
    expected: dict | None = None
    change_type: str | None = None
    parent: int | None = None
    code: str | None = None
    check: Callable | None = None

    def __post_init__(self):
        if self.artifact not in ARTIFACTS:
            raise ValueError(f"artifact is one of {ARTIFACTS}, not {self.artifact!r}")
        if self.change_type is not None and self.change_type not in CHANGE_TYPES:
            raise ValueError(f"change_type is one of {CHANGE_TYPES} or None, not {self.change_type!r}")
        self.evidence = list(self.evidence or ())

    @property
    def kind(self):
        """The artifact kind the approval table reads: the item's kind for an item, else the artifact."""
        if self.artifact == "item":
            return (self.payload or {}).get("kind")
        return self.artifact

    @property
    def store_source(self):
        """The knowledge store's source for this change (None for a search proposal without evidence)."""
        if self.source == "search" and not self.evidence:
            return None
        return STORE_SOURCE.get(self.source)

    @property
    def by(self):
        return f"{self.source}:{self.proposer}" if self.proposer else self.source

    def item_kwargs(self):
        """The arguments of KnowledgeStore.add / proposal for an item change."""
        p = dict(self.payload or {})
        body = p.get("body")
        if self.change_type is not None and isinstance(body, dict) and "change" not in body:
            body = {**body, "change": self.change_type}
        kw = {"kind": p.get("kind"), "body": body, "scope": p.get("scope"), "source": self.store_source, "by": self.by,
              "evidence": self.evidence or None, "derived_from": p.get("derived_from") or (),
              "supersedes": p.get("supersedes"), "valid": p.get("valid")}
        if p.get("reconfirm_after") is not None:
            kw["reconfirm_after"] = p["reconfirm_after"]
        return kw

    def content(self):
        """What identifies the candidate (the failure memory compares it): source, artifact, content, code."""
        if self.artifact == "item":
            c = {k: v for k, v in self.item_kwargs().items() if k not in ("by", "evidence")}
        else:                                         # "compiled", or an "update" whose payload names its state
            c = self.payload.get("fingerprint") if isinstance(self.payload, dict) else getattr(self.payload, "fingerprint", None)
        chk = None
        if self.check is not None:
            from ...core.provenance import code_fingerprint
            chk = code_fingerprint(self.check)
        return {"source": self.source, "artifact": self.artifact, "content": c, "code": self.code, "check": chk}

    def fingerprint(self):
        return vhash(self.content())[:24]

    def to_dict(self):
        return {**self.content(), "proposer": self.proposer, "reason": self.reason, "evidence": self.evidence,
                "expected": self.expected, "change_type": self.change_type, "parent": self.parent,
                "fingerprint": self.fingerprint()}


@dataclass
class Result:
    """What one proposal (or ruling) came to: status "promoted" | "rejected" | "held"; the version for a promotion; the
    stage that refused or held it; every verdict."""
    change_id: str
    status: str
    version: int | None = None
    stage: str | None = None
    reason: str = ""
    verdicts: list = field(default_factory=list)
    ref: Any = None

    @property
    def promoted(self):
        return self.status == "promoted"

    def __str__(self):
        head = {"promoted": f"change {self.change_id}: promoted as version {self.version}",
                "rejected": f"change {self.change_id}: rejected by {self.stage}: {self.reason}",
                "held": f"change {self.change_id}: held at {self.stage}: {self.reason}"}[self.status]
        return "\n".join([head] + [f"  {'✓' if v.ok else '…' if v.decision == 'hold' else '✗'} {v.stage}"
                                   + (f": {v.reason}" if v.reason else "") for v in self.verdicts])


SEARCH_UNCONFIRMED = "search proposal: confirmation on its evidence did not run"


def _confirmed_on_evidence(verdicts):
    """Did the confirmation stage run Confirmation.on_evidence and admit? (a search proposal is promotable only then)"""
    v = verdicts.get("confirmation")
    return v is not None and v.ok and bool((v.measured or {}).get("on_evidence"))


def _ruling(r):
    """A person's ruling → (ok, note): True / False, "approve" / "reject", or a compile Ruling ("keep" / "drop")."""
    if isinstance(r, bool):
        return r, ""
    if isinstance(r, str) and r in ("approve", "reject"):
        return r == "approve", ""
    v = getattr(r, "verdict", None)
    if v in ("keep", "drop"):
        return v == "keep", getattr(r, "note", "")
    raise ValueError(f"a ruling is True / False, 'approve' / 'reject' or a Ruling keep / drop, not {r!r}")


class Pipeline:
    """The change pipeline (see the module docstring). stages: Stage objects run in order; storage: the journal's
    TraceStorage (a path or a store); store: the KnowledgeStore items go into; approval: an Approval, or its table
    (appended as the last stage when no Approval is among the stages); targets: extra targets (VersionsTarget,
    LearningTarget) — a KnowledgeTarget on `store` is added by itself."""

    experimental = True

    def __init__(self, stages, storage, *, store=None, approval=None, targets=()):
        self.stages = list(stages)
        for s in self.stages:
            if not callable(getattr(s, "run", None)) or not getattr(s, "name", None):
                raise TypeError(f"a stage has a name and run(change, ctx) → StageVerdict; {s!r} has not")
        if approval is not None and not any(isinstance(s, Approval) for s in self.stages):
            self.stages.append(approval if isinstance(approval, Approval) else Approval(approval))
        self.journal = ChangeJournal(storage)
        self.store = store
        self.targets = {}
        if store is not None:
            self.targets["item"] = KnowledgeTarget(store)
        for t in targets:
            self.targets[t.artifact] = t
        self._held = {}                               # change id → (change, stage index, verdicts)

    # ------------------------------------------------------------------------------------------------ proposing
    def propose(self, change: Change) -> Result:
        """Run the stages on a candidate, record every verdict, then promote it, reject it, or hold it → Result."""
        if not isinstance(change, Change):
            raise TypeError("propose takes a Change")
        parent = self.current if change.parent is None else change.parent
        rec = self.journal.write("propose", change=change.to_dict(), fingerprint=change.fingerprint(), parent=parent,
                                 source=change.source, artifact=change.artifact)
        return self._run(change, rec["id"], 0, {}, parent)

    def rule(self, change_id, ruling, *, by) -> Result:
        """A person's ruling on a held change: approve (True, "approve", a Ruling "keep") or reject (False, "reject",
        a Ruling "drop"). The stages after the one that held it run next."""
        if change_id not in self._held:
            raise KeyError(f"no held change {change_id!r} in this pipeline")
        change, i, verdicts, parent = self._held.pop(change_id)
        ok, note = _ruling(ruling)
        self.journal.write("ruling", change=change_id, approve=ok, by=str(by), note=note)
        return self._run(change, change_id, i, verdicts, parent, ruling=(ok, str(by), note))

    def _run(self, change, cid, start, verdicts, parent, ruling=None):
        appr = next((s for s in self.stages if isinstance(s, Approval)), None)
        level = appr.level(change, verdicts) if appr is not None else "gate"
        ctx = Context(self, self.store, self.journal, cid, dict(verdicts), level, ruling)
        if ruling is not None:                        # the stage that held the change is decided by the ruling
            st = self.stages[start]
            v = StageVerdict("admit" if ruling[0] else "refuse", f"{'approved' if ruling[0] else 'rejected'} by {ruling[1]}"
                        + (f": {ruling[2]}" if ruling[2] else ""), {"level": level, "by": ruling[1]}, st.name)
            self.journal.write("verdict", change=cid, **v.to_dict(), advisory=False)
            ctx.verdicts[st.name] = v
            if not v.ok:
                return self._reject(change, cid, st.name, v.reason, ctx)
            start += 1
        for i in range(start, len(self.stages)):
            st = self.stages[i]
            try:
                v = StageVerdict.of(st.run(change, ctx), st.name)
            except Exception as e:  # noqa: BLE001 — a stage that raises refuses (recorded), the pipeline goes on
                v = StageVerdict.refuse(f"the stage raised {type(e).__name__}: {e}")
            v.stage = st.name
            advisory = (v.decision == "refuse" and level == "person" and st.name not in ("static", "approval"))
            self.journal.write("verdict", change=cid, **v.to_dict(), advisory=advisory)
            ctx.verdicts[st.name] = v
            if v.decision == "hold":
                self._held[cid] = (change, i, {k: x for k, x in ctx.verdicts.items() if k != st.name}, parent)
                self.journal.write("held", change=cid, stage=st.name, reason=v.reason, level=self._level(change, ctx))
                return Result(cid, "held", stage=st.name, reason=v.reason, verdicts=list(ctx.verdicts.values()))
            if v.decision == "refuse" and not advisory:
                return self._reject(change, cid, st.name, v.reason, ctx)
        if change.source == "search" and not _confirmed_on_evidence(ctx.verdicts):
            return self._reject(change, cid, "confirmation", SEARCH_UNCONFIRMED, ctx)
        target = self.targets.get(change.artifact)
        if target is None:
            return self._reject(change, cid, "promote", f"no target for artifact {change.artifact!r}", ctx)
        try:
            ref = target.promote(change, self._summary(ctx.verdicts))
        except Refused as e:
            return self._reject(change, cid, "promote", str(e), ctx)
        version = self._next_version()
        self.journal.write("promoted", change=cid, version=version, parent=parent, artifact=change.artifact, ref=ref,
                           expected=change.expected, level=self._level(change, ctx))
        return Result(cid, "promoted", version=version, verdicts=list(ctx.verdicts.values()), ref=ref)

    def _reject(self, change, cid, stage, reason, ctx):
        self.journal.write("rejected", change=cid, fingerprint=change.fingerprint(), stage=stage, reason=reason,
                           level=self._level(change, ctx))
        return Result(cid, "rejected", stage=stage, reason=reason, verdicts=list(ctx.verdicts.values()))

    def _level(self, change, ctx):
        """The approval level of a change as its final record states it: the approval table's, raised for a change the
        consistency stage classified as widening (the same rule Approval applies), whether or not Approval ran."""
        appr = next((s for s in self.stages if isinstance(s, Approval)), None)
        return appr.level(change, ctx.verdicts) if appr is not None else "gate"

    @staticmethod
    def _summary(verdicts):
        """What a target's own gates may read (a knowledge store's ShadowGate): gain, changed share, honest."""
        out = {}
        f = verdicts.get("fresh_environments")
        if f is not None and f.measured and "confirm" in f.measured:
            out["gain"] = f.measured["confirm"]["gain"][0]
        im = verdicts.get("impact")
        if im is not None and im.measured:
            out["changed"] = im.measured.get("share")
        c = verdicts.get("canaries")
        if c is not None:
            out["honest"] = c.ok
        return out

    # ------------------------------------------------------------------------------------------------ versions
    def history(self, action=None):
        """The journal's change records in order (dicts)."""
        return self.journal.records(action)

    def _state(self):
        in_force, promoted = [], {}
        for d in self.journal.records():
            if d["action"] == "promoted":
                promoted[d["version"]] = d
                in_force.append(d["version"])
            elif d["action"] == "rollback":
                in_force = [v for v in in_force if v not in set(d["undone"])]
        return in_force, promoted

    def _next_version(self):
        _, promoted = self._state()
        return max(promoted, default=0) + 1

    @property
    def current(self):
        """The version in force: the last promoted one not rolled back (0: before the first change)."""
        in_force, _ = self._state()
        return in_force[-1] if in_force else 0

    def versions(self):
        """Every promoted version → [{"version", "change", "parent", "artifact", "ref", "in_force", "id"}]."""
        in_force, promoted = self._state()
        return [{"version": v, "change": d["change"], "parent": d["parent"], "artifact": d["artifact"], "ref": d["ref"],
                 "in_force": v in in_force, "id": d["id"]} for v, d in sorted(promoted.items())]

    def rollback(self, version, *, by=None, why=None):
        """Undo every version in force after `version` (newest first) through its target; recorded. → the undone
        version numbers."""
        in_force, promoted = self._state()
        if version != 0 and version not in in_force:
            raise KeyError(f"version {version} is not in force (in force: {in_force})")
        undone = [v for v in reversed(in_force) if v > version]
        for v in undone:
            d = promoted[v]
            self.targets[d["artifact"]].rollback(d["ref"], by=by, why=why or f"rollback to version {version}")
        self.journal.write("rollback", to=version, undone=undone, by=None if by is None else str(by), why=why)
        return undone

    def lineage(self, change_id):
        """The change and its ancestors: [{"change", "version", "parent"}], following each one's parent version back to
        version 0."""
        props = {d["id"]: d for d in self.journal.records("propose")}
        _, promoted = self._state()
        by_change = {d["change"]: v for v, d in promoted.items()}
        out, cid = [], change_id
        while cid is not None:
            p = props[cid]
            out.append({"change": cid, "version": by_change.get(cid), "parent": p["parent"]})
            cid = promoted[p["parent"]]["change"] if p["parent"] in promoted else None
        return out

    def observe(self, change_id, effect: dict, *, by=None):
        """Record what the change did next to what its proposer expected → {"expected", "observed"}."""
        props = {d["id"]: d for d in self.journal.records("propose")}
        expected = props[change_id]["change"].get("expected")
        self.journal.write("observed", change=change_id, expected=expected, observed=effect,
                           by=None if by is None else str(by))
        return {"expected": expected, "observed": effect}

    def held(self):
        """The ids of the changes held in this pipeline, waiting for a ruling."""
        return sorted(self._held)


__all__ = ["allowed", "Approval", "ARTIFACTS", "Canaries", "catalog_of", "Change", "CHANGE_TYPES", "ChangeJournal", "CleanCases",
           "Confirmation", "Consistency", "Context", "DEFAULT_APPROVAL", "EvalLocked", "EvaluatorLock",
           "FreshEnvironments", "gate_of", "gates_of", "hazard_body", "hazard_gate", "Impact", "KnowledgeTarget",
           "KnownFailures",
           "LearningTarget", "LEVELS", "paired_bootstrap", "Pipeline", "Refused", "Result", "run_episode", "SOURCES",
           "Stage", "Static", "STORE_SOURCE", "StageVerdict", "VersionsTarget"]
