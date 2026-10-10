"""Where an admitted change goes: one target per artifact, each a thin adapter over what the library already has.

    KnowledgeTarget(store)      "item"      KnowledgeStore.add (its own gates run again) → the item id;
                                            rollback: KnowledgeStore.rollback (the previous version comes back)
    VersionsTarget(versions)    "compiled"  solvi.core.versions.Versions.add → the version number; rollback:
                                            the version in force goes back (Versions keeps every version)
    LearningTarget(system, ...) "update"    the update ladder and its gates (_updates.py: consistency, held-out,
                                            honesty, act_guard, size) on the stored labels → the update's version;
                                            each gate's numbers are a stage verdict of the change ("update:<gate>");
                                            rollback: the state of the version before comes back

A target implements `artifact`, `promote(change, measured) → a JSON-able reference` (raise Refused when the artifact's
own gates say no) and `rollback(ref, by=None, why=None)`. A target that can answer also implements
`answerer(change or None, ctx)` → a callable input → answer (None: it cannot give one): the answerer in force for None,
the candidate's for a change — then the pipeline runs RefusalRegression on every change to it (not switchable).
VersionsTarget and LearningTarget answer (System.ask without storing); KnowledgeTarget does not (an item answers
nothing by itself: give the pipeline answer_of= for the agent that reads the store). A target whose own gates measure something returns
Promoted(ref, verdicts) and raises Refused(reason, verdicts=..., stage=...): the pipeline journals those verdicts as
the change's stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Refused(RuntimeError):
    """The target's own gates refused the change at promotion: the reason is recorded with `stage` (default "promote"),
    and `verdicts` (StageVerdicts of the target's gates, when it has them) are journaled before it."""

    def __init__(self, reason, *, verdicts=(), stage=None):
        super().__init__(reason)
        self.verdicts, self.stage = list(verdicts), stage


@dataclass
class Promoted:
    """A target's promotion with the verdicts of its own gates: `ref` is journaled as the change's reference, each
    verdict as a stage of the change."""
    ref: Any
    verdicts: list = field(default_factory=list)


class KnowledgeTarget:
    artifact = "item"

    def __init__(self, store):
        self.store = store

    def promote(self, change, measured):
        ks = self.store
        shadow = {"gain": measured.get("gain", 0.0), "changed": measured.get("changed"), "honest": measured.get("honest")}
        iid = ks.add(**change.item_kwargs(), shadow=shadow)
        if iid is None:
            raise Refused(ks.journal[-1].get("why", "refused by the knowledge store"))
        if ks.status(iid) != "active":
            held = [r for r in ks.journal if r.get("op") == "held" and r.get("item") == iid]
            why = "; ".join(f"{v['gate']}: {v['reason']}" for v in (held[-1]["verdicts"] if held else []) if not v["admit"])
            raise Refused(f"the knowledge store keeps it a {ks.status(iid)}" + (f" ({why})" if why else ""))
        return iid

    def rollback(self, ref, by=None, why=None):
        return self.store.rollback(ref, by=by, why=why)


class VersionsTarget:
    artifact = "compiled"

    def __init__(self, versions):
        self.versions = versions
        cur = getattr(versions, "current", None)              # solvi.core.versions.Versions keeps the version in force
        self.in_force = cur if cur is not None else (versions.numbers() or [None])[-1]

    def promote(self, change, measured):
        before = self.in_force
        n = self.versions.add(change.payload, label=change.reason or None)
        self.in_force = n
        return {"version": n, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.in_force = ref["before"]
        if ref["before"] is not None and callable(getattr(self.versions, "rollback", None)):
            self.versions.rollback(ref["before"], by=by, why=why)

    def system(self, **kw):
        """The System of the version in force."""
        return self.versions.system(self.in_force, **kw)

    def answerer(self, change, ctx=None):
        """input → Response: the version in force (None before any), or the candidate's catalog version (its
        system(allow_unaccepted=True)); None when it cannot be loaded."""
        try:
            if change is None:
                if self.in_force is None or not callable(getattr(self.versions, "system", None)):
                    return None
                s = self.versions.system(self.in_force)
            else:
                s = change.payload.system(allow_unaccepted=True)
        except Exception:  # noqa: BLE001 — no answerer: the refusal stage refuses ("cannot measure") or records it
            return None
        return lambda x: s.ask(x, store=False)


class LearningTarget:
    """Labelled updates as candidate changes: LearningTarget(system, storage=None, *, parts=None, ladder=None,
    gates=None, changelog=None, holdout=0.3, calibration=0.2, gate_teach=True) — the arguments of the ladder and gates
    (_updates.Updates; their defaults in _updates.LADDER / GATES). While a target is attached, System.teach only stores
    a correction (gate_teach=False keeps the instant update; detach() ends it) and a Change(artifact="update") learns
    from the stored labels: promoted with the version of the update, or refused at the first gate that failed."""
    artifact = "update"

    def __init__(self, system, storage=None, *, parts=None, ladder=None, gates=None, changelog=None, holdout=0.3,
                 calibration=0.2, gate_teach=True):
        from ._updates import Updates
        self.updates = Updates(system, storage, parts, ladder, gates, changelog, holdout, calibration, gate_teach)

    @property
    def parts(self):
        """{question: the decision part an update changes}."""
        return self.updates.parts

    def promote(self, change, measured):
        from ._stages import StageVerdict
        before = self.updates._baseline()
        rep = self.updates.run()
        if rep.action == "none":
            raise Refused(str(rep).splitlines()[0] + (f" ({len(rep.rejected_labels)} label(s) refused)"
                                                     if rep.rejected_labels else ""))
        verdicts = [StageVerdict("admit", "; ".join(f"{q}: {x['rung']} on {x['train']} training label(s)"
                                                    for q, x in rep.questions.items()),
                                 {"questions": rep.questions}, "update:ladder")]
        for g, r in rep.gates.items():
            why = r.get("why", "")
            if g == "promotion" and r.get("restored"):
                why += f" — the live parts were put back as they were (version {rep.current})"
            verdicts.append(StageVerdict("admit" if r.get("ok") else "refuse", why,
                                         {k: v for k, v in r.items() if k != "why"}, f"update:{g}"))
        if not rep.promoted:
            first = next(v for v in verdicts if not v.ok)
            raise Refused(first.reason, verdicts=verdicts, stage=first.stage)
        return Promoted({"version": rep.version, "before": before, "questions": rep.questions, "fp_before": rep.fp_before,
                         "fp_after": rep.fp_after, "record": rep.record_id}, verdicts)

    def rollback(self, ref, by=None, why=None):
        self.updates.rollback(ref["before"])

    def answerer(self, change, ctx=None):
        """input → Response: the live system for None; for a change, the candidate the ladder would build from the
        stored labels on a shadow (nothing promoted or recorded) — the system as it is when there is nothing new."""
        s = self.updates.system if change is None else self.updates.candidate()
        return lambda x: s.ask(x, store=False)

    def restore(self, version):
        """Restore any recorded version of the update (also one recorded by another process on the same store) → it."""
        return self.updates.rollback(version)

    def detach(self):
        """Stop gating System.teach (the update records stay)."""
        self.updates.detach()

    def labels(self):
        """The trusted labels → {"labels": [Label], "rejected": [(id, why)]}."""
        return self.updates.labels()

    def history(self):
        """Every update record (kind "update"), in order."""
        return self.updates.history()

    def versions(self):
        """The promoted update versions → [{"version", "fp", "time", "action", "id"}]."""
        return self.updates.versions()

    @property
    def current(self):
        """The update version in force (None when the live state is no recorded version)."""
        return self.updates.current

    def fingerprint(self):
        """The fingerprint of the updated parts' state."""
        return self.updates.fingerprint()

    def version_of(self, part_fp):
        """The update versions in which a part had this fingerprint (as a decision's trace records it)."""
        return self.updates.version_of(part_fp)


__all__ = ["KnowledgeTarget", "LearningTarget", "Promoted", "Refused", "VersionsTarget"]
