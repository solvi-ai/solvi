"""Where an admitted change goes: one target per artifact, each a thin adapter over what the library already has.

    KnowledgeTarget(store)      "item"      KnowledgeStore.add (its own gates run again) → the item id;
                                            rollback: KnowledgeStore.rollback (the previous version comes back)
    VersionsTarget(versions)    "compiled"  solvi.experimental.compile.Versions.add → the version number; rollback:
                                            the version in force goes back (Versions keeps every version)
    LearningTarget(loop)        "update"    solvi.experimental.learning.Learning.run (its own gates) → the version;
                                            rollback: Learning.rollback to the version before

A target implements `artifact`, `promote(change, measured) → a JSON-able reference` (raise Refused when the artifact's
own gates say no) and `rollback(ref, by=None, why=None)`."""
from __future__ import annotations


class Refused(RuntimeError):
    """The target's own gates refused the change at promotion (the reason is recorded as stage "promote")."""


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
        self.in_force = (versions.numbers() or [None])[-1]

    def promote(self, change, measured):
        before = self.in_force
        n = self.versions.add(change.payload, label=change.reason or None)
        self.in_force = n
        return {"version": n, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.in_force = ref["before"]

    def system(self, **kw):
        """The System of the version in force."""
        return self.versions.system(self.in_force, **kw)


class LearningTarget:
    artifact = "update"

    def __init__(self, loop):
        self.loop = loop

    def promote(self, change, measured):
        before = self.loop._baseline()
        rep = self.loop.run()
        if not rep.promoted:
            raise Refused(str(rep))
        return {"version": rep.version, "before": before}

    def rollback(self, ref, by=None, why=None):
        self.loop.rollback(ref["before"])


__all__ = ["KnowledgeTarget", "LearningTarget", "Refused", "VersionsTarget"]
