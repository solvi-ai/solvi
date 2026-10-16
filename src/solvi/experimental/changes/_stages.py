"""The stages of the change pipeline: the Stage protocol, its StageVerdict, and the built-in stages.

    class MyStage:                                 # the Stage protocol (like a WriteGate of the knowledge store)
        name = "mine"
        def run(self, change, ctx) -> StageVerdict: ...  # StageVerdict.admit / .refuse / .hold, with what it measured

Built in, in the order the pipeline usually runs them (what each one wraps, ports, or adds):

  KnownFailures      a candidate identical to one rejected (or rolled back with a reason) before is refused; in a
                     context, for a ttl, when given (reads the change journal)                                  new
  Static             the source and the artifact are well-formed; the store's SourceGate; rule code passes the
                     compiler's sandbox check; evidence exists; a parse callback                                 wraps
  Consistency        the store's ConsistencyGate on the proposed item; conflicts on its key; narrows / widens   wraps
  Confirmation       a callback: the compilation's acceptance, or the rule replayed on its own evidence          wraps
  CleanCases         Agenda.dry_run over recorded successes: how many the change would block                    wraps
  Impact             decision_diff / store.diff.diff / a callback: the share of past decisions it changes       wraps
  FreshEnvironments  paired selection on fresh worlds, then confirmation on other fresh worlds with a bootstrap
                     lower bound > 0, a deterministic replay, rules that never crash, progress not lower         ports
  Approval           a table (artifact kind, source) → "gate" | "gate+person" | "person"; holds until a ruling  new
  Canaries           a honesty run against a baseline; a suspiciously perfect score is a flag (hold)            wraps
  RefusalRegression  the share refused on a held should-abstain set before and after the change; refused when the
                     change answers what was refused before. Not switchable: the pipeline runs it for every
                     change whose target can answer, before Approval, final under every approval level           new
  SliceGuard         per-slice paired losses against the version in force and against the original version, so
                     harm in a slice under an overall gain and a chain of small changes ("salami") are refused   new

`ctx` is the pipeline's Context: the KnowledgeStore (ctx.store, may be None), the journal, the verdicts so far
(ctx.verdicts: stage name → StageVerdict), the approval level, and the person's ruling once given."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

DECISIONS = ("admit", "refuse", "hold")
SOURCES = ("document", "outcome", "person", "search")          # the four sources of change
LEVELS = ("gate", "gate+person", "person")


@dataclass
class StageVerdict:
    """A stage's answer about one change: "admit", "refuse" or "hold" (wait for a person), why, and what it measured."""
    decision: str
    reason: str = ""
    measured: dict | None = None
    stage: str = ""

    def __post_init__(self):
        if self.decision not in DECISIONS:
            raise ValueError(f"a verdict is one of {DECISIONS}, not {self.decision!r}")

    @property
    def ok(self):
        return self.decision == "admit"

    @classmethod
    def admit(cls, reason="", measured=None, stage=""):
        return cls("admit", reason, measured, stage)

    @classmethod
    def refuse(cls, reason="", measured=None, stage=""):
        return cls("refuse", reason, measured, stage)

    @classmethod
    def hold(cls, reason="", measured=None, stage=""):
        return cls("hold", reason, measured, stage)

    @classmethod
    def of(cls, x, stage=""):
        """A callback's answer as a StageVerdict: a StageVerdict, a knowledge-store gate Verdict (admit: bool), a bool, a dict
        {"ok": bool, "reason"?, ...measurements}, or None (admitted, nothing measured)."""
        if isinstance(x, StageVerdict):
            x.stage = x.stage or stage
            return x
        if x is None:
            return cls.admit(stage=stage)
        if isinstance(x, bool):
            return cls("admit" if x else "refuse", "" if x else "the check said no", stage=stage)
        if isinstance(x, dict):
            m = {k: v for k, v in x.items() if k not in ("ok", "reason")}
            return cls("admit" if x.get("ok", True) else "refuse", str(x.get("reason", "")), m or None, stage)
        if isinstance(getattr(x, "admit", None), bool):                  # gates.Verdict
            return cls("admit" if x.admit else "refuse", x.reason, x.measured, stage or getattr(x, "gate", ""))
        raise TypeError(f"a stage callback returns a StageVerdict, a bool, a dict or None, not {type(x).__name__}")

    def to_dict(self):
        return {"stage": self.stage, "decision": self.decision, "reason": self.reason, "measured": self.measured}


@runtime_checkable
class Stage(Protocol):
    """A stage of the change pipeline.

    You implement: `name` and `run(change, ctx) → StageVerdict` (admit / refuse / hold, with measurements).

    You get for free: it runs in the pipeline's order; its verdict is a journal record (kind "change"); a refusal
    rejects the change (recorded with the stage, so the same candidate is refused next time); a hold keeps it until
    `Pipeline.rule(...)`.

    Stability: experimental."""
    name: str

    def run(self, change, ctx) -> StageVerdict: ...


def _call(fn, change, ctx):
    return fn(change, ctx)


# ───────────────────────────────────────────────────────────── 9. failure memory (new)
IDENTITY_KEYS = ("s", "r", "o", "kind", "change")   # what a keyed item body says; its other keys are not its identity


def identity(content):
    """What the failure memory compares: Change.content() (or a journaled change dict) without the body keys of a keyed
    item (a body with a subject "s" and a relation "r") other than IDENTITY_KEYS — an added "nonce" or note does not
    make a rejected rule new. Other candidates: their whole content, as Change.fingerprint."""
    from ...core.runtime import vhash
    c = content.get("content")
    if content.get("artifact") == "item" and isinstance(c, dict):
        body = c.get("body")
        if isinstance(body, dict) and "s" in body and "r" in body:
            c = {**c, "body": {k: body[k] for k in IDENTITY_KEYS if k in body}}
    return vhash({**{k: content.get(k) for k in ("source", "artifact", "code", "check")}, "content": c})[:24]


class KnownFailures:
    """A candidate whose identity (identity(): Change.fingerprint's source, artifact, content and code, with a keyed
    item's body read as its subject, relation, value and kind only) was rejected before, or was rolled back with a
    reason (Pipeline.rollback(why=...), failed=True), is refused with the earlier stage and reason — a search does not
    walk in circles, and an extra key in a rule's body does not make a rejected rule new. A stage that raised (status
    "error") is not a failure: the candidate runs again. Reads the change journal (kept up to date incrementally).

    By default a failure is remembered for ever and everywhere. `context(change)` → a JSON value (the tests' version,
    a season, a regime): a failure counts only for a candidate proposed in the same context — a change rejected under
    old tests is retried under new ones. `ttl` (seconds): a failure older than that no longer counts (as
    solvi.core.knowledge.FailureMemory unblocks a plan after a window); the age is `clock()` (default time.time) minus
    the journal record's time, so give the journal's storage the same clock. `key(change)` → a JSON value replaces
    identity() (two candidates with the same key are one). The key and context are journaled in this stage's verdict
    (measured["memory"]), so the memory survives the process. A refusal by this stage does not renew a failure.

    rollbacks: do rollbacks with a reason count as failures (default True; False: only rejections do)."""
    name = "failure_memory"

    def __init__(self, *, rollbacks=True, key=None, context=None, ttl=None, clock=None):
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or ttl <= 0):
            raise ValueError(f"ttl is a number of seconds > 0 or None, not {ttl!r}")
        self.rollbacks, self.key, self.context, self.ttl = bool(rollbacks), key, context, ttl
        self.clock = clock or __import__("time").time

    @staticmethod
    def _hash(x):
        from ...core.runtime import vhash
        return vhash(x)[:24]

    def _memory(self, change):
        """(key, context) of a candidate as this stage compares them."""
        k = identity(change.content()) if self.key is None else self._hash(self.key(change))
        return k, (None if self.context is None else self._hash(self.context(change)))

    @staticmethod
    def _step(st, d):
        """The failure index over the journal: (key, context) → the latest rejection, the latest failed rollback (kept
        up to date by ChangeJournal.fold). A candidate's (key, context) is what this stage journaled for it, else its
        identity() in no context."""
        a, cid = d.get("action"), d.get("change")
        if a == "propose":
            st["mem"][d["id"]] = (identity(d.get("change") or {}), None)
        elif a == "verdict" and d.get("stage") == KnownFailures.name and cid in st["mem"]:
            m = (d.get("measured") or {}).get("memory")
            if isinstance(m, list) and len(m) == 2:
                st["mem"][cid] = tuple(m)
        elif a == "rejected" and d.get("stage") != KnownFailures.name and cid in st["mem"]:
            st["failed"][st["mem"][cid]] = ("rejected", d.get("time"), d, cid)
        elif a == "rollback" and d.get("failed"):
            for c in d.get("changes") or ():
                if c in st["mem"]:
                    st["rolled"][st["mem"][c]] = ("rolled back", d.get("time"), d, c)

    def run(self, change, ctx):
        mem, fp = self._memory(change), change.fingerprint()
        idx = ctx.journal.fold("known_failures", lambda: {"mem": {}, "failed": {}, "rolled": {}}, self._step)
        now = self.clock()
        for table in ("failed", "rolled") if self.rollbacks else ("failed",):
            hit = idx[table].get(mem)
            if hit is None:
                continue
            what, t, d, cid = hit
            age = None if t is None else now - float(t)
            if self.ttl is not None and age is not None and age >= self.ttl:
                continue
            when = "" if self.ttl is None or age is None else f" ({age:.0f}s ago)"
            if what == "rolled back":
                return StageVerdict.refuse(f"rolled back before{when}: {d.get('why')}", {"earlier": cid})
            same = "" if d.get("fingerprint") == fp or self.key is not None else \
                " (the same subject, relation and value; other body keys differ)"
            return StageVerdict.refuse(f"rejected before{when} by stage {d.get('stage')!r}{same}: {d.get('reason')}",
                                       {"earlier": cid})
        if self.key is None and self.context is None:
            return StageVerdict.admit()
        return StageVerdict.admit(measured={"memory": list(mem)})


# ───────────────────────────────────────────────────────────── 1. static (wraps)
class Static:
    """Well-formedness before anything is run.

    Checks: the source is one of the four; a "search" proposal carries evidence (it enters the store as an
    observation, so it must point at what was observed); an item's kind and body are accepted by
    KnowledgeStore.proposal and its source by the store's SourceGate; `change.code` (rule code) passes the compiler's
    sandbox check (solvi.experimental.compile.sandbox.check: pure functions, standard-library imports); each evidence id
    exists (`evidence_exists(eid) → bool`, e.g. a TraceStorage's record or an experience log); `parse(change, ctx)` →
    a list of problems (e.g. a lawlang parse).

    A candidate that carries a catalog (catalog_of: a Catalog or a System as the payload, or as its "catalog" /
    "system" key; a catalog version — solvi.core.versions.CatalogVersion, a Compiled too — when `sensitive` or `text` is given) is
    refused when solvi.check.lint flags a part that reads a sensitive field (`reads_sensitive`, `reads_sensitive_via`):
    the catalog's own Catalog(sensitive=...) and `sensitive` here; and, with `text` (the policy its parts cite), when
    solvi.check.citations finds an error — a quote not in the text, and with `spec` (a Spec) a clause that does not
    exist or does not hold the quote. `catalog_of(payload)` → a Catalog / System or None replaces how the catalog is
    found."""
    name = "static"

    def __init__(self, *, parse: Callable | None = None, evidence_exists: Callable | None = None, sandbox: bool = True,
                 sensitive=None, catalog_of: Callable | None = None, text: str | None = None, spec=None):
        self.parse, self.evidence_exists, self.sandbox = parse, evidence_exists, sandbox
        self.sensitive = None if sensitive is None else ([sensitive] if isinstance(sensitive, str) else list(sensitive))
        self.catalog_of = catalog_of
        self.text, self.spec = text, spec

    def run(self, change, ctx):
        problems = []
        if change.source not in SOURCES:
            problems.append(f"source {change.source!r} is not one of {SOURCES}")
        elif change.source == "search" and not change.evidence:
            problems.append("a search proposal carries evidence (stored ids of what it was confirmed on): a proposal "
                            "without it is the system's own unverified answer")
        if change.artifact == "item":
            if ctx.store is None:
                problems.append("an item change needs a KnowledgeStore (Pipeline(store=...))")
            elif not problems:
                try:
                    prop = ctx.store.proposal(**change.item_kwargs())
                except (ValueError, TypeError) as e:
                    problems.append(f"not a knowledge item: {e}")
                else:
                    from ...core.knowledge.gates import SourceGate
                    v = SourceGate().admit(ctx.store, prop)
                    if not v.admit:
                        problems.append(f"source gate: {v.reason}")
        if change.code is not None and self.sandbox:
            from ..compile import sandbox
            problems += [f"sandbox: {p}" for p in sandbox.check(change.code)]
        if self.evidence_exists is not None:
            missing = [e for e in change.evidence if not self.evidence_exists(e)]
            if missing:
                problems.append(f"evidence not found: {missing[:5]}")
        if self.parse is not None:
            problems += list(self.parse(change, ctx) or ())
        cat = self._catalog(change.payload)
        if cat is not None:
            problems += self._lint(cat)
        return StageVerdict.refuse("; ".join(problems), {"problems": problems}) if problems else StageVerdict.admit()

    def _catalog(self, payload):
        """The candidate's catalog (a Catalog) or None."""
        if self.catalog_of is not None:
            found = self.catalog_of(payload)
        else:
            found = catalog_of(payload, compiled=bool(self.sensitive) or self.text is not None)
        if found is None:
            return None
        return found.catalog if hasattr(found, "questions") and not hasattr(found, "rules") else found

    def _lint(self, cat):
        from ...check import Report, _sensitive, citations
        rep = Report()
        _sensitive(cat, rep, self.sensitive)
        if self.text is not None:
            rep.findings += citations(cat, self.text, spec=self.spec).findings
        return [f"{f.code}: {f.where}: {f.message}" for f in rep.errors]


def catalog_of(payload, *, compiled=True):
    """The catalog a change carries → a Catalog, a System or None: the payload itself (a Catalog or a System), a dict
    payload's "system" / "catalog", or — with compiled=True — a catalog version's (CatalogVersion.catalog(), checked and
    loaded as pure functions)."""
    if isinstance(payload, dict):
        payload = payload.get("system", payload.get("catalog"))
    if payload is None:
        return None
    if callable(getattr(payload, "catalog", None)) and hasattr(payload, "source"):      # a CatalogVersion
        return payload.catalog(allow_unaccepted=True)[0] if compiled else None
    if hasattr(payload, "catalog") and hasattr(payload, "questions"):                  # a System
        return payload
    if hasattr(payload, "parts") and hasattr(payload, "rules"):                        # a Catalog
        return payload
    return None


# ───────────────────────────────────────────────────────────── 2. consistency (wraps)
class Consistency:
    """The knowledge store's ConsistencyGate on the proposed item (premises exist and are present; not self-defeating),
    plus what it measures: the active items on the same key holding another value (`conflicts`: the store resolves
    them by source rank or asks a person — not a refusal here) and, with `classify(change, ctx) → "narrows" | "widens"`,
    the direction of the behaviour change (Approval upgrades "gate" to "gate+person" for a widening change); classify
    may also return a dict {"direction": ..., other measurements} — all of it is recorded. An artifact other than an
    item is admitted with only the classification."""
    name = "consistency"

    def __init__(self, *, classify: Callable | None = None):
        self.classify = classify

    def run(self, change, ctx):
        m = {}
        if self.classify is not None:
            d = self.classify(change, ctx)
            if isinstance(d, dict):                   # {"direction", ...what it measured, e.g. conflicts}
                m.update(d)
            else:
                m["direction"] = d
        if change.artifact != "item" or ctx.store is None:
            return StageVerdict.admit("not a knowledge item", m or None)
        from ...core.knowledge.gates import ConsistencyGate
        ks = ctx.store
        prop = ks.proposal(**change.item_kwargs())
        v = ConsistencyGate().admit(ks, prop)
        if prop["key"] is not None:
            others = [i for i in ks.by_key.get(prop["key"], ()) if i != prop["id"] and ks.status(i) == "active"]
            m["conflicts"] = sorted(i for i in others if ks.items[i].value != (prop["body"] or {}).get("o"))
        return StageVerdict("admit" if v.admit else "refuse", v.reason, m or None)


# ───────────────────────────────────────────────────────────── 3. independent confirmation (wraps)
class Confirmation:
    """Independent confirmation by a callback `check(change, ctx)` → a StageVerdict, bool or dict. Two ready-made ones:

    Confirmation.compiled()           a "compiled" artifact is confirmed when the compilation was accepted (two drafts
                                      agree on the pool, separate tests pass, the person's rulings hold —
                                      solvi.experimental.compile);
    Confirmation.on_evidence(replay)  every evidence id must be confirmed by `replay(eid, change) → bool` (an outcome rule
                                      must fire on what it was learned from, e.g. rewound k steps before the failure).

    refusals: should-abstain inputs kept with the confirmation's examples; RefusalRegression reads them."""
    name = "confirmation"

    def __init__(self, check: Callable, *, refusals=()):
        self.check = check
        self.refusals = refusals                      # should-abstain inputs kept with it (read by RefusalRegression)

    def run(self, change, ctx):
        return StageVerdict.of(_call(self.check, change, ctx))

    @classmethod
    def compiled(cls, **kw):
        def check(change, ctx):
            c = change.payload
            ok = bool(getattr(c, "accepted", False))
            return StageVerdict("admit" if ok else "refuse", "" if ok else f"compilation not accepted: {getattr(c, 'reason', '')}",
                           {"fingerprint": getattr(c, "fingerprint", None)})
        return cls(check, **kw)

    @classmethod
    def on_evidence(cls, replay: Callable, *, min_share: float = 1.0, **kw):
        def check(change, ctx):
            if not change.evidence:
                return StageVerdict.refuse("no evidence to confirm the change on")
            hits = [bool(replay(e, change)) for e in change.evidence]
            share = sum(hits) / len(hits)
            ok = share >= min_share
            return StageVerdict("admit" if ok else "refuse",
                                "" if ok else f"confirmed on {sum(hits)} of {len(hits)} evidence",
                                {"confirmed": sum(hits), "evidence": len(hits), "share": share, "on_evidence": True})
        return cls(check, **kw)


# ───────────────────────────────────────────────────────────── 4. clean cases (wraps Agenda.dry_run)
def hazard_body(name, when, actions, **extra):
    """A hazard rule as a knowledge item body: {"s": "hazard:<name>", "r": "forbid", "o": {"when", "actions"}, "kind":
    "hazard"} — in state `when`, the actions matching `actions` (glob patterns) are forbidden. This `rule` item is the
    stored artifact; no second item is written through Agenda.gate (gates_of gives a read-only view instead).

    The condition language is the caller's: pass `parse=` (body → (check, blocks)) to gate_of / gates_of / CleanCases,
    e.g. a lawlang condition over a macro catalog. The built-in hazard_gate reads `when` as equality of state keys
    only — a placeholder, not the decided rule form."""
    return {"s": f"hazard:{name}", "r": "forbid", "o": {"when": dict(when), "actions": list(actions)}, "kind": "hazard",
            **extra}


def hazard_gate(body):
    """A hazard body → (check, blocks) for Agenda.gate: check(state) is True (allowed) unless every `when` key matches
    (equality only: the placeholder condition language; see hazard_body)."""
    o = body["o"]
    when = dict(o["when"])

    def check(state):
        return not all(isinstance(state, dict) and state.get(k) == v for k, v in when.items())
    return check, list(o["actions"])


def gate_of(change, parse: Callable = hazard_gate):
    """The (check, blocks) a change installs as an agenda gate: `change.check` when given (with blocks from its body),
    else `parse(body)` (default hazard_gate)."""
    body = (change.payload or {}).get("body") or {}
    if change.check is not None:
        return change.check, list((body.get("o") or {}).get("actions") or ["*"])
    return parse(body)


def gates_of(store, scope=None, *, parse: Callable = hazard_gate):
    """A read-only view: the (check, blocks) of every active, usable hazard rule item in a KnowledgeStore (under `scope`
    when given), by the same `parse` the dry run uses → [(check, blocks)]. An agent installs them live
    (Agenda.gate(name, check, blocks=blocks) on an Agenda without a store, or `allowed(state, action, gates)`); nothing
    is written, so no item is duplicated."""
    out = []
    for r in store.find(kind="rule", scope=scope, status="active"):
        if (r["body"] or {}).get("kind") == "hazard" and store.usable(r["id"]):
            out.append(parse(r["body"]))
    return out


def allowed(state, action, gates):
    """Does any (check, blocks) gate forbid `action` in `state`? (a policy's helper) → bool."""
    for check, blocks in gates:
        if any(fnmatch.fnmatchcase(str(action), b) for b in blocks) and not check(state):
            return False
    return True


class CleanCases:
    """How many recorded successes the change would block: the change installed as a gate of an empty
    solvi.core.knowledge.Agenda and `dry_run(records)` over (state, action) pairs that worked (windows without a failure,
    accepted calls). Refused when the share of records it blocks exceeds `max_share` (default 0: a hazard rule may not
    block anything that worked). records: a list or `records(change, ctx)`; gate: change → (check, blocks) (default
    gate_of with `parse`); parse: the condition language, body → (check, blocks) (default hazard_gate). refusals: the
    should-abstain inputs kept next to the clean cases (RefusalRegression reads them)."""
    name = "clean_cases"

    def __init__(self, records, *, gate: Callable | None = None, parse: Callable = hazard_gate, max_share: float = 0.0,
                 refusals=()):
        self.records, self.max_share, self.refusals = records, float(max_share), refusals
        self.gate = gate if gate is not None else (lambda change: gate_of(change, parse))

    def run(self, change, ctx):
        from ...core.knowledge.agenda import Agenda
        recs = self.records(change, ctx) if callable(self.records) else self.records
        recs = list(recs)
        check, blocks = self.gate(change)
        ag = Agenda(None).gate("candidate", check, blocks=blocks)
        rep = ag.dry_run(recs)
        share = rep["blocked_any"] / rep["records"] if rep["records"] else 0.0
        m = {"records": rep["records"], "would_block": rep["blocked_any"], "share": share,
             "examples": rep["gates"]["candidate"]["examples"]}
        if share > self.max_share:
            return StageVerdict.refuse(f"would block {rep['blocked_any']} of {rep['records']} recorded successes "
                                  f"({share:.1%} > {self.max_share:.1%})", m)
        return StageVerdict.admit("", m)


# ───────────────────────────────────────────────────────────── 5. impact (wraps decision_diff / diff / redecide)
class Impact:
    """The share of past decisions the change would change. measure(change, ctx) → {"changed", "total"} (or a
    DiffReport / DecisionDiff); refused when changed / total > max_share (None: only recorded), or when the measure's
    dict says "ok": False (the caller's own rule; its "reason" is the refusal's).

    Exact exceptions: a change whose `expected["only"]` lists the decisions it may change (stored decision ids; with a
    DecisionDiff over inputs, the inputs' ids) is refused when the diff moves any other — reason "changes N
    decision(s) other than the allowed ones: [the first ten ids]" — and the measurement records `extraneous` (the ids
    outside `only`). A dict measure gives the ids it changed as "ids" (or its own "extraneous"). Ready-made:

    Impact.of_system(storage, system_of)   solvi.core.store.diff.diff(storage, system_of(change, ctx)) — the stored
                                           decisions re-run with the candidate system;
    Impact.of_compiled(storage, old_of)    solvi.core.versions.decision_diff(old_of(ctx), change.payload,
                                           store=storage) — with the clauses that cause each change."""
    name = "impact"

    def __init__(self, measure: Callable, *, max_share: float | None = None):
        self.measure, self.max_share = measure, max_share

    def run(self, change, ctx):
        r = self.measure(change, ctx)
        if hasattr(r, "moved") and hasattr(r, "total"):                 # DecisionDiff
            ids = list(dict.fromkeys(x.get("input", x["id"]) for x in r.moved))
            m = {"changed": len(r.moved), "total": r.total, "ids": ids[:20]}
        elif hasattr(r, "changed") and hasattr(r, "total"):             # DiffReport
            ids = list(dict.fromkeys(x["id"] for x in r.changed))
            m = {"changed": len(r.changed), "total": r.total, "ids": ids[:20]}
        else:
            m = dict(r)
            ids = m.get("ids")
        own = (m.pop("ok", True), str(m.pop("reason", "")))
        share = m["changed"] / m["total"] if m.get("total") else 0.0
        m["share"] = share
        only = (change.expected or {}).get("only") if isinstance(change.expected, dict) else None
        if only is not None:
            allowed = set(only)
            if "extraneous" in m:
                ext = list(m["extraneous"])
            elif ids is not None:
                ext = [i for i in dict.fromkeys(ids) if i not in allowed]
            else:
                return StageVerdict.refuse("expected['only'] is set, but the impact measure gives no decision ids to "
                                           "compare with it", m)
            m["extraneous"] = ext
            if ext:
                return StageVerdict.refuse(f"changes {len(ext)} decision(s) other than the allowed ones: {ext[:10]}", m)
        if not own[0]:
            return StageVerdict.refuse(own[1] or "the impact measure refused the change", m)
        if self.max_share is not None and share > self.max_share:
            return StageVerdict.refuse(f"changes {m['changed']} of {m['total']} past decisions ({share:.1%} > "
                                  f"{self.max_share:.1%})", m)
        return StageVerdict.admit("", m)

    @classmethod
    def of_system(cls, storage, system_of: Callable, *, max_share=None, **filters):
        def measure(change, ctx):
            from ...core.store.diff import diff
            return diff(storage, system_of(change, ctx), only=_only(change), **filters)
        return cls(measure, max_share=max_share)

    @classmethod
    def of_compiled(cls, storage, old_of: Callable, *, max_share=None, **filters):
        def measure(change, ctx):
            from ...core.versions import decision_diff
            return decision_diff(old_of(ctx), change.payload, store=storage, only=_only(change), **filters)
        return cls(measure, max_share=max_share)


def _only(change):
    return (change.expected or {}).get("only") if isinstance(change.expected, dict) else None


# ───────────────────────────────────────────────────────────── 6. fresh environments (ported)
class FreshEnvironments:
    """The candidate against the current agent on fresh worlds of an Environment, paired by seed (ported from a gated
    evolution harness).

    env: a function → a new Environment; agent: a function (change or None, ctx) → a policy (state, actions) → action
    (None: the current agent); select / confirm: seeds, or functions (change, ctx) → seeds — two disjoint sets of worlds
    never used before (draw them fresh per candidate; never the held-out evaluation seeds, see EvaluatorLock); score:
    row → number, higher is better (default: 1 when the episode did not fail); progress: row → number (default the
    episode's info["progress"]). on_episode_start: a function (policy, seed) called before every episode, with the
    policy about to play and the world's seed — to reset a policy's memory between worlds, seed its own randomness or
    log; None (default): nothing.

    The bar, in order: (1) selection — the mean paired gain on `select` is > 0; (2) the candidate's rules never crash
    (a policy that raises); (3) replay — the candidate replays its first `replay` selection worlds with the same
    actions; (4) confirmation — on `confirm`, the bootstrap lower bound of the paired gain is > `min_lower`; (5)
    progress not lower — the upper bound of the paired progress change on `confirm` is ≥ −`progress_tolerance`
    (a rule that makes the agent overcautious is refused). Everything measured is recorded."""
    name = "fresh_environments"

    def __init__(self, env: Callable, agent: Callable, *, select, confirm, score: Callable | None = None,
                 progress: Callable | None = None, max_steps: int = 200, boot: int = 2000, seed: int = 0,
                 min_lower: float = 0.0, progress_tolerance: float = 0.0, replay: int = 2,
                 on_episode_start: Callable | None = None):
        if on_episode_start is not None and not callable(on_episode_start):
            raise TypeError("on_episode_start is a function (policy, seed) or None")
        self.env, self.agent, self.select, self.confirm = env, agent, select, confirm
        self.on_episode_start = on_episode_start
        self.score = score or (lambda r: 0.0 if r["failed"] else 1.0)
        self.progress = progress or (lambda r: r["progress"])
        self.max_steps, self.boot, self.seed = max_steps, boot, seed
        self.min_lower, self.progress_tolerance, self.replay = min_lower, progress_tolerance, replay

    def _seeds(self, s, change, ctx):
        return list(s(change, ctx) if callable(s) else s)

    def _play(self, policy, seeds):
        from ._paired import run_episode
        rows = []
        for s in seeds:
            if self.on_episode_start is not None:
                self.on_episode_start(policy, s)
            rows.append(run_episode(self.env(), policy, s, max_steps=self.max_steps))
        return rows

    def run(self, change, ctx):
        from ._paired import paired_bootstrap
        sel, conf = self._seeds(self.select, change, ctx), self._seeds(self.confirm, change, ctx)
        if not sel or not conf:
            return StageVerdict.refuse("no selection or confirmation worlds")
        if set(sel) & set(conf):
            return StageVerdict.refuse(f"selection and confirmation worlds overlap: {sorted(set(sel) & set(conf))[:5]}")
        base, cand = self.agent(None, ctx), self.agent(change, ctx)
        rb, rc = self._play(base, sel), self._play(cand, sel)
        m, lo, hi = paired_bootstrap([self.score(r) for r in rb], [self.score(r) for r in rc], boot=self.boot,
                                     seed=self.seed)
        meas = {"select": {"worlds": len(sel), "gain": [m, lo, hi], "failed": [sum(r["failed"] for r in rb),
                                                                              sum(r["failed"] for r in rc)]}}
        errs = sum(r["errors"] for r in rc)
        if errs:
            return StageVerdict.refuse(f"the candidate's rules crashed in {errs} selection world(s): "
                                  f"{next(r['error'] for r in rc if r['errors'])}", meas)
        if m <= 0:
            return StageVerdict.refuse(f"no gain on the selection worlds (mean {m:+.3f})", meas)
        again = self._play(self.agent(change, ctx), sel[:self.replay])
        same = all(a["actions"] == b["actions"] for a, b in zip(again, rc))
        meas["deterministic"] = same
        if not same:
            return StageVerdict.refuse("not deterministic: a selection world replayed with other actions", meas)
        cb, cc = self._play(base, conf), self._play(cand, conf)
        errs = sum(r["errors"] for r in cc)
        cm, clo, chi = paired_bootstrap([self.score(r) for r in cb], [self.score(r) for r in cc], boot=self.boot,
                                        seed=self.seed + 1)
        pm, plo, phi = paired_bootstrap([self.progress(r) for r in cb], [self.progress(r) for r in cc], boot=self.boot,
                                        seed=self.seed + 2)
        meas["confirm"] = {"worlds": len(conf), "gain": [cm, clo, chi], "progress": [pm, plo, phi],
                           "failed": [sum(r["failed"] for r in cb), sum(r["failed"] for r in cc)]}
        if errs:
            return StageVerdict.refuse(f"the candidate's rules crashed in {errs} confirmation world(s)", meas)
        if clo <= self.min_lower:
            return StageVerdict.refuse(f"not confirmed: the gain's lower bound {clo:+.3f} ≤ {self.min_lower:+.3f}", meas)
        if phi < -self.progress_tolerance:
            return StageVerdict.refuse(f"progress lower: the change's upper bound {phi:+.3f} < {-self.progress_tolerance:+.3f}",
                                  meas)
        return StageVerdict.admit("", meas)


# ───────────────────────────────────────────────────────────── 7. approval (new: a table)
DEFAULT_APPROVAL = {
    ("rule", "outcome"): "gate", ("skill", "outcome"): "gate", ("action", "outcome"): "gate",
    ("fact", "outcome"): "gate", ("fact", "person"): "gate", ("rule", "person"): "gate",
    ("rule", "search"): "gate+person", ("skill", "search"): "gate+person",
    ("rule", "document"): "gate+person", ("compiled", "document"): "gate+person",
    ("update", "person"): "gate", ("update", "outcome"): "gate",
    ("evaluator", "person"): "person", ("goal", "person"): "person",
}


class Approval:
    """Who approves: table {(artifact kind, source): "gate" | "gate+person" | "person"} (the artifact kind is the
    item's kind for an item — "rule", "fact", ... — else "compiled" / "update"; a key may also be the kind alone);
    `default` for a pair not in the table. "gate": admitted. "gate+person" and "person": held until a person's ruling
    (Pipeline.rule), then admitted or refused. A change the consistency stage classified as widening behaviour is
    raised from "gate" to `widen`. Decision: under "person", the automated stages still run, and their refusals are
    advisory (recorded; the person decides), except the static stage's."""
    name = "approval"

    def __init__(self, table: dict | None = None, *, default: str = "gate+person", widen: str = "gate+person"):
        self.table = dict(DEFAULT_APPROVAL if table is None else table)
        for v in [*self.table.values(), default, widen]:
            if v not in LEVELS:
                raise ValueError(f"an approval level is one of {LEVELS}, not {v!r}")
        self.default, self.widen = default, widen

    def level(self, change, verdicts=None):
        k = change.kind
        lv = self.table.get((k, change.source), self.table.get(k, self.default))
        c = (verdicts or {}).get("consistency")
        if lv == "gate" and c is not None and (c.measured or {}).get("direction") == "widens":
            lv = self.widen
        return lv

    def run(self, change, ctx):
        lv = self.level(change, ctx.verdicts)
        if lv == "gate":
            return StageVerdict.admit("", {"level": lv})
        if ctx.ruling is None:
            return StageVerdict.hold(f"approval level {lv!r}: waiting for a person's ruling", {"level": lv})
        ok, by, note = ctx.ruling
        return StageVerdict("admit" if ok else "refuse", f"{'approved' if ok else 'rejected'} by {by}" + (f": {note}" if note else ""),
                       {"level": lv, "by": by})


# ───────────────────────────────────────────────────────────── 8. canaries (wraps solvi.testing.honesty)
def _perfect(m):
    return bool(m.get("n")) and m.get("confident_errors") == 0 and m.get("escalated") == 0


class Canaries:
    """A honesty run of the candidate: run(change, ctx) → metrics (solvi.testing.honesty.metrics's dict, or your own
    numbers in the same direction). Refused when a gated number is worse than `baseline` by more than `tolerance`
    (solvi.testing.honesty.compare); held — a flag, not a success — when `perfect(metrics)` says the score is
    suspiciously perfect (default: every case answered and none wrong, on a set built so that a perfect score proves
    access to the answers: a removed test, a reference the candidate should not see). Ready-made:
    Canaries.honesty(system_of, cases) asks the candidate system the cases (honesty.load_set(path)["cases"])."""
    name = "canaries"

    def __init__(self, run: Callable, *, baseline: dict | None = None, tolerance: float = 0.02,
                 perfect: Callable | None = _perfect):
        self._run, self.baseline, self.tolerance, self.perfect = run, baseline, tolerance, perfect

    def run(self, change, ctx):
        from ...testing.honesty import compare
        m = dict(self._run(change, ctx))
        worse = compare(m, self.baseline, self.tolerance) if self.baseline else []
        meas = {"metrics": m, "worse": worse}
        if worse:
            return StageVerdict.refuse("honesty got worse: " + "; ".join(worse), meas)
        if self.perfect is not None and self.perfect(m):
            return StageVerdict.hold("a suspiciously perfect score on the canaries: a flag for a person, not a success", meas)
        return StageVerdict.admit("", meas)

    @classmethod
    def honesty(cls, system_of: Callable, cases, *, risk: float = 0.10, **kw):
        def run(change, ctx):
            from ...testing import honesty
            return honesty.metrics(honesty.run(system_of(change, ctx), cases, store=False), risk=risk)
        return cls(run, **kw)


# ───────────────────────────────────────────────────────────── 10. refusal regression (new; not switchable)
REFUSAL_WORDS = frozenset({"abstain", "abstained", "abstains", "escalate", "escalated", "hand over", "handover",
                           "hand_over", "handed over", "defer", "deferred", "unknown", "i don't know", "don't know"})


def is_refusal(answer, question=None):
    """Did an answer refuse or hand over? None; a solvi Response (the question's result, or any question when
    `question` is None, has status "abstain"); a result or a dict whose status is "abstain" / "escalate"; a
    Decision whose `escalate` is set; a string such as "abstain", "escalate", "hand over", "unknown" (REFUSAL_WORDS,
    any case). A domain answer like "refuse" (a loan refused) or "no" is an answer, not a refusal: pass your own
    `refused=` to RefusalRegression when your system hands over with another word (e.g. "review")."""
    if answer is None:
        return True
    results = getattr(answer, "results", None)
    if isinstance(results, dict):                                     # a Response
        rs = [results[question]] if question is not None and question in results else list(results.values())
        return any(is_refusal(r) for r in rs)
    if isinstance(answer, dict):
        st = answer.get("status")
        return isinstance(st, str) and st.lower() in ("abstain", "escalate", "escalated", "handover")
    st = getattr(answer, "status", None)
    if isinstance(st, str) and st.lower() in ("abstain", "escalate", "escalated", "handover"):
        return True
    esc = getattr(answer, "escalate", None)
    if isinstance(esc, str) and esc:
        return True
    if isinstance(answer, str):
        return answer.strip().lower() in REFUSAL_WORDS
    return False


def _example_id(x, i):
    if isinstance(x, dict) and "id" in x:
        return str(x["id"])
    return f"#{i}"


class RefusalRegression:
    """Refusal is the first casualty of self-change, so every change whose target can answer — an answerer, a threshold, a head, a gate — is checked
    on a held should-abstain set: outside or unknown inputs the system must refuse or hand over.

    The set: the pipeline's `refusals=` (a list of inputs, or `refusals(change, ctx)` → inputs), the `refusals` kept by
    any other stage (CleanCases(refusals=), Confirmation(refusals=), a stage of yours with a `refusals` attribute),
    and this stage's own `refusals=`. Each input is asked with the answerer in force and with the candidate's —
    `answer_of(change or None, ctx)` (the pipeline's answer_of=, else the target's `answerer(change or None, ctx)`) — and
    `refused(answer)` (default is_refusal; `question=` picks one question of a Response) decides.

    It records: n, refused before / after, the shares, `lost` (inputs refused before and answered after — a change that
    adds refusals elsewhere does not hide them), their first ids, and the count per source of the set. Refused when
    lost > max(count_tolerance, floor(tolerance · n)); default tolerance 0.0 and count_tolerance 0: one input that was
    refused and is now answered refuses the change (raise count_tolerance — e.g. to 1 — for an answerer that flips a
    borderline input between runs). Refused too when the candidate's answerer cannot be had ("cannot measure"). With
    nothing in force before (no answerer for None), nothing can regress: admitted, recorded. With an empty set the
    change is HELD ("no should-abstain set declared for a target that can answer") until a person rules: a check that
    cannot be switched off must measure something, and a vacuous pass would be a switch in disguise — so every
    pipeline whose target can answer declares a set. The set follows the policy version: a document version may
    rightly change what goes to a person, so declare the set of the version in force.

    Not switchable: Pipeline inserts this stage (before Approval) for every change whose target can answer, when it
    is not among the stages; a stage of yours (other tolerances) is used in its place. Its refusal is final under every
    approval level (like the static stage's: a person's "person" level does not make it advisory)."""
    name = "refusal_regression"

    def __init__(self, *, refusals=(), refused: Callable | None = None, question: str | None = None,
                 tolerance: float = 0.0, count_tolerance: int = 0):
        if not 0.0 <= float(tolerance) <= 1.0:
            raise ValueError(f"tolerance is a share in [0, 1], not {tolerance!r}")
        if int(count_tolerance) < 0:
            raise ValueError(f"count_tolerance is a count ≥ 0, not {count_tolerance!r}")
        self.refusals, self.question = refusals, question
        self.refused = refused or (lambda a: is_refusal(a, question))
        self.tolerance, self.count_tolerance = float(tolerance), int(count_tolerance)

    def held_set(self, change, ctx):
        """The should-abstain inputs for this change and how many each source gave → (inputs, {source: count})."""
        out, sources = [], {}

        def take(name, xs):
            xs = list(xs(change, ctx) if callable(xs) else (xs or ()))
            if xs:
                out.extend(xs)
                sources[name] = sources.get(name, 0) + len(xs)
        pipe = getattr(ctx, "pipeline", None)
        if pipe is not None:
            take("pipeline", getattr(pipe, "refusals", ()))
            for st in getattr(pipe, "stages", ()):
                if st is not self and not isinstance(st, RefusalRegression) and getattr(st, "refusals", None):
                    take(st.name, st.refusals)
        take(self.name, self.refusals)
        return out, sources

    def run(self, change, ctx):
        xs, sources = self.held_set(change, ctx)
        base = {"n": len(xs), "sources": sources, "tolerance": self.tolerance, "count_tolerance": self.count_tolerance}
        if not xs:
            return StageVerdict.hold("no should-abstain set declared for a target that can answer: nothing to measure "
                                     "refusal on (declare Pipeline(refusals=), CleanCases / Confirmation(refusals=) or "
                                     "RefusalRegression(refusals=)); a person decides", {**base, "measured": False})
        pipe = ctx.pipeline
        before, after = pipe.answerer(None, ctx, change), pipe.answerer(change, ctx, change)
        if before is None:
            return StageVerdict.admit("nothing in force before the change: no refusal to lose", {**base, "measured": False})
        if after is None:
            return StageVerdict.refuse("refusal regression cannot measure the change: no answerer for the candidate",
                                       {**base, "measured": False})
        rb = [bool(self.refused(before(x))) for x in xs]
        ra = [bool(self.refused(after(x))) for x in xs]
        lost = [_example_id(x, i) for i, (b, a, x) in enumerate(zip(rb, ra, xs)) if b and not a]
        n, nb, na = len(xs), sum(rb), sum(ra)
        allowed = max(self.count_tolerance, int(self.tolerance * n))
        m = {**base, "measured": True, "refused_before": nb, "refused_after": na, "share_before": nb / n,
             "share_after": na / n, "lost": len(lost), "lost_examples": lost[:10], "allowed": allowed}
        where = ", ".join(f"{k} {v}" for k, v in sources.items())
        if len(lost) > allowed:
            return StageVerdict.refuse(
                f"refusal regression on the should-abstain set ({where}; {n} inputs): the change answers {len(lost)} of "
                f"{nb} it refused before (share refused {nb / n:.2f} → {na / n:.2f}; allowed {allowed}): {lost[:5]}", m)
        return StageVerdict.admit(f"refused {nb} → {na} of {n} on the should-abstain set ({where}); lost {len(lost)}", m)


# ───────────────────────────────────────────────────────────── 11. slices, and the original version (new)
SMALL = ("hold", "refuse", "admit")


def _paired(ref, cand):
    """Mean and standard error of the per-case differences candidate − reference (cases assumed independent)."""
    d = [c - r for c, r in zip(cand, ref)]
    n = len(d)
    m = sum(d) / n
    v = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return m, (v / n) ** 0.5


class SliceGuard:
    """Per-slice non-regression, against the version in force AND against the original version (new).

    An aggregate gate passes harm in a small slice under cover of an improvement elsewhere; a gate that compares only
    with the version in force passes a chain of small changes ("salami") each under the tolerance and not significant
    on its own. This stage compares every slice, both ways.

    measure(change, ctx) → {slice: (in_force_losses, candidate_losses)}: per-case losses (lower is better) of the
    version in force and of the candidate on the SAME cases, in the same order (paired).
    original: the original version's losses on those cases, {slice: losses} — a dict, or `original(change, ctx)` → one;
    None (default): pinned from the in-force losses the first time this guard runs (so "original" is the version in
    force then; the pin lives in this object — pass the dict to keep it across processes); False: no comparison with
    the original.

    A slice is refused against a reference when the mean excess d = mean(candidate − reference) is above the tolerance
    (`tol` against the version in force, `cumulative_tol` (default tol) against the original), or when d > 0 is
    significant: one-sided z at `alpha` divided by the number of comparisons made (Bonferroni — valid however the
    slices depend on each other; within a slice the cases are taken as independent for the standard error). An
    excess with no spread at all (every case worse by the same amount) counts as significant. A slice with fewer than
    `min_n` cases, or one the original has and the measurement lacks: on_small = "hold" (default: a person decides) |
    "refuse" | "admit". A slice the original lacks is compared with the version in force only (listed as "new").
    The verdict's numbers: per slice {"n", "in_force": {"d", "se"}, "original": {"d", "se"}}, the z used, "small", "new"."""
    name = "slices"

    def __init__(self, measure: Callable, *, original=None, tol: float = 0.02, cumulative_tol: float | None = None,
                 alpha: float = 0.05, min_n: int = 30, on_small: str = "hold"):
        if on_small not in SMALL:
            raise ValueError(f"on_small is one of {SMALL}, not {on_small!r}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha is in (0, 1), not {alpha!r}")
        for nm, t in (("tol", tol), ("cumulative_tol", cumulative_tol)):
            if t is not None and t < 0:
                raise ValueError(f"{nm} is >= 0, not {t!r}")
        self.measure, self.original = measure, original
        self.tol = float(tol)
        self.cumulative_tol = self.tol if cumulative_tol is None else float(cumulative_tol)
        self.alpha, self.min_n, self.on_small = float(alpha), int(min_n), on_small
        self._pinned = None

    def z(self, m):
        """The one-sided critical z for m comparisons (Bonferroni)."""
        from statistics import NormalDist
        return NormalDist().inv_cdf(1.0 - self.alpha / max(1, m))

    def _original(self, change, ctx, m):
        if self.original is False:
            return None, False
        if self.original is None:
            if self._pinned is None:
                self._pinned = {s: list(now) for s, (now, _) in m.items()}
                return self._pinned, True
            return self._pinned, False
        o = self.original(change, ctx) if callable(self.original) else self.original
        return {s: list(v) for s, v in o.items()}, False

    def run(self, change, ctx):
        m = {s: (list(a), list(b)) for s, (a, b) in self.measure(change, ctx).items()}
        orig, pinned_now = self._original(change, ctx, m)
        for s, (now, cand) in m.items():
            if len(now) != len(cand):
                raise ValueError(f"slice {s!r}: the version in force and the candidate must come from the same cases")
            if orig is not None and s in orig and len(orig[s]) != len(cand):
                raise ValueError(f"slice {s!r}: the original must come from the same cases as the candidate "
                                 f"({len(orig[s])} vs {len(cand)})")
        small = sorted(s for s, (_, cand) in m.items() if len(cand) < self.min_n)
        small += sorted(s for s in (orig or {}) if s not in m)              # unmeasured now: not a measured success
        new = sorted(s for s in m if orig is not None and s not in orig)
        pairs = [(s, ref) for s in sorted(m) if s not in small
                 for ref in ("in_force", "original") if ref == "in_force" or (orig is not None and s in orig)]
        z = self.z(len(pairs))
        rep, bad = {}, []
        for s, ref in pairs:
            now, cand = m[s]
            d, se = _paired(now if ref == "in_force" else orig[s], cand)
            rep.setdefault(s, {"n": len(cand)})[ref] = {"d": round(d, 6), "se": round(se, 6)}
            tol = self.tol if ref == "in_force" else self.cumulative_tol
            if d > tol or (d > 0 and (se == 0 or d - z * se > 0)):
                why = "above the tolerance" if d > tol else "significant"
                bad.append(f"{s}: {d:+.4f} vs {ref.replace('_', ' ')} ({why}, tol {tol})")
        rep.update({"z": round(z, 4), "small": small, "new": new, "original_pinned_now": pinned_now})
        if bad:
            return StageVerdict.refuse("slice regression: " + "; ".join(bad), rep)
        if small and self.on_small != "admit":
            why = f"slices with < {self.min_n} cases or unmeasured: {small}"
            return StageVerdict.hold(why, rep) if self.on_small == "hold" else StageVerdict.refuse(why, rep)
        return StageVerdict.admit(f"no slice regresses ({len(pairs)} comparisons)", rep)


@dataclass
class Context:
    """What a stage sees besides the change: the pipeline, its KnowledgeStore (or None), its journal, the change's
    journal id, the verdicts so far (stage name → StageVerdict), the approval level, and the ruling (ok, by, note) once
    a person gave one."""
    pipeline: Any
    store: Any
    journal: Any
    change_id: str
    verdicts: dict = field(default_factory=dict)
    level: str | None = None
    ruling: tuple | None = None


__all__ = ["Approval", "allowed", "Canaries", "is_refusal", "REFUSAL_WORDS", "RefusalRegression", "catalog_of", "CleanCases", "Confirmation", "Consistency", "Context", "DECISIONS",
           "DEFAULT_APPROVAL", "FreshEnvironments", "gate_of", "gates_of", "hazard_body", "hazard_gate", "Impact",
           "KnownFailures", "SliceGuard",
           "LEVELS", "SOURCES", "Stage", "Static", "StageVerdict"]
