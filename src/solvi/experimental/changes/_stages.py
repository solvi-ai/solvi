"""The stages of the change pipeline: the Stage protocol, its StageVerdict, and the built-in stages.

    class MyStage:                                 # the Stage protocol (like a WriteGate of the knowledge store)
        name = "mine"
        def run(self, change, ctx) -> StageVerdict: ...  # StageVerdict.admit / .refuse / .hold, with what it measured

Built in, in the order the pipeline usually runs them (what each one wraps, ports, or adds):

  KnownFailures      a candidate identical to one rejected before is refused (reads the change journal)          new
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
class KnownFailures:
    """A candidate whose fingerprint (Change.fingerprint: source, artifact, content, code) was rejected before is refused
    with the earlier stage and reason — a search does not walk in circles. Reads the change journal; nothing expires
    (unlike solvi.core.knowledge.FailureMemory, which unblocks a plan after a window)."""
    name = "failure_memory"

    def run(self, change, ctx):
        fp = change.fingerprint()
        for d in ctx.journal.records("rejected"):
            if d.get("fingerprint") == fp:
                return StageVerdict.refuse(f"rejected before by stage {d.get('stage')!r}: {d.get('reason')}",
                                      {"earlier": d.get("change")})
        return StageVerdict.admit()


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
    "system" key; a compiled catalog — solvi.experimental.compile.Compiled — when `sensitive` or `text` is given) is
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
    payload's "system" / "catalog", or — with compiled=True — a compiled catalog's (Compiled.catalog(), loaded through
    the compiler's sandbox)."""
    if isinstance(payload, dict):
        payload = payload.get("system", payload.get("catalog"))
    if payload is None:
        return None
    if callable(getattr(payload, "catalog", None)) and hasattr(payload, "source"):      # a Compiled
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
                                      must fire on what it was learned from, e.g. rewound k steps before the failure)."""
    name = "confirmation"

    def __init__(self, check: Callable):
        self.check = check

    def run(self, change, ctx):
        return StageVerdict.of(_call(self.check, change, ctx))

    @classmethod
    def compiled(cls):
        def check(change, ctx):
            c = change.payload
            ok = bool(getattr(c, "accepted", False))
            return StageVerdict("admit" if ok else "refuse", "" if ok else f"compilation not accepted: {getattr(c, 'reason', '')}",
                           {"fingerprint": getattr(c, "fingerprint", None)})
        return cls(check)

    @classmethod
    def on_evidence(cls, replay: Callable, *, min_share: float = 1.0):
        def check(change, ctx):
            if not change.evidence:
                return StageVerdict.refuse("no evidence to confirm the change on")
            hits = [bool(replay(e, change)) for e in change.evidence]
            share = sum(hits) / len(hits)
            ok = share >= min_share
            return StageVerdict("admit" if ok else "refuse",
                                "" if ok else f"confirmed on {sum(hits)} of {len(hits)} evidence",
                                {"confirmed": sum(hits), "evidence": len(hits), "share": share, "on_evidence": True})
        return cls(check)


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
    gate_of with `parse`); parse: the condition language, body → (check, blocks) (default hazard_gate)."""
    name = "clean_cases"

    def __init__(self, records, *, gate: Callable | None = None, parse: Callable = hazard_gate, max_share: float = 0.0):
        self.records, self.max_share = records, float(max_share)
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
    Impact.of_compiled(storage, old_of)    solvi.experimental.compile.decision_diff(old_of(ctx), change.payload,
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
            from ..compile import decision_diff
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
    episode's info["progress"]).

    The bar, in order: (1) selection — the mean paired gain on `select` is > 0; (2) the candidate's rules never crash
    (a policy that raises); (3) replay — the candidate replays its first `replay` selection worlds with the same
    actions; (4) confirmation — on `confirm`, the bootstrap lower bound of the paired gain is > `min_lower`; (5)
    progress not lower — the upper bound of the paired progress change on `confirm` is ≥ −`progress_tolerance`
    (a rule that makes the agent overcautious is refused). Everything measured is recorded."""
    name = "fresh_environments"

    def __init__(self, env: Callable, agent: Callable, *, select, confirm, score: Callable | None = None,
                 progress: Callable | None = None, max_steps: int = 200, boot: int = 2000, seed: int = 0,
                 min_lower: float = 0.0, progress_tolerance: float = 0.0, replay: int = 2):
        self.env, self.agent, self.select, self.confirm = env, agent, select, confirm
        self.score = score or (lambda r: 0.0 if r["failed"] else 1.0)
        self.progress = progress or (lambda r: r["progress"])
        self.max_steps, self.boot, self.seed = max_steps, boot, seed
        self.min_lower, self.progress_tolerance, self.replay = min_lower, progress_tolerance, replay

    def _seeds(self, s, change, ctx):
        return list(s(change, ctx) if callable(s) else s)

    def _play(self, policy, seeds):
        from ._paired import run_episode
        return [run_episode(self.env(), policy, s, max_steps=self.max_steps) for s in seeds]

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


__all__ = ["Approval", "allowed", "Canaries", "catalog_of", "CleanCases", "Confirmation", "Consistency", "Context", "DECISIONS",
           "DEFAULT_APPROVAL", "FreshEnvironments", "gate_of", "gates_of", "hazard_body", "hazard_gate", "Impact",
           "KnownFailures",
           "LEVELS", "SOURCES", "Stage", "Static", "StageVerdict"]
