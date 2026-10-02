"""Flow execution: computed_state with provenance and a hash chain, answers, hard checks, independent replay."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .core import Decision, Quote, Serial, Unknown, accept, evidence_rows, ground, has_evidence, locate, unwrap, validated
from .provenance import TIMED_OUT, model_info


def _canon(v):
    if isinstance(v, Quote):
        return {"quote": [_canon(v.value), v.start, v.end, v.source]}
    if isinstance(v, Decision):
        return {"decision": [_canon(v.value), _canon(v.probs)]}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (set, frozenset)):          # in a fixed order: repr(set) depends on PYTHONHASHSEED
        return sorted((_canon(x) for x in v), key=_ckey)
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    if v is Unknown:
        return {"not_stated": True}
    if v is MISSING:                              # a failed step: repr(object()) carries a memory address
        return {"missing": True}
    return repr(v)


def _ckey(c):
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def vhash(v) -> str:
    return hashlib.sha256(json.dumps(_canon(v), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def srepr(v):
    """repr with sets in vhash's fixed order (repr(set) depends on PYTHONHASHSEED): for text that is stored, e.g. `why`."""
    if isinstance(v, (set, frozenset)):
        if not v:
            return repr(v)
        items = ", ".join(srepr(x) for x in sorted(v, key=lambda x: _ckey(_canon(x))))
        return "{" + items + "}" if type(v) is set else f"frozenset({{{items}}})"
    if type(v) is list:
        return "[" + ", ".join(srepr(x) for x in v) + "]"
    if type(v) is tuple:
        return "(" + ", ".join(srepr(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if type(v) is dict:
        return "{" + ", ".join(f"{srepr(k)}: {srepr(x)}" for k, x in v.items()) + "}"
    return repr(v)


MISSING = object()

MISMATCH_KINDS = ("integrity", "recompute", "model_changed", "missing_part", "missing_input", "flow", "answer",
                  "not_restored", "error")


class Mismatch(tuple):
    """One replay mismatch: the triple (step, name, reason), as replay has always returned it, plus `.kind`:

        integrity      the hash chain, a record's hash, the input's hash or a recorded input hash does not verify: the data
                       was changed after the run
        recompute      a step does not give the recorded value: its code, a declaration or what it reads changed
        model_changed  the step's model has another fingerprint than the recorded one
        missing_part   the part (or a producer) is no longer in the catalog: renamed or removed
        missing_input  a part now reads an input the trace does not hold
        flow           a planned step is not in the trace
        answer         a stored answer is not the one the trace gives under the same questions and catalog: the answer
                       was changed after the run (replay with the System; a bare Catalog cannot check answers)
        not_restored   a hash or a step does not verify because a value it rests on did not come back from storage as it
                       was: its type is neither one the dump restores (date, Decimal, set, ...) nor declared (an untyped
                       enum or object; an untyped date in a record stored by solvi ≤ 0.7.1) — no verdict on the data
        error          the replay itself failed (the record could not be loaded, or replay raised): no verdict on the data

    It compares, unpacks and serializes as the plain triple."""

    def __new__(cls, step, name, reason, kind="recompute"):
        m = tuple.__new__(cls, (step, name, reason))
        m.kind = kind
        return m

    def __getnewargs__(self):
        return (*self, self.kind)

    def to_dict(self):
        return {"step": self[0], "name": self[1], "reason": self[2], "kind": self.kind}


def mismatch_summary(mismatches, catalog=None):
    """Replay mismatches → {"kinds": {kind: count}, "summary": one line}. "data damaged" when anything of kind integrity;
    else what the mismatches say about a trace whose data is intact: the catalog changed (parts missing, new inputs), the
    model changed, steps do not recompute, or the replay itself failed. `catalog`: the replay's catalog verdict."""
    kinds = {}
    for m in mismatches:
        k = getattr(m, "kind", "recompute")
        kinds[k] = kinds.get(k, 0) + 1
    if not kinds:
        s = "ok"
    elif "integrity" in kinds:
        s = "data damaged: the hash chain or a record does not verify"
    elif "answer" in kinds:
        s = "data damaged: a stored answer is not the one its trace gives"
    elif set(kinds) <= {"error"}:
        s = "replay failed (no verdict on the data)"
    elif "not_restored" in kinds:
        s = "not verified: values stored without their type did not come back as they were (no verdict on the data)"
    elif "missing_part" in kinds:
        s = "data intact, catalog changed (parts missing)"
    elif catalog == "changed" or "missing_input" in kinds:
        s = "data intact, catalog changed"
    elif set(kinds) <= {"model_changed"}:
        s = "data intact, model changed"
    else:
        s = "data intact, steps do not recompute"
    return {"kinds": kinds, "summary": s}


@dataclass
class Record(Serial):
    step: int
    kind: str
    name: str
    inputs: dict                    # name → hash of the input value
    value: Any
    quote: tuple | None = None      # (start, end, source) for extract
    confidence: float = 1.0
    error: str | None = None
    prev: str = ""
    hash: str = ""
    producer: str | None = None     # a fact with alternative producers: the one whose output was used
    tried: list | None = None       # ... and every producer that ran, in order, with its outcome
    provenance: str | None = None   # given | computed | quoted | decided | learned | proposed (see solvi.provenance)
    model: dict | None = None       # model-backed step: {"type", "id", "fp"} of the model that produced the value
    probs: dict | None = None       # a model decision (or a learned answer head): its probabilities
    extra: dict | None = None       # a model decision's details: act probability, expected level, the shared forward pass
    tried_models: dict | None = None  # a fact with alternative producers: the model-backed producers that ran and were
                                    # not used (rejected, or a shadow run): producer → {"type", "id", "fp"} (+ "probs"
                                    # of the decision it proposed); the model that was used is in `model`

    @property
    def default_provenance(self):
        return "quoted" if self.quote else "computed"

    @property
    def origin(self):
        """The provenance kind of this record's value."""
        return self.provenance or self.default_provenance

    def body(self):
        b = {"step": self.step, "kind": self.kind, "name": self.name, "inputs": self.inputs, "value": vhash(self.value),
             "quote": self.quote, "error": self.error, "prev": self.prev}
        if self.tried is not None:                    # single-producer records hash exactly as before
            b["producer"], b["tried"] = self.producer, self.tried
        if self.provenance is not None and self.provenance != self.default_provenance:   # so do plain computed / quoted ones
            b["provenance"] = self.provenance
        if self.model is not None:
            b["model"] = self.model
        if self.probs is not None:
            b["probs"] = {str(k): round(float(v), 6) for k, v in self.probs.items()}
        if self.extra is not None:                    # records without details hash exactly as before
            b["extra"] = self.extra
        if self.tried_models is not None:             # and so do records where no model's output was turned down
            b["tried_models"] = self.tried_models
        return b


@dataclass
class Result(Serial):
    answer: Any
    confidence: float
    why: str
    status: str = "ok"             # ok | forced | abstain
    probs: dict = field(default_factory=dict)
    provenance: str | None = None  # of the answer: computed (rule, hard check), learned (head, learned rule), decided (model rule)
    source: str | None = None      # what produced it: the rule, the hard check, or the head type
    guard: str | None = None       # the safeguard that settled it: hard_check, outside_options, rule_abstained, low_confidence, missing_facts
    repaired: tuple | None = None  # (previous answer, [constraints]) when joint decoding changed it
    kind: str | None = None        # the answer type's kind: yes_no, choice, ordinal, multi, span, rank, estimate
    evidence: list = field(default_factory=list)   # supporting quotes: [Quote(text, start, end, source)], checked in the text
    extra: dict | None = None      # a primitive's details: an estimate's interval and coverage, a ranking's scores

    @property
    def not_stated(self):
        """Is the answer "the text does not state it" (solvi.Unknown)? — a real answer, unlike an abstention."""
        return self.answer is Unknown

    @property
    def span(self):
        """A span answer's Quote (text, start, end, source), else None."""
        return self.evidence[0] if self.kind == "span" and self.evidence and self.answer is not Unknown else None

    @property
    def interval(self):
        """An estimate's interval (lo, hi) holding `coverage` of the probability, else None."""
        x = (self.extra or {}).get("interval")
        return tuple(x) if x is not None else None

    @property
    def scores(self):
        """A ranking's score per option, else None."""
        return (self.extra or {}).get("scores")


@dataclass
class Trace(Serial):
    init_hash: str
    records: list
    init: dict
    skipped: list = field(default_factory=list)     # [(part, why)] steps of the flow that did not need to run
    schedule: list = field(default_factory=list)    # learned order: why each hard check ran when it did (not hashed)
    timings: dict = field(default_factory=dict)     # part (and producer) → run time in ms (not hashed)
    rejected: list = field(default_factory=list)    # [(given fact, why)] inputs that failed System(inputs=...) (not facts)
    fingerprint: dict = field(default_factory=dict)  # which catalog / questions / models decided (System.fingerprint; not hashed)
    early_exit: bool = True                         # False: the whole flow ran although a hard check failed (ask(early_exit=False))

    def explain_order(self):
        """The learned schedule as text: which hard check ran first and why."""
        if not self.schedule:
            return "default order: hard checks and their inputs first, in flow order"
        out = []
        for s in self.schedule:
            out.append(f"{s['rank']}. {s['check']}: P(fail) {s['p_fail']:.2f} × saves {s['saves_ms']:.1f} ms ÷ costs "
                       f"{s['cost_ms']:.1f} ms = {s['score']:.3g}" + (f" [{s['why']}]" if s.get("why") else "")
                       + f" → {s['result']}")
        return "\n".join(out)

    def value(self, name):
        for r in self.records:
            if r.name == name:
                return r.value
        return self.init.get(name, MISSING)

    def replay(self, catalog, flow=None, trust_models=False):
        """Independent replay: recompute every step from its recorded inputs, verify the value, the quote, the error and the
        hash chain, and that the chain starts from the hash of the recorded input. With `flow` (res.flow) it also checks the
        trace is complete: every planned step is either recorded or listed as skipped at run time.

        Model-backed steps (a part with `model=`, a learned rule, an answer head): the recorded model fingerprint is compared
        with the catalog's current model — a difference is a mismatch ("model changed since this decision"). If the model is
        the same and deterministic, the step is re-run and compared like any other. With `trust_models=True`, or when the
        model is not available (no model on the part, `model.available is False`, no head), the model is not re-run: the
        recorded output is verified instead — a quote must be literally at its offsets, a decision among its options.
        Pass the System instead of the catalog to verify answer-head records too (else they are reported as unavailable).

        → {"ok", "steps", "mismatches": [(step, name, reason)], "models": [(step, name, verdict)], "catalog"} with verdicts
        "recomputed", "trusted", "unavailable" or "changed". "catalog": "same", "changed" (with "changed_parts": the parts of
        this trace whose code or declarations differ from when it was recorded) or "unrecorded" — for information: a
        changed part that still re-computes the recorded values is not a mismatch.
        Each mismatch is a Mismatch: the triple, with `.kind` (see Mismatch). When there are mismatches the result also has
        "kinds": {kind: count} and "summary": one line that tells damaged data ("data damaged: ...") from a catalog or a
        model that changed since ("data intact, catalog changed", "data intact, model changed", ...). A part that is no
        longer in the catalog (renamed or removed) is a mismatch of kind missing_part, not an exception; the steps after
        it are still checked.
        Limits: a trace rebuilt honestly from a *different* input is internally consistent — compare `init_hash` with a
        receipt you published elsewhere to catch that."""
        heads = system = None
        if hasattr(catalog, "catalog") and hasattr(catalog, "heads"):          # a System: its catalog and answer heads
            system, heads, catalog = catalog, catalog.heads, catalog.catalog
        vals = dict(self.init)
        prev = self.init_hash
        bad, models = [], []
        lossy = getattr(self, "unrestored", None) or {}   # loaded from JSON: values that did not come back as they were
        if vhash(self.init) != self.init_hash:
            lost = sorted(k for k in lossy if k in self.init)
            if lost:
                bad.append(Mismatch(0, "init", "the recorded input cannot be checked: " + _lost_text(lossy, lost),
                                    "not_restored"))
            else:
                bad.append(Mismatch(0, "init", "init_hash does not match the recorded input", "integrity"))
        n0, cur = len(bad), None
        for r in self.records:
            if lossy and cur is not None:             # what the record before added: explained by a lost value, or not
                _not_restored(bad, n0, lossy, cur)
            if r.prev != prev:                        # a broken link is damage whatever the values are
                bad.append(Mismatch(r.step, r.name, "hash chain broken", "integrity"))
            n0, cur = len(bad), r
            if vhash(r.body()) != r.hash:
                bad.append(Mismatch(r.step, r.name, "record modified after execution", "integrity"))
            prev = r.hash
            if r.kind == "head":
                bad += _replay_head(r, heads, vals, trust_models, models)
                continue
            if r.kind == "textin":                        # ask_text: the entry point and fields read from the text
                from .textin import replay_record
                bad += replay_record(r, self.init)
                if r.model is not None:
                    models.append((r.step, r.name, "trusted"))
                continue
            if r.kind == "plan":                          # a strategist's plan record (solvi.strategy): re-verified, not re-run
                from .strategy import replay_plan
                bad += replay_plan(r, catalog, self.init)
                continue
            part = catalog.rules.get(r.name[7:]) if r.kind == "rule" else catalog.parts.get(r.name)
            if part is None:                              # renamed or removed since the run: a verdict, not a KeyError
                bad.append(Mismatch(r.step, r.name, f"part {r.name} is not in the catalog (renamed or removed)",
                                    "missing_part"))
                vals[r.name] = r.value                    # the steps that read it are still checked, on the recorded value
                continue
            if part.alternatives is not None and r.tried and set(part.inputs) - set(r.inputs):
                from .strategy import narrowed            # a strategist kept only some producers of this fact
                part = narrowed(part, r.tried)
            args = {x: vals.get(x, MISSING) for x in part.inputs}
            lost = [x for x, v in args.items() if v is MISSING]
            if lost:
                if not (r.error or "").startswith("missing inputs") or r.value is not MISSING:
                    bad.append(Mismatch(r.step, r.name, "input " + ", ".join(lost) + " missing from the trace",
                                        "missing_input"))
                vals[r.name] = r.value
                continue
            for x, v in args.items():
                if r.inputs.get(x) != vhash(v):
                    bad.append(Mismatch(r.step, r.name, f"input {x} does not match the recorded one",
                                        "integrity" if x in r.inputs else "recompute"))
            vals[r.name] = r.value
            if r.value is MISSING and _timed_out(r):     # a call that did not finish in time (aask): nothing to re-run
                continue
            if part.alternatives is not None:
                bad += _replay_group(part, r, args, self.init, trust_models, models)
                continue
            if r.model is not None or part.model is not None:
                verdict, extra = _model_check(part.model, r, trust_models)
                bad += extra
                models.append((r.step, r.name, verdict))
                if verdict != "recomputed":
                    bad += _grounded(part, r, self.init) + _checked(part.model, r)
                    continue
            bad += _recompute(part, r, args, self.init, catalog)
        if lossy and cur is not None:
            _not_restored(bad, n0, lossy, cur)
        if flow is not None:
            seen = {r.name for r in self.records} | {n for n, _ in self.skipped}
            ran = {r.name for r in self.records}
            for st in flow.steps:
                if st.part.name not in seen:
                    bad.append(Mismatch(0, st.part.name, "planned step missing from the trace", "flow"))
                elif not self.early_exit and st.part.name not in ran:
                    bad.append(Mismatch(0, st.part.name, "planned step not recorded, though the trace says the whole flow "
                                                         "was computed (early_exit=False)", "flow"))
        bad = [m if isinstance(m, Mismatch) else Mismatch(*m) for m in bad]
        answers = "unchecked"                          # needs the System (its questions) and the response's answers
        if system is not None and getattr(self, "answers", None) is not None:
            answers = "skipped: the trace does not replay"
            if not bad:
                wrong = _answers_differ(self, system, flow)
                bad += wrong
                answers = "differ" if wrong else "same"
        out = {"ok": not bad, "steps": len(self.records), "mismatches": bad, "models": models, "answers": answers}
        out.update(_catalog_verdict(self.fingerprint, catalog))
        if bad:
            out.update(mismatch_summary(bad, out["catalog"]))
        return out


def _lost_text(lossy, names):
    return "; ".join(f"{k} came back from storage as JSON gave it — {lossy[k]}" for k in names)


def _not_restored(bad, n0, lossy, r):
    """The mismatches of one record (bad[n0:]) that a value lost in storage explains — the record's own value, or a
    fact it read — become kind "not_restored": the step cannot be checked, which is no verdict on the data."""
    lost = ([r.name] if r.name in lossy else []) + [x for x in r.inputs if x in lossy]
    if not lost:
        return
    for i in range(n0, len(bad)):
        m = bad[i] if isinstance(bad[i], Mismatch) else Mismatch(*bad[i])
        if m.kind in ("integrity", "recompute"):
            bad[i] = Mismatch(m[0], m[1], f"{m[2]} — not checked: {_lost_text(lossy, lost)}", "not_restored")


def _answers_differ(trace, system, flow):
    """The answers stored with a trace against the ones the trace gives under `system` (System.answers_of) → mismatches.
    Kind "answer" when the system's questions and catalog are the recorded ones — then a stored answer that differs was
    edited after the run; else "recompute": the questions or the catalog changed since. Compared: the answer and its
    status. Not compared: an abstention ask_text made before any flow ran (the entry point escalated)."""
    stored = {q: r for q, r in trace.answers.items() if not (getattr(r, "source", None) == "textin" and r.status == "abstain")}
    gone = [q for q in stored if q not in system.questions]
    try:
        now = system.answers_of(trace, [q for q in stored if q not in gone], flow)
    except Exception as e:  # noqa: BLE001 — a flow that cannot be planned any more, a head that raises
        return [Mismatch(0, "answers", f"the answers could not be derived from the trace: {type(e).__name__}: {str(e)[:120]}",
                         "error")]
    fp = trace.fingerprint or {}
    same_system = bool(fp.get("questions")) and fp.get("questions") == system._questions_fp() \
        and _catalog_verdict(fp, system.catalog)["catalog"] == "same"
    out = [Mismatch(0, f"answer:{q}", "the question is not in the system (renamed or removed)", "missing_part") for q in gone]
    for q, a in now.items():
        was = stored[q]
        if vhash(was.answer) != vhash(a.answer) or was.status != a.status:
            out.append(Mismatch(0, f"answer:{q}", f"stored answer {was.answer!r} [{was.status}] is not the one the trace "
                                f"gives: {a.answer!r} [{a.status}]", "answer" if same_system else "recompute"))
    return out


def _timed_out(r):
    """Did this step fail by a timeout (aask)? A timeout depends on the moment, not on the inputs, so a replay does not
    re-run it (a producer of a fact that timed out: see _replay_group)."""
    return r.tried is None and bool(r.error) and r.error.startswith(TIMED_OUT)


def _catalog_verdict(fp, catalog):
    """The recorded catalog fingerprint against the catalog replaying the trace."""
    if not fp or not fp.get("catalog") or catalog is None or not hasattr(catalog, "parts"):
        return {"catalog": "unrecorded"}
    from .provenance import catalog_fingerprint
    now = catalog_fingerprint(catalog)
    if now["fp"] == fp["catalog"]:
        return {"catalog": "same"}
    was = fp.get("parts") or {}
    return {"catalog": "changed", "changed_parts": sorted(n for n, h in was.items() if now["parts"].get(n) != h)}


def _pass_siblings(catalog, r):
    """A decision recorded as part of a shared forward pass → the decision parts of that pass (by step name), so a replay
    scores it in the same pass; None when the record was not batched, [] when a part of the pass is gone."""
    ps = (r.extra or {}).get("pass") if isinstance(r.extra, dict) else None
    if not isinstance(ps, dict) or not ps.get("with") or catalog is None:
        return None
    out = []
    for n in ps["with"]:
        p = catalog.rules.get(n[7:]) if n.startswith("answer:") else catalog.parts.get(n)
        d = getattr(p.func, "__solvi_decision__", None) if p is not None and p.func is not None else None
        if d is None:
            return []
        out.append(d)
    return out


def _recompute(part, r, args, init, catalog=None):
    """Re-run a part on its recorded inputs and compare its output (value, quote, grounding / type / validation outcome).
    A decision made in a shared forward pass is re-scored in the same pass."""
    if r.error is not None and r.value is not MISSING:           # a failed step never carries a value
        return [(r.step, r.name, f"recorded error {r.error!r}, but the record carries the value {r.value!r}")]
    plain = _plain_args(args, part.inputs)
    why = None
    if part.tin is not None:
        from .typed import typed_in
        plain, why = typed_in(part, plain)
    sibs = _pass_siblings(catalog, r) if r.extra is not None else None
    if sibs == []:
        return [(r.step, r.name, "a decision of its shared pass is no longer in the catalog")]
    if why is None:
        try:
            v = resolved(part.func(**plain) if sibs is None else part.func.in_pass(sibs, plain, r.extra["pass"]["with"]))
        except Exception as e:  # noqa: BLE001
            return [] if r.error is not None else [(r.step, r.name, f"recompute failed: {type(e).__name__}")]
        if isinstance(v, Decision) and isinstance(r.extra, dict):     # several models (solvi.multi): every proposal;
            from .multi import RECORD_KEYS                            # a long text: the sections read (solvi.longdoc)
            for k in RECORD_KEYS + ("long", "memory"):
                if (k in r.extra or k in v.extra) and vhash(r.extra.get(k)) != vhash(v.extra.get(k)):
                    return [(r.step, r.name, f"recorded {k} differ from the recomputed ones")]
        v = locate(part, v, init)
        value, quote, _, _ = unwrap(v)
        why = ground(part, v, init)
        if not why and part.tout is not None:
            from .typed import typed_out
            value, why = typed_out(part, value)
        if not why:
            value, why = check_value(part, value)
        why = why or validated(part, value, plain)
        if not why and r.error is None and (has_evidence(v) or (isinstance(r.extra, dict) and r.extra.get("evidence"))):
            got = evidence_rows(v) if has_evidence(v) else []
            was = [list(x) for x in (r.extra or {}).get("evidence") or ()]
            if got != was:
                return [(r.step, r.name, f"evidence {was} ≠ recomputed {got}")]
    if r.error is not None:
        if why is None:
            return [(r.step, r.name, f"recorded error {r.error!r}, but the step recomputes fine")]
        if why != r.error:
            return [(r.step, r.name, f"recorded error {r.error!r}, but the step recomputes with {why!r}")]
        return []
    if why:
        return [(r.step, r.name, why)]
    if vhash(value) != vhash(r.value):
        return [(r.step, r.name, f"value {r.value!r} ≠ recomputed {value!r}")]
    if quote is not None and r.quote is not None and tuple(quote) != tuple(r.quote):
        return [(r.step, r.name, f"quote {tuple(r.quote)} ≠ recomputed {tuple(quote)}")]
    return []


def _model_check(model, r, trust_models):
    """Compare the recorded model with the catalog's current one → (verdict, mismatches)."""
    from .provenance import fingerprint
    rec = r.model
    if model is None or getattr(model, "available", True) is False:
        return "unavailable", []
    if rec is None:                                   # recorded without model identity (older trace): re-run and compare
        return "recomputed", []
    fp = fingerprint(model)
    if fp != rec.get("fp"):
        return "changed", [Mismatch(r.step, r.name, (f"model changed since this decision: {rec.get('id')} "
                                                     f"#{rec.get('fp')} was used, the catalog now has #{fp}"),
                                    "model_changed")]
    if trust_models or getattr(model, "deterministic", True) is False:
        return "trusted", []
    return "recomputed", []


def _grounded(part, r, init):
    """Without re-running the model: is the recorded output still grounded? A quote literally at its offsets in the recorded
    input, a decision among the part's options (or its recorded probabilities)."""
    from .provenance import NOT_GROUNDED, OUTSIDE_OPTIONS, QUOTE_OUTSIDE, matches
    if r.value is MISSING or r.error is not None:
        return []                                     # a rejected or failed output: the error is what was recorded
    if r.quote:
        s, e, src = r.quote
        text = init.get(src, "")
        if not (isinstance(text, str) and 0 <= s <= e <= len(text)):
            return [(r.step, r.name, QUOTE_OUTSIDE)]
        if (part.strict() or r.model is not None) and r.value is not None and matches(r.value, text[s:e]) is False:
            return [(r.step, r.name, f"{NOT_GROUNDED}: recorded {r.value!r} is not the text at [{s}:{e}]")]
    if isinstance(r.extra, dict) and r.extra.get("evidence"):
        from .core import check_evidence
        why = check_evidence([Quote(t, s, e, src) for s, e, src, t in r.extra["evidence"]], init)
        if why:
            return [(r.step, r.name, f"recorded {why}")]
    opts = part.options if part.options is not None else (list(r.probs) if r.probs else None)
    if part.options is None and isinstance(r.extra, dict) and "interval" in r.extra:
        opts = None                                   # a number decision: a summary of its distribution over the bins
    if opts:
        vs = list(r.value) if isinstance(r.value, (list, tuple, set)) else [r.value]
        if any(x not in opts for x in vs):
            return [(r.step, r.name, f"recorded decision {r.value!r} is {OUTSIDE_OPTIONS} {list(opts)}")]
    return []


def _checked(model, r):
    """Without re-running the models: a model that can check its own record (a combination of models, solvi.multi: the
    answer follows from the recorded proposals by its rule) → mismatches."""
    chk = getattr(model, "check_record", None)
    return [(r.step, r.name, f"recorded proposals: {why}") for why in chk(r)] if callable(chk) else []


def _replay_head(r, heads, vals, trust_models, models):
    """An answer-head record: the answer must follow from the recorded probabilities; with the System, the head's
    fingerprint must match and (unless trusted) its probabilities are recomputed from the recorded facts."""
    from .provenance import fingerprint
    bad = []
    q = r.name[7:]
    for f, h in r.inputs.items():
        if f not in vals or vhash(vals[f]) != h:
            bad.append(Mismatch(r.step, r.name, f"input {f} does not match the recorded one", "integrity"))
    p = r.probs or {}
    if p and r.value not in _head_answers(p):
        bad.append((r.step, r.name, f"answer {r.value!r} does not follow from the recorded probabilities"))
    head = None if heads is None else heads.get(q)
    if head is None:
        models.append((r.step, r.name, "unavailable"))
        return bad
    fp = fingerprint(head)
    if r.model is not None and fp != r.model.get("fp"):
        models.append((r.step, r.name, "changed"))
        return bad + [Mismatch(r.step, r.name, (f"model changed since this decision: {r.model.get('type')} "
                                                f"#{r.model.get('fp')} was used, the system now has #{fp}"),
                               "model_changed")]
    if trust_models:
        models.append((r.step, r.name, "trusted"))
        return bad
    models.append((r.step, r.name, "recomputed"))
    now = head.predict({f: vals.get(f) for f in r.inputs})
    if set(now) != set(p) or any(abs(float(now[k]) - float(p[k])) > 1e-6 for k in p):
        bad.append((r.step, r.name, "head probabilities differ from the recorded ones"))
    return bad


def _head_answers(p):
    """The answers a head can give from its probabilities: the most likely option, the median (ordinal), the options at ≥ 0.5
    (multi-label)."""
    opts = list(p)
    out = [max(p, key=p.get), tuple(o for o in opts if p[o] >= 0.5)]
    acc = 0.0
    for o in opts:
        acc += p[o]
        if acc >= 0.5:
            out.append(o)
            break
    return out


def _plain_args(args, names):
    return {x: (args[x].value if isinstance(args[x], Quote) else args[x]) for x in names}


def _typed_args(part, args, known=None):
    """A typed part's arguments validated / coerced → (args, None) or (args, the rejection reason); untyped: as they are.
    known: fact → the type its value already passed in this run (a typed producer, System(inputs=...)): not re-validated."""
    if part.tin is None:
        return args, None
    from .typed import typed_in
    return typed_in(part, args, known)


def _replay_group(group, r, args, init, trust_models=False, models=None):
    """A fact with alternative producers: recompute with the producer that was used, check it still passes its validator and
    gives the recorded value, and that the producers tried before it are still rejected (shadow runs are not re-checked).
    A model-backed producer is checked like a model-backed part (fingerprint; re-run, or its recorded output verified)."""
    bad = []
    alts = {a.name: a for a in group.alternatives}
    if r.tried is None:
        return [(r.step, r.name, "record of a fact with alternative producers has no 'tried' list")]
    for name, outcome in r.tried:
        if name not in alts:
            bad.append(Mismatch(r.step, r.name, f"producer {name} is not in the catalog (renamed or removed)",
                                "missing_part"))
            continue
        if outcome.startswith(("shadow", TIMED_OUT)):   # shadow runs, and producers that did not finish in time (aask)
            continue
        a = alts[name]
        if name == r.producer and (a.model is not None or r.model is not None):
            verdict, extra = _model_check(a.model, r, trust_models)
            bad += extra
            if models is not None:
                models.append((r.step, f"{r.name} ({name})", verdict))
            if verdict != "recomputed":
                bad += _grounded(a, r, init) + _checked(a.model, r)
                break
        elif name != r.producer and a.model is not None:
            was = (r.tried_models or {}).get(name)    # a rejected model output: its recorded model against today's
            if was is not None and getattr(a.model, "available", True) is not False:
                from .provenance import fingerprint
                fp = fingerprint(a.model)
                if fp != was.get("fp"):
                    bad.append(Mismatch(r.step, r.name, (f"model changed since this decision: {was.get('id')} "
                                                         f"#{was.get('fp')} ran in {name} (rejected), the catalog now has "
                                                         f"#{fp}"), "model_changed"))
                    if models is not None:
                        models.append((r.step, f"{r.name} ({name})", "changed"))
                    continue
            if trust_models or getattr(a.model, "available", True) is False:
                continue                              # a rejected model output is not re-run when models are trusted
        try:
            v, val, (aargs, why) = None, None, _typed_args(a, _plain_args(args, a.inputs))
            ok = False
            if why is None:
                v = resolved(a.func(**aargs))
                g = _plain_args(args, group.inputs)
                ok, why, val = accept(a, v, {**g, **aargs} if a.tin else g, init)
        except Exception as e:  # noqa: BLE001
            v, ok, why = None, False, f"error: {type(e).__name__}"
        if name == r.producer:
            if not ok:
                bad.append((r.step, r.name, f"producer {name} was used, but its output is not accepted on replay ({why})"))
            elif vhash(val) != vhash(r.value):
                bad.append((r.step, r.name, f"value {r.value!r} ≠ recomputed by {name} {v!r}"))
            break
        if ok:
            bad.append((r.step, r.name, f"producer {name} is recorded as rejected, but its output is accepted on replay"))
    else:
        if r.producer is not None:
            bad.append((r.step, r.name, f"producer {r.producer} not in the tried list"))
    return bad


_THREADS = sys.platform != "emscripten"         # no threads in the browser (Pyodide): steps then run one by one


@dataclass
class StepOut:
    value: Any
    quote: tuple | None
    confidence: float
    error: str | None
    hashes: dict
    ms: float = 0.0
    producer: str | None = None
    tried: list | None = None
    alt_ms: dict | None = None        # producer → ms
    outcomes: dict | None = None      # producer → (accepted, value), for the producer policy
    row: dict | None = None           # the features the producer policy saw
    notes: dict | None = None         # producer policy's reasons for the order
    probs: dict | None = None         # a Decision's probabilities
    typed: Any = None                 # the type the value passed (typed parts)
    extra: dict | None = None         # a Decision's details (act probability, expected level, shared pass)
    tried_models: dict | None = None  # producer → model_info (+ probs) of the model-backed producers not used


class HashMemo(dict):
    """vhash of the values of one run, by object: a value read by several steps is hashed once (only non-scalar values;
    each entry keeps its value alive, so an id is not reused within the run)."""

    def __call__(self, v):
        if v is None or type(v) in (str, int, float, bool):
            return vhash(v) if type(v) is not str or len(v) < 256 else self._get(v)
        return self._get(v)

    def _get(self, v):
        e = self.get(id(v))
        if e is None:
            e = self[id(v)] = (v, vhash(v))
        return e[1]


def _extra(v):
    x = v.extra or None if isinstance(v, Decision) else None
    if has_evidence(v):                               # located supporting quotes (outputs without evidence: as before)
        x = {**(x or {}), "evidence": evidence_rows(v)}
    return x


class PartTimeout(Exception):
    """A part did not finish within its timeout (aask): the step fails with this reason, like any other error."""


def run_sync(aw):
    """Wait for an awaitable from sync code (an `async def` part called by `ask`, a replay, `facts_for`): in a new event
    loop — on a worker thread when this thread already runs one (ask called from async code)."""
    async def main():
        return await aw
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, main()).result()


def resolved(v):
    """A part's output: awaited when it is an awaitable (an `async def` part called from sync code)."""
    return run_sync(v) if inspect.isawaitable(v) else v


def is_async_func(f):
    """Is f an `async def` function (or an object with an async __call__, or a wrapper of one)?"""
    if f is None:
        return False
    f = inspect.unwrap(f) if callable(f) else f
    return inspect.iscoroutinefunction(f) or inspect.iscoroutinefunction(getattr(f, "__call__", None))  # noqa: B004 — an async __call__


def async_parts(catalog):
    """The parts (producers, rules) of a catalog that aask awaits: `async def` functions and parts marked blocking=True."""
    out = []
    for p in list(catalog.parts.values()) + list(catalog.rules.values()):
        for a in (p.alternatives if p.alternatives is not None else [p]):
            if a.blocking or is_async_func(a.func):
                out.append(a.name)
    return out


def _drive(g):
    """Run a step generator (see _step) from sync code: each call of user code is made here, an awaitable it returns is
    awaited in an event loop of its own."""
    try:
        req = next(g)
        while True:
            try:
                v = resolved(req[1]())
            except Exception as e:  # noqa: BLE001 — the step decides what an exception means
                req = g.throw(e)
            else:
                req = g.send(v)
    except StopIteration as s:
        return s.value


async def _acall(part, call, timeout):
    """One call of user code under aask: an `async def` part is awaited, a part marked blocking runs in a worker thread,
    a plain sync part runs inline; a timeout (the part's, else the run's) raises PartTimeout."""
    t = part.timeout if part.timeout is not None else timeout
    if part.blocking and _THREADS:
        aw = asyncio.to_thread(call)
    else:
        v = call()
        if not inspect.isawaitable(v):
            return v
        aw = v
    if t is None:
        v = await aw
    else:
        try:
            v = await asyncio.wait_for(aw, t)
        except asyncio.TimeoutError:
            raise PartTimeout(f"{TIMED_OUT} after {t:g} s") from None
    return await v if inspect.isawaitable(v) else v


async def _adrive(g, timeout):
    """Run a step generator under aask (see _acall)."""
    try:
        req = next(g)
        while True:
            try:
                v = await _acall(req[0], req[1], timeout)
            except Exception as e:  # noqa: BLE001 — CancelledError is not an Exception: a cancelled step stops here
                req = g.throw(e)
            else:
                req = g.send(v)
    except StopIteration as s:
        return s.value


def check_value(p, value):
    """A check answers True or False → (value, why not). numpy's bool is taken as a bool; anything else — 0, None from a
    forgotten return, a list, a string — rejects the step, so the fact is missing and whatever depends on it abstains: a
    check that failed with a falsy non-bool would otherwise count as passed. A typed check (`-> bool`) is validated by its
    type instead."""
    if p.kind != "check" or p.tout is not None or isinstance(value, bool):
        return value, None
    if type(value).__module__ == "numpy" and type(value).__name__ in ("bool", "bool_"):    # amount <= limit over numpy
        return bool(value), None                                                          # numbers; numpy is not imported
    return value, f"a check returns True or False, not {type(value).__name__} ({str(value)[:40]!r}); declare it `-> bool`"


def _step(p, vals, init_state, policy=None, costs=None, known=None, memo=None, batch=None):
    """Evaluate one part on the current facts → StepOut. A generator: every call of user code is yielded as (part, call) and
    made by the driver (_drive: sync, _adrive: async), which sends back its value or throws its exception in. batch: (step
    names, decision parts) of a shared forward pass this decision part belongs to (see Flow.batches)."""
    t0 = time.perf_counter()
    args = {x: vals.get(x, MISSING) for x in p.inputs}
    h = memo if memo is not None else vhash
    hashes = {x: h(v) for x, v in args.items() if v is not MISSING}
    if any(v is MISSING for v in args.values()):
        err = "missing inputs: " + ", ".join(x for x, v in args.items() if v is MISSING)
        return StepOut(MISSING, None, 1.0, err, hashes, 0.0, tried=[] if p.alternatives is not None else None)
    if p.alternatives is not None:
        out = yield from _group(p, args, init_state, policy, costs, known)
        out.hashes = hashes
        out.ms = (time.perf_counter() - t0) * 1000
        return out
    if p.tin is not None:                             # typed arguments: validated / coerced, a failure rejects the step
        args, why = _typed_args(p, args, known)
        if why:
            return StepOut(MISSING, None, 1.0, why, hashes, (time.perf_counter() - t0) * 1000)
    try:
        v = yield (p, (lambda: p.func(**args)) if batch is None else (lambda: p.func.in_pass(batch[1], args, batch[0])))
    except PartTimeout as e:
        return StepOut(MISSING, None, 1.0, str(e), hashes, (time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        return StepOut(MISSING, None, 1.0, f"{type(e).__name__}: {str(e)[:120]}", hashes, (time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    v = locate(p, v, init_state)
    value, quote, conf, probs = unwrap(v)
    why = ground(p, v, init_state)
    if not why and p.tout is not None:                # typed output: validated / coerced (a closed set: outside the options)
        from .typed import typed_out
        value, why = typed_out(p, value)
    if not why:
        value, why = check_value(p, value)
    why = why or validated(p, value, args)
    if why:                                           # rejected: the fact is missing, the claim stays visible in the error
        return StepOut(MISSING, quote, conf, why, hashes, ms, probs=probs, extra=_extra(v))
    return StepOut(value, quote, conf, None, hashes, ms, probs=probs, typed=p.tout.type if p.tout is not None else None,
                   extra=_extra(v))


def _run_step(p, vals, init_state, policy=None, costs=None, known=None, memo=None, batch=None):
    """_step, driven from sync code."""
    return _drive(_step(p, vals, init_state, policy, costs, known, memo, batch))


def group_features(group, args):
    from .learned import scalar_row
    plain = _plain_args(args, group.inputs)
    if group.features is not None:
        names = list(inspect.signature(group.features).parameters)
        try:
            return scalar_row(group.features(**{x: plain[x] for x in names}))
        except Exception:  # noqa: BLE001
            return {}
    return scalar_row(plain)


def _group(group, args, init_state, policy, costs, known=None):
    """Alternative producers of one fact: try them in the policy's order (declaration order without a policy); the first
    output that passes its producer's checks is used. A shadow run (policy exploration) runs the rest too, for learning only.
    A generator of calls, like _step: a producer that times out is rejected and the next one is tried."""
    row, notes, shadow = None, None, False
    order = list(group.alternatives)
    if policy is not None:
        row = group_features(group, args)
        order, notes, shadow = policy.plan(group, row, costs)
    tried, outcomes, alt_ms, used, tried_models = [], {}, {}, None, {}
    plain = _plain_args(args, group.inputs)
    for a in order:
        if used is not None and not shadow:
            break
        t = time.perf_counter()
        try:
            v, val, ok = None, None, False
            aargs, why = _typed_args(a, {x: args[x] for x in a.inputs}, known)
            if why is None:
                v = locate(a, (yield (a, lambda: a.func(**aargs))), init_state)
                ok, why, val = accept(a, v, {**plain, **aargs} if a.tin else plain, init_state)
        except PartTimeout as e:
            v, ok, why = None, False, str(e)
        except Exception as e:  # noqa: BLE001
            v, ok, why = None, False, f"error: {type(e).__name__}: {str(e)[:80]}"
        alt_ms[a.name] = (time.perf_counter() - t) * 1000
        outcomes[a.name] = (ok, v)
        if a.model is not None and not (ok and used is None):     # a model that ran and is not the one used: its
            tm = dict(model_info(a.model))            # identity is recorded too, else a decision made by a later producer
            if isinstance(v, Decision) and v.probs:   # leaves no trace of the model that was asked first
                tm["probs"] = {str(k): round(float(p), 6) for k, p in v.probs.items()}
            tried_models[a.name] = tm
        if used is None:
            tried.append([a.name, why])
            if ok:
                used = (a, v, val)
        else:
            same = ok and vhash(val) == vhash(used[2])
            tried.append([a.name, "shadow: " + ("agrees" if same else ("differs" if ok else why))])
    if used is None:
        return StepOut(MISSING, None, 1.0, "no producer accepted: " + "; ".join(f"{n} {w}" for n, w in tried), {},
                       tried=tried, alt_ms=alt_ms, outcomes=outcomes, row=row, notes=notes, tried_models=tried_models or None)
    a, v, value = used
    _, quote, conf, probs = unwrap(v)
    return StepOut(value, quote, conf, None, {}, producer=a.name, tried=tried, alt_ms=alt_ms, outcomes=outcomes, row=row,
                   notes=notes, probs=probs, typed=a.tout.type if a.tout is not None else None, extra=_extra(v),
                   tried_models=tried_models or None)


def provenance_of(part, out):
    """The provenance kind of a part's output: declared, else from what it returned and whether a model is behind it."""
    if part.provenance is not None:
        return part.provenance
    if out.quote is not None:
        return "quoted"
    if out.probs is not None:
        return "decided"
    if part.model is not None:
        if part.kind == "extract":
            return "quoted"
        if part.kind == "rule" or part.options is not None:
            return "decided"
        return "learned"
    return "computed"


def execute(catalog, flow, init_state, workers=1, early_exit=True, order=None, costs=None, policy=None, known=None):
    """Run the flow. Hard checks and what they depend on run first; a failed hard check settles the questions whose flow contains
    it, and the steps only those questions needed are skipped (early exit). Steps that do not depend on each other run in
    parallel when workers > 1 (threads: suits I/O-bound parts such as API calls and model inference). Records are always written
    in flow order, so the hash chain and replay do not depend on scheduling. An `async def` part is awaited in an event loop
    of its own (aexecute runs such catalogs concurrently).

    order: None — all hard checks (and their inputs) first, together. An object with `p_fail(check, row)` and `row(vals,
    init_keys)` (solvi.learned.OrderModel, or an oracle) — hard checks one at a time, most expected saving first, stopping
    as soon as the failed ones settle every question they govern; answers are the same as with the default order.
    costs: a CostBook (ms per part) for the learned order and the producer policy. policy: a ProducerPolicy for facts with
    alternative producers (without one they are tried in declaration order). known: given fact → the type its value was
    already validated against (System(inputs=...)), so typed parts reading it with that type skip re-validation.
    early_exit=False: every step of the flow runs, whatever the hard checks say (the answers are the same: a failed hard
    check still decides); the trace records it (`trace.early_exit`)."""
    run = _Run(catalog, flow, init_state, early_exit, order, costs, policy, known)
    for idxs in run.phases():
        run.run_sync(idxs, workers)
    return run.finish()


async def aexecute(catalog, flow, init_state, early_exit=True, order=None, costs=None, policy=None, known=None,
                   timeout=None, speculate=False):
    """execute, on an event loop: `async def` parts are awaited, parts marked blocking run in worker threads, plain sync
    parts inline; steps whose inputs are ready run concurrently. timeout: seconds per call of a part without its own
    `timeout=` (None: no limit) — a step that does not finish in time fails with "timed out after ... s".
    speculate=False: the phases of execute (hard checks and their inputs first, then what the open questions need), so no
    step starts that execute would not run. speculate=True (default order only): every step starts as soon as its inputs
    are ready; a failed hard check cancels the pending steps that only the questions it settles needed, and steps that
    finished anyway are dropped. Either way the records, values and hashes are those of execute (a timeout aside)."""
    run = _Run(catalog, flow, init_state, early_exit, order, costs, policy, known)
    run.timeout = timeout
    if speculate and early_exit and run.hard and order is None:
        await run.speculate()
    else:
        for idxs in run.phases():
            await run.run_async(idxs)
    return run.finish()


class _Run:
    """One execution of a flow: the plan of phases and the state the sync and async drivers share."""

    def __init__(self, catalog, flow, init_state, early_exit=True, order=None, costs=None, policy=None, known=None):
        from .learned import CostBook
        self.catalog, self.flow, self.init_state = catalog, flow, init_state
        self.early_exit, self.order, self.policy = early_exit, order, policy
        self.costs = costs if costs is not None else CostBook()
        self.vals = dict(init_state)
        self.steps = steps = flow.steps
        self.names = names = [st.part.name for st in steps]
        self.index = index = {n: i for i, n in enumerate(names)}
        self.done = {}
        self.live = set(flow.per_question)
        self.settled_by = {}
        self.schedule = []
        self.timeout = None
        self.known = dict(known) if known else ({} if (catalog.readers or catalog.types) else None)
        self.memo = HashMemo()
        self.batch_of = {}                            # step name → (names, decision parts) of its shared forward pass
        for group in getattr(flow, "batches", None) or ():
            group = [n for n in group if n in index]
            if len(group) > 1:
                b = (group, [steps[index[n]].part.func for n in group])
                for n in group:
                    self.batch_of[n] = b
        self.hard = [i for i, st in enumerate(steps) if st.part.kind == "check" and st.part.hard]
        self.gov = {}
        for i in self.hard:
            part = steps[i].part
            checkpoint_of = {r.split(" ", 1)[1] for r in steps[i].reasons if r.startswith("checkpoint ")}
            self.gov[i] = {q for q in flow.per_question if names[i] in flow.per_question[q]
                           and (not part.then or q in part.then or q in checkpoint_of)}

    def step(self, i, vals):
        return _step(self.steps[i].part, vals, self.init_state, self.policy, self.costs, self.known, self.memo,
                     self.batch_of.get(self.names[i]) if self.batch_of else None)

    def keep(self, i):
        self.vals[self.names[i]] = self.done[i].value
        if self.known is not None and self.done[i].typed is not None:
            self.known[self.names[i]] = self.done[i].typed

    def ancestors(self, i, acc=None):
        acc = set() if acc is None else acc
        if i not in acc and i not in self.done:
            acc.add(i)
            for x in self.steps[i].part.inputs:
                if x in self.index:
                    self.ancestors(self.index[x], acc)
        return acc

    def needed(self, i, live=None):
        live = self.live if live is None else live
        p = self.steps[i].part
        if not self.early_exit:
            return True
        if p.kind == "rule":
            return p.question in live
        return any(p.name in self.flow.per_question.get(q, ()) for q in live)

    def first(self):
        """The hard checks and every step they read (the first phase of the default order)."""
        acc = set()
        for i in self.hard:
            self.ancestors(i, acc)
        return acc

    def settle(self):
        """Default order, every hard check evaluated: the questions settled by the failed ones (the first in flow order)."""
        for i in self.hard:
            if self.done[i].value is False:
                for q in [q for q in self.live if q in self.gov[i]]:
                    self.live.discard(q)
                    self.settled_by[q] = self.names[i]

    def phases(self):
        """The sets of steps to run, one after another (a driver runs each and fills `done` before asking for the next)."""
        if self.early_exit and self.hard and self.order is None:
            yield sorted(self.first())
            self.settle()
        elif self.early_exit and self.hard:
            yield from _learned_hard_checks(self)
        yield [i for i in range(len(self.steps)) if i not in self.done and self.needed(i)]

    def _got(self, i, out):
        self.done[i] = out
        if out.value is not MISSING:
            self.keep(i)

    def run_sync(self, idxs, workers=1):
        if workers <= 1 or len(idxs) <= 1 or not _THREADS:
            for i in idxs:                                  # idxs are in topological (flow) order
                self._got(i, _drive(self.step(i, self.vals)))
            return
        # dependency-driven: a step starts as soon as the steps it reads have finished
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        todo = set(idxs)
        deps = {i: {self.index[x] for x in self.steps[i].part.inputs if x in self.index and self.index[x] in todo}
                for i in idxs}
        running = {}
        with ThreadPoolExecutor(workers) as ex:
            while todo or running:
                for i in sorted(i for i in todo if not deps[i] & (todo | set(running.values()))):
                    todo.discard(i)
                    running[ex.submit(lambda i, v: _drive(self.step(i, v)), i, dict(self.vals))] = i
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in finished:
                    i = running.pop(fut)
                    self._got(i, fut.result())

    async def _astep(self, i):
        return await _adrive(self.step(i, self.vals), self.timeout)

    async def run_async(self, idxs):
        """Run steps as asyncio tasks, each as soon as the steps it reads have finished (started in flow order)."""
        todo = set(idxs)
        deps = {i: {self.index[x] for x in self.steps[i].part.inputs if x in self.index and self.index[x] in todo}
                for i in idxs}
        await self._tasks(todo, deps, lambda i: True)

    async def speculate(self):
        """Every step the default order could run starts as soon as its inputs are ready; a failed hard check cancels the
        pending steps no open question needs. Then what the default order would not have run is dropped."""
        first = self.first()
        live = set(self.live)
        hard = set(self.hard)

        def want(i):
            return i in first or self.needed(i, live)

        def seen(i, out):
            if i in hard and out.value is False:
                live.difference_update(self.gov[i])
        todo = {i for i in range(len(self.steps)) if want(i)}
        deps = {i: {self.index[x] for x in self.steps[i].part.inputs if x in self.index} for i in todo}
        await self._tasks(todo, deps, want, seen)
        self.settle()
        for i in [i for i in self.done if i not in first and not self.needed(i)]:
            del self.done[i]                          # finished before the check that makes it unnecessary: not recorded
            self.vals.pop(self.names[i], None)
            if self.known is not None:
                self.known.pop(self.names[i], None)

    async def _tasks(self, todo, deps, want, seen=None):
        running = {}
        try:
            while todo or running:
                for t, i in list(running.items()):
                    if not want(i):                   # no open question needs it any more: stop the call
                        t.cancel()
                        del running[t]
                todo.difference_update([i for i in todo if not want(i)])
                busy = todo | set(running.values())
                for i in sorted(i for i in todo if not deps[i] & busy):
                    todo.discard(i)
                    running[asyncio.ensure_future(self._astep(i))] = i
                if not running:
                    break
                finished, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
                for t in sorted(finished, key=lambda t: running[t]):
                    i = running.pop(t)
                    self._got(i, t.result())
                    if seen is not None:
                        seen(i, self.done[i])
        finally:
            for t in running:                         # aask itself cancelled (or failed): no call keeps running
                t.cancel()

    def finish(self):
        """The records in flow order (hash-chained), the skipped steps and why → (trace, values)."""
        steps, done, catalog, flow = self.steps, self.done, self.catalog, self.flow
        init_hash = vhash(self.init_state)
        self.memo.clear()
        prev, recs, skipped, timings = init_hash, [], [], {}
        for i, st in enumerate(steps, 1):
            if i - 1 not in done:
                by = sorted({self.settled_by[q] for q in self.settled_by if st.part.name in flow.per_question.get(q, ()) or
                             (st.part.kind == "rule" and st.part.question == q)})
                skipped.append((st.part.name, "not needed: hard check " + ", ".join(by) + " failed" if by else "not needed"))
                continue
            o = done[i - 1]
            timings[st.part.name] = o.ms
            for a, ms in (o.alt_ms or {}).items():
                timings[a] = ms
            part = st.part
            if o.producer is not None:
                part = catalog.alternative(st.part.name, o.producer)
            rec = Record(step=i, kind=part.kind, name=st.part.name, inputs=o.hashes, value=o.value, quote=o.quote,
                         confidence=o.confidence, error=o.error, prev=prev, producer=o.producer, tried=o.tried,
                         provenance=provenance_of(part, o), model=model_info(part.model) if part.model is not None else None,
                         probs=o.probs, extra=o.extra, tried_models=o.tried_models)
            rec.hash = vhash(rec.body())
            prev = rec.hash
            recs.append(rec)
        final = {k: v for k, v in self.vals.items()}
        trace = Trace(init_hash, recs, dict(self.init_state), skipped, self.schedule, timings,
                      early_exit=bool(self.early_exit))
        trace._outs = {self.names[i]: o for i, o in done.items() if o.outcomes is not None}   # for the producer policy
        return trace, final


def _learned_hard_checks(run):
    """Hard checks one at a time, most expected saving first. A question is settled by a failed hard check only when every
    hard check governing it and declared earlier in the catalog has been evaluated and passed — so the check that decides is
    the same as with the default order (the first declared failed one). Stops when no unevaluated hard check is in the flow
    of a question that is still open. A generator of phases (see _Run.phases)."""
    catalog, flow, init_state, order, costs = run.catalog, run.flow, run.init_state, run.order, run.costs
    steps, names, index, done, vals = run.steps, run.names, run.index, run.done, run.vals
    live, settled_by, hard, gov, ancestors, schedule = run.live, run.settled_by, run.hard, run.gov, run.ancestors, run.schedule
    pos = {n: k for k, n in enumerate(catalog.parts)}
    need_by = {}
    for j, st in enumerate(steps):
        p = st.part
        need_by[j] = {p.question} if p.kind == "rule" else {q for q, fs in flow.per_question.items() if p.name in fs}
    governing = {q: sorted((i for i in hard if q in gov[i]), key=lambda i: pos.get(names[i], 0)) for q in flow.per_question}
    init_keys = list(init_state)
    feats = [f for f in getattr(order, "features", []) or [] if f in index]
    if feats:                                           # cheap computed facts the order model reads: compute them first
        acc = set()
        for f in feats:
            ancestors(index[f], acc)
        yield sorted(acc)

    def settle():
        for q in list(live):
            for i in governing.get(q, ()):
                if i not in done:
                    break
                if done[i].value is False:
                    live.discard(q)
                    settled_by[q] = names[i]
                    break

    def cost(js):
        return sum(costs.get(steps[j].part) for j in js)
    rank = 0
    settle()
    while True:
        cand = [i for i in hard if i not in done and any(q in live for q in need_by[i])]
        if not cand:
            break
        row = order.row(vals, init_keys)
        pending = [j for j in range(len(steps)) if j not in done]
        # questions already decided by a failed check that still wait for earlier-declared checks: evaluating those
        # blockers settles the question whatever they return, so they are worth the saving they unlock
        blocked = {}                                    # question → the unevaluated earlier-declared checks it waits for
        for q in live:
            gs = governing.get(q, ())
            hit = next((k for k, i in enumerate(gs) if i in done and done[i].value is False), None)
            if hit is not None:
                blocked[q] = [i for i in gs[:hit] if i not in done]
        best = None
        for i in cand:
            sure_qs = {q for q, bs in blocked.items() if i in bs}
            if sure_qs:
                acc = set()
                for q in sure_qs:
                    for k in blocked[q]:
                        ancestors(k, acc)
                sure_saves = cost([j for j in pending if j not in acc and need_by[j] & live and need_by[j] & live <= sure_qs])
                sure = sure_saves / max(cost(acc), 1e-3)
            else:
                sure = 0.0
            anc = ancestors(i)
            qs = {q for q in gov[i] if q in live}
            pre = set()                                 # earlier-declared checks that must pass before i can decide
            for q in qs:
                for k in governing[q]:
                    if k == i:
                        break
                    ancestors(k, pre)
            c_eval = cost(anc)
            c_settle = cost(pre - anc)
            saves = cost([j for j in pending if j not in anc and need_by[j] & live and need_by[j] & live <= qs])
            p = float(order.p_fail(names[i], row))
            score = p * saves / max(c_eval + c_settle, 1e-3)
            why = ""
            if sure > score:
                score, why = sure, "unblocks " + ", ".join(sorted(sure_qs)) + " (already decided by a failed check)"
            key = (-score, index[names[i]])
            if best is None or key < best[0]:
                best = (key, i, anc, p, saves, c_eval + c_settle, score, why)
        _, i, anc, p, saves, c, score, why = best
        yield sorted(anc)
        rank += 1
        failed = done[i].value is False
        settle()
        schedule.append({"rank": rank, "check": names[i], "p_fail": p, "saves_ms": saves, "cost_ms": c, "score": score,
                         "why": why, "result": ("failed" if failed else "passed") + (
                             f"; settles {', '.join(sorted(q for q, by in settled_by.items() if by == names[i]))}"
                             if failed and any(by == names[i] for by in settled_by.values()) else "")})


def path_confidence(catalog, trace, facts):
    """A fact's confidence is the minimum confidence of the extractions it depends on. A fact with alternative producers
    depends on what the producer that was used reads (the record's `producer`), not on the inputs of producers that
    did not give the value."""
    by = {r.name: r for r in trace.records}
    memo = {}

    def conf(f):
        if f in memo:
            return memo[f]
        r = by.get(f)
        if r is None:
            memo[f] = 1.0
            return 1.0
        memo[f] = 1.0                     # being walked: a fact is never an input of itself (guards a ring of facts)
        part = catalog.parts.get(f)
        c = r.confidence                  # 1.0 for plain computations; a quote's or a decision's confidence otherwise
        if part is not None:
            ins = part.inputs
            if r.producer is not None and part.alternatives is not None:
                ins = next((a.inputs for a in part.alternatives if a.name == r.producer), ins)
            for x in ins:
                c = min(c, conf(x))
        memo[f] = c
        return c
    return min([conf(f) for f in facts] or [1.0])


def now_ms():
    return time.perf_counter() * 1000
