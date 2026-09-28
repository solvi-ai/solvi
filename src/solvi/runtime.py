"""Flow execution: computed_state with provenance and a hash chain, answers, hard checks, independent replay."""
from __future__ import annotations

import hashlib
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .core import Decision, Quote, Serial, Unknown, accept, evidence_rows, ground, has_evidence, locate, unwrap, validated
from .provenance import model_info


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
    return repr(v)


def _ckey(c):
    return json.dumps(c, ensure_ascii=False, sort_keys=True)


def vhash(v) -> str:
    return hashlib.sha256(json.dumps(_canon(v), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


MISSING = object()


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

        → {"ok", "steps", "mismatches": [(step, name, reason)], "models": [(step, name, verdict)]} with verdicts
        "recomputed", "trusted", "unavailable" or "changed".
        Limits: a trace rebuilt honestly from a *different* input is internally consistent — compare `init_hash` with a
        receipt you published elsewhere to catch that."""
        heads = None
        if hasattr(catalog, "catalog") and hasattr(catalog, "heads"):          # a System: its catalog and answer heads
            heads, catalog = catalog.heads, catalog.catalog
        vals = dict(self.init)
        prev = self.init_hash
        bad, models = [], []
        if vhash(self.init) != self.init_hash:
            bad.append((0, "init", "init_hash does not match the recorded input"))
        for r in self.records:
            if r.prev != prev:
                bad.append((r.step, r.name, "hash chain broken"))
            if vhash(r.body()) != r.hash:
                bad.append((r.step, r.name, "record modified after execution"))
            prev = r.hash
            if r.kind == "head":
                bad += _replay_head(r, heads, vals, trust_models, models)
                continue
            if r.kind == "plan":                          # a strategist's plan record (solvi.strategy): re-verified, not re-run
                from .strategy import replay_plan
                bad += replay_plan(r, catalog, self.init)
                continue
            part = catalog.rules[r.name[7:]] if r.kind == "rule" else catalog.parts[r.name]
            if part.alternatives is not None and r.tried and set(part.inputs) - set(r.inputs):
                from .strategy import narrowed            # a strategist kept only some producers of this fact
                part = narrowed(part, r.tried)
            args = {x: vals.get(x, MISSING) for x in part.inputs}
            lost = [x for x, v in args.items() if v is MISSING]
            if lost:
                if not (r.error or "").startswith("missing inputs") or r.value is not MISSING:
                    bad.append((r.step, r.name, "input " + ", ".join(lost) + " missing from the trace"))
                vals[r.name] = r.value
                continue
            for x, v in args.items():
                if r.inputs.get(x) != vhash(v):
                    bad.append((r.step, r.name, f"input {x} does not match the recorded one"))
            vals[r.name] = r.value
            if part.alternatives is not None:
                bad += _replay_group(part, r, args, self.init, trust_models, models)
                continue
            if r.model is not None or part.model is not None:
                verdict, extra = _model_check(part.model, r, trust_models)
                bad += extra
                models.append((r.step, r.name, verdict))
                if verdict != "recomputed":
                    bad += _grounded(part, r, self.init)
                    continue
            bad += _recompute(part, r, args, self.init, catalog)
        if flow is not None:
            seen = {r.name for r in self.records} | {n for n, _ in self.skipped}
            for st in flow.steps:
                if st.part.name not in seen:
                    bad.append((0, st.part.name, "planned step missing from the trace"))
        return {"ok": not bad, "steps": len(self.records), "mismatches": bad, "models": models}


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
            v = part.func(**plain) if sibs is None else part.func.in_pass(sibs, plain, r.extra["pass"]["with"])
        except Exception as e:  # noqa: BLE001
            return [] if r.error is not None else [(r.step, r.name, f"recompute failed: {type(e).__name__}")]
        v = locate(part, v, init)
        value, quote, _, _ = unwrap(v)
        why = ground(part, v, init)
        if not why and part.tout is not None:
            from .typed import typed_out
            value, why = typed_out(part, value)
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
        return "changed", [(r.step, r.name, (f"model changed since this decision: {rec.get('id')} #{rec.get('fp')} "
                                             f"was used, the catalog now has #{fp}"))]
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


def _replay_head(r, heads, vals, trust_models, models):
    """An answer-head record: the answer must follow from the recorded probabilities; with the System, the head's
    fingerprint must match and (unless trusted) its probabilities are recomputed from the recorded facts."""
    from .provenance import fingerprint
    bad = []
    q = r.name[7:]
    for f, h in r.inputs.items():
        if f not in vals or vhash(vals[f]) != h:
            bad.append((r.step, r.name, f"input {f} does not match the recorded one"))
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
        return bad + [(r.step, r.name, (f"model changed since this decision: {r.model.get('type')} #{r.model.get('fp')} "
                                        f"was used, the system now has #{fp}"))]
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
            bad.append((r.step, r.name, f"unknown producer {name}"))
            continue
        if outcome.startswith("shadow"):
            continue
        a = alts[name]
        if name == r.producer and (a.model is not None or r.model is not None):
            verdict, extra = _model_check(a.model, r, trust_models)
            bad += extra
            if models is not None:
                models.append((r.step, f"{r.name} ({name})", verdict))
            if verdict != "recomputed":
                bad += _grounded(a, r, init)
                break
        elif name != r.producer and a.model is not None and (trust_models or getattr(a.model, "available", True) is False):
            continue                                  # a rejected model output is not re-run when models are trusted
        try:
            v, val, (aargs, why) = None, None, _typed_args(a, _plain_args(args, a.inputs))
            ok = False
            if why is None:
                v = a.func(**aargs)
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


def _run_step(p, vals, init_state, policy=None, costs=None, known=None, memo=None, batch=None):
    """Evaluate one part on the current facts → StepOut. batch: (step names, decision parts) of a shared forward pass this
    decision part belongs to (see Flow.batches)."""
    t0 = time.perf_counter()
    args = {x: vals.get(x, MISSING) for x in p.inputs}
    h = memo if memo is not None else vhash
    hashes = {x: h(v) for x, v in args.items() if v is not MISSING}
    if any(v is MISSING for v in args.values()):
        err = "missing inputs: " + ", ".join(x for x, v in args.items() if v is MISSING)
        return StepOut(MISSING, None, 1.0, err, hashes, 0.0, tried=[] if p.alternatives is not None else None)
    if p.alternatives is not None:
        out = _run_group(p, args, init_state, policy, costs, known)
        out.hashes = hashes
        out.ms = (time.perf_counter() - t0) * 1000
        return out
    if p.tin is not None:                             # typed arguments: validated / coerced, a failure rejects the step
        args, why = _typed_args(p, args, known)
        if why:
            return StepOut(MISSING, None, 1.0, why, hashes, (time.perf_counter() - t0) * 1000)
    try:
        v = p.func(**args) if batch is None else p.func.in_pass(batch[1], args, batch[0])
    except Exception as e:  # noqa: BLE001
        return StepOut(MISSING, None, 1.0, f"{type(e).__name__}: {str(e)[:120]}", hashes, (time.perf_counter() - t0) * 1000)
    ms = (time.perf_counter() - t0) * 1000
    v = locate(p, v, init_state)
    value, quote, conf, probs = unwrap(v)
    why = ground(p, v, init_state)
    if not why and p.tout is not None:                # typed output: validated / coerced (a closed set: outside the options)
        from .typed import typed_out
        value, why = typed_out(p, value)
    why = why or validated(p, value, args)
    if why:                                           # rejected: the fact is missing, the claim stays visible in the error
        return StepOut(MISSING, quote, conf, why, hashes, ms, probs=probs, extra=_extra(v))
    return StepOut(value, quote, conf, None, hashes, ms, probs=probs, typed=p.tout.type if p.tout is not None else None,
                   extra=_extra(v))


def group_features(group, args):
    from .learned import scalar_row
    plain = _plain_args(args, group.inputs)
    if group.features is not None:
        import inspect
        names = list(inspect.signature(group.features).parameters)
        try:
            return scalar_row(group.features(**{x: plain[x] for x in names}))
        except Exception:  # noqa: BLE001
            return {}
    return scalar_row(plain)


def _run_group(group, args, init_state, policy, costs, known=None):
    """Alternative producers of one fact: try them in the policy's order (declaration order without a policy); the first
    output that passes its producer's checks is used. A shadow run (policy exploration) runs the rest too, for learning only."""
    row, notes, shadow = None, None, False
    order = list(group.alternatives)
    if policy is not None:
        row = group_features(group, args)
        order, notes, shadow = policy.plan(group, row, costs)
    tried, outcomes, alt_ms, used = [], {}, {}, None
    plain = _plain_args(args, group.inputs)
    for a in order:
        if used is not None and not shadow:
            break
        t = time.perf_counter()
        try:
            v, val, ok = None, None, False
            aargs, why = _typed_args(a, {x: args[x] for x in a.inputs}, known)
            if why is None:
                v = locate(a, a.func(**aargs), init_state)
                ok, why, val = accept(a, v, {**plain, **aargs} if a.tin else plain, init_state)
        except Exception as e:  # noqa: BLE001
            v, ok, why = None, False, f"error: {type(e).__name__}: {str(e)[:80]}"
        alt_ms[a.name] = (time.perf_counter() - t) * 1000
        outcomes[a.name] = (ok, v)
        if used is None:
            tried.append([a.name, why])
            if ok:
                used = (a, v, val)
        else:
            same = ok and vhash(val) == vhash(used[2])
            tried.append([a.name, "shadow: " + ("agrees" if same else ("differs" if ok else why))])
    if used is None:
        return StepOut(MISSING, None, 1.0, "no producer accepted: " + "; ".join(f"{n} {w}" for n, w in tried), {},
                       tried=tried, alt_ms=alt_ms, outcomes=outcomes, row=row, notes=notes)
    a, v, value = used
    _, quote, conf, probs = unwrap(v)
    return StepOut(value, quote, conf, None, {}, producer=a.name, tried=tried, alt_ms=alt_ms, outcomes=outcomes, row=row,
                   notes=notes, probs=probs, typed=a.tout.type if a.tout is not None else None, extra=_extra(v))


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
    in flow order, so the hash chain and replay do not depend on scheduling.

    order: None — all hard checks (and their inputs) first, together. An object with `p_fail(check, row)` and `row(vals,
    init_keys)` (solvi.learned.OrderModel, or an oracle) — hard checks one at a time, most expected saving first, stopping
    as soon as the failed ones settle every question they govern; answers are the same as with the default order.
    costs: a CostBook (ms per part) for the learned order and the producer policy. policy: a ProducerPolicy for facts with
    alternative producers (without one they are tried in declaration order). known: given fact → the type its value was
    already validated against (System(inputs=...)), so typed parts reading it with that type skip re-validation."""
    from .learned import CostBook
    costs = costs if costs is not None else CostBook()
    vals = dict(init_state)
    steps = flow.steps
    names = [st.part.name for st in steps]
    index = {n: i for i, n in enumerate(names)}
    done = {}
    live = set(flow.per_question)
    settled_by = {}
    schedule = []

    known = dict(known) if known else ({} if (catalog.readers or catalog.types) else None)

    memo = HashMemo()
    batch_of = {}                                     # step name → (names, decision parts) of its shared forward pass
    for group in getattr(flow, "batches", None) or ():
        group = [n for n in group if n in index]
        if len(group) > 1:
            b = (group, [steps[index[n]].part.func for n in group])
            for n in group:
                batch_of[n] = b

    def one(i, v):
        return _run_step(steps[i].part, v, init_state, policy, costs, known, memo, batch_of.get(names[i]) if batch_of else None)

    def keep(i):
        vals[names[i]] = done[i].value
        if known is not None and done[i].typed is not None:
            known[names[i]] = done[i].typed

    def run(idxs):
        if workers <= 1 or len(idxs) <= 1 or not _THREADS:
            for i in idxs:                                  # idxs are in topological (flow) order
                done[i] = one(i, vals)
                if done[i].value is not MISSING:
                    keep(i)
            return
        # dependency-driven: a step starts as soon as the steps it reads have finished
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        todo = set(idxs)
        deps = {i: {index[x] for x in steps[i].part.inputs if x in index and index[x] in todo} for i in idxs}
        running = {}
        with ThreadPoolExecutor(workers) as ex:
            while todo or running:
                for i in sorted(i for i in todo if not deps[i] & (todo | set(running.values()))):
                    todo.discard(i)
                    running[ex.submit(one, i, dict(vals))] = i
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in finished:
                    i = running.pop(fut)
                    done[i] = fut.result()
                    if done[i].value is not MISSING:
                        keep(i)

    def ancestors(i, acc=None):
        acc = set() if acc is None else acc
        if i not in acc and i not in done:
            acc.add(i)
            for x in steps[i].part.inputs:
                if x in index:
                    ancestors(index[x], acc)
        return acc

    hard = [i for i, st in enumerate(steps) if st.part.kind == "check" and st.part.hard]
    gov = {}
    for i in hard:
        part = steps[i].part
        checkpoint_of = {r.split(" ", 1)[1] for r in steps[i].reasons if r.startswith("checkpoint ")}
        gov[i] = {q for q in flow.per_question if names[i] in flow.per_question[q]
                  and (not part.then or q in part.then or q in checkpoint_of)}
    if early_exit and hard and order is None:
        first = set()
        for i in hard:
            ancestors(i, first)
        run(sorted(first))
        for i in hard:
            if done[i].value is False:
                for q in [q for q in live if q in gov[i]]:
                    live.discard(q)
                    settled_by[q] = names[i]
    elif early_exit and hard:
        _learned_hard_checks(catalog, flow, init_state, order, costs, steps, names, index, done, vals, live, settled_by,
                             hard, gov, run, ancestors, schedule)

    def needed(i):
        p = steps[i].part
        if not early_exit:
            return True
        if p.kind == "rule":
            return p.question in live
        return any(p.name in flow.per_question.get(q, ()) for q in live)
    run([i for i in range(len(steps)) if i not in done and needed(i)])

    init_hash = vhash(init_state)
    memo.clear()
    prev, recs, skipped, timings = init_hash, [], [], {}
    for i, st in enumerate(steps, 1):
        if i - 1 not in done:
            by = sorted({settled_by[q] for q in settled_by if st.part.name in flow.per_question.get(q, ()) or
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
                     probs=o.probs, extra=o.extra)
        rec.hash = vhash(rec.body())
        prev = rec.hash
        recs.append(rec)
    final = {k: v for k, v in vals.items()}
    trace = Trace(init_hash, recs, dict(init_state), skipped, schedule, timings)
    trace._outs = {names[i]: o for i, o in done.items() if o.outcomes is not None}    # for the producer policy (not hashed)
    return trace, final


def _learned_hard_checks(catalog, flow, init_state, order, costs, steps, names, index, done, vals, live, settled_by, hard,
                         gov, run, ancestors, schedule):
    """Hard checks one at a time, most expected saving first. A question is settled by a failed hard check only when every
    hard check governing it and declared earlier in the catalog has been evaluated and passed — so the check that decides is
    the same as with the default order (the first declared failed one). Stops when no unevaluated hard check is in the flow
    of a question that is still open."""
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
        run(sorted(acc))

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
        run(sorted(anc))
        rank += 1
        failed = done[i].value is False
        settle()
        schedule.append({"rank": rank, "check": names[i], "p_fail": p, "saves_ms": saves, "cost_ms": c, "score": score,
                         "why": why, "result": ("failed" if failed else "passed") + (
                             f"; settles {', '.join(sorted(q for q, by in settled_by.items() if by == names[i]))}"
                             if failed and any(by == names[i] for by in settled_by.values()) else "")})


def path_confidence(catalog, trace, facts):
    """A fact's confidence is the minimum confidence of the extractions it depends on."""
    by = {r.name: r for r in trace.records}
    memo = {}

    def conf(f):
        if f in memo:
            return memo[f]
        r = by.get(f)
        if r is None:
            memo[f] = 1.0
            return 1.0
        part = catalog.parts.get(f)
        c = r.confidence                  # 1.0 for plain computations; a quote's or a decision's confidence otherwise
        if part is not None:
            for x in part.inputs:
                c = min(c, conf(x))
        memo[f] = c
        return c
    return min([conf(f) for f in facts] or [1.0])


def now_ms():
    return time.perf_counter() * 1000
