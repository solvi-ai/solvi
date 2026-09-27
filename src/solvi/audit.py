"""Audit of a response: for each answer, what it rests on — given inputs, computed facts, quotes (with offsets and the quoted
text), model decisions (model id, probabilities), learned parts, checks and constraints — and which safeguards fired
(grounding rejections, answers outside the options, low-confidence abstentions, hard checks, constraint repairs, fallbacks).
Also a summary: how much of the answer's support is deterministic and how much comes from models."""
from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import FUZZY, classify, matches, snippet

STAT_KEYS = {"grounding": "grounding_rejected", "outside_options": "outside_options", "low_confidence": "low_confidence",
             "validator": "validator_rejected", "hard_check": "forced_by_hard_check",
             "constraint_repair": "constraint_repairs", "fallback": "fallbacks"}
STATS = ["asks", "answers", "abstained", "model_outputs"] + list(STAT_KEYS.values())

LABEL = {"grounding": "grounding rejected", "outside_options": "outside the options", "low_confidence": "low confidence",
         "validator": "rejected by validate", "hard_check": "hard check decided", "constraint_repair": "constraint repair",
         "fallback": "fallback producer"}


def _missing(v):
    from .runtime import MISSING
    return v is MISSING


def collect(res, catalog=None):
    """The safeguards that fired in a response → ([event], number of model outputs). An event is a dict
    {"kind", "fact", "detail", "questions"}."""
    per_q = res.flow.per_question
    asked = set(res.results)

    def qs_of(fact):
        return sorted(q for q, fs in per_q.items() if fact in fs and q in asked)
    events, n_model = [], 0
    for r in res.trace.records:
        if r.model is not None or r.origin in FUZZY:
            n_model += 1
        where = [r.name[7:]] if r.name.startswith("answer:") else qs_of(r.name)
        if r.tried is not None:
            first = None
            for name, why in r.tried:
                if why.startswith("shadow"):
                    continue
                first = first or name
                if name == r.producer:
                    continue
                alt = _alt(catalog, r.name, name)
                if alt is not None and alt.model is not None:
                    n_model += 1
                k = classify(why)
                if k:
                    events.append({"kind": k, "fact": r.name, "detail": f"{name}: {why}", "questions": where})
            if r.producer is not None and first is not None and first != r.producer:
                rejected = [n for n, w in r.tried if not w.startswith("shadow") and n != r.producer]
                events.append({"kind": "fallback", "fact": r.name, "questions": where,
                               "detail": f"{r.producer} used after {', '.join(rejected)} rejected"})
        elif r.error:
            k = classify(r.error)
            if k:
                events.append({"kind": k, "fact": r.name, "detail": r.error, "questions": where})
    for q, a in res.results.items():
        if a.guard in ("hard_check", "outside_options", "low_confidence"):
            events.append({"kind": a.guard, "fact": "answer:" + q, "detail": a.why, "questions": [q]})
        if a.repaired:
            was, cons = a.repaired
            events.append({"kind": "constraint_repair", "fact": "answer:" + q, "questions": [q],
                           "detail": f"changed from {was!r} to satisfy {', '.join(cons)}"})
    return events, n_model


def _alt(catalog, fact, name):
    if catalog is None:
        return None
    try:
        return catalog.alternative(fact, name)
    except (KeyError, TypeError):
        return None


def _short(v, n=48):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _model(m):
    return f"{m['type']} {m['id']} #{m['fp'][:8]}" if m else ""


@dataclass
class AnswerAudit:
    question: str
    answer: object
    status: str
    confidence: float
    provenance: str | None
    source: str | None
    why: str
    given: list = field(default_factory=list)        # [{"name", "value"}]
    computed: list = field(default_factory=list)     # [{"name", "value", "error"}]
    quoted: list = field(default_factory=list)       # [{"name", "value", "start", "end", "source", "text", "match", "model", ...}]
    decided: list = field(default_factory=list)      # [{"name", "value", "probs", "model"}]
    learned: list = field(default_factory=list)      # [{"name", "value", "model", "probs"}]
    checks: list = field(default_factory=list)       # [{"name", "value", "hard", "decides"}]
    rule: dict | None = None                         # the answer step: {"name", "provenance", "model", "probs"}
    constraints: list = field(default_factory=list)  # [{"name", "satisfied"}]
    safeguards: list = field(default_factory=list)   # events (see collect)
    not_run: list = field(default_factory=list)      # [(part, why)] parts of the flow skipped at run time
    counts: dict = field(default_factory=dict)       # support by provenance: given, computed, quoted, quoted_by_model, ...

    @property
    def deterministic(self):
        c = self.counts
        return c.get("given", 0) + c.get("computed", 0) + c.get("quoted", 0)

    @property
    def fuzzy(self):
        return sum(self.counts.values()) - self.deterministic

    @property
    def share_deterministic(self):
        n = sum(self.counts.values())
        return 1.0 if n == 0 else self.deterministic / n

    def support_line(self):
        c = self.counts
        parts = [f"{c[k]} {k.replace('_', ' ')}" for k in ("given", "computed", "quoted", "quoted_by_model", "decided",
                                                             "learned", "proposed") if c.get(k)]
        return (f"{sum(c.values())} items ({', '.join(parts) or 'none'}): {self.share_deterministic:.0%} deterministic"
                + (f", {self.fuzzy} from models" if self.fuzzy else ""))

    def safeguard_line(self):
        if not self.safeguards:
            return "none fired"
        n = {}
        for e in self.safeguards:
            n[e["kind"]] = n.get(e["kind"], 0) + 1
        return ", ".join(f"{LABEL[k]} ×{v}" for k, v in n.items())

    def __str__(self):
        a = "—" if self.answer is None else self.answer
        lines = [f"{self.question} = {a!r}  [{self.status}]  confidence {self.confidence:.2f}"
                 + (f"  ← {self.provenance}" if self.provenance else "") + (f" by {self.source}" if self.source else "")]
        if self.given:
            lines.append("  given       " + "; ".join(f"{g['name']} = {_short(g['value'], 32)}" for g in self.given))
        for c in self.computed:
            lines.append(f"  computed    {c['name']} = " + (f"— rejected/failed: {c['error']}" if c.get("error") else _short(c["value"])))
        for q in self.quoted:
            if q.get("error"):
                lines.append(f"  quoted      {q['name']}: REJECTED — {q['error']}" + (f"  [{q['model']}]" if q.get("model") else ""))
                continue
            how = {True: "literal", False: "derived from", None: "converted from"}[q["match"]]
            lines.append(f"  quoted      {q['name']} = {_short(q['value'])}  {q['source']}[{q['start']}:{q['end']}] {how} "
                         f"{q['text']!r}" + (f"  confidence {q['confidence']:.2f}" if q["confidence"] < 1 else "")
                         + (f"  [{q['model']}]" if q.get("model") else ""))
        for d in self.decided:
            if d.get("error"):
                lines.append(f"  decided     {d['name']}: REJECTED — {d['error']}" + (f"  [{d['model']}]" if d.get("model") else ""))
                continue
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((d["probs"] or {}).items(), key=lambda t: -t[1])[:4])
            lines.append(f"  decided     {d['name']} = {_short(d['value'])}" + (f"  ({p})" if p else "")
                         + (f"  [{d['model']}]" if d.get("model") else ""))
        for l_ in self.learned:
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((l_.get("probs") or {}).items(), key=lambda t: -t[1])[:4])
            lines.append(f"  learned     {l_['name']} = " + (f"— {l_['error']}" if l_.get("error") else _short(l_["value"]))
                         + (f"  ({p})" if p else "") + (f"  [{l_['model']}]" if l_.get("model") else ""))
        for c in self.checks:
            tag = "hard" if c["hard"] else "soft"
            lines.append(f"  check       {c['name']} = {c['value']!r} ({tag}" + (", decides the answer" if c["decides"] else "") + ")"
                         + (f" — {c['error']}" if c.get("error") else ""))
        if self.rule:
            lines.append(f"  rule        {self.rule['name']} ({self.rule['provenance']})"
                         + (f"  [{self.rule['model']}]" if self.rule.get("model") else "")
                         + (f" — not computed: {self.rule['error']}" if self.rule.get("error") else ""))
        for c in self.constraints:
            lines.append(f"  constraint  {c['name']}: " + ("satisfied" if c["satisfied"] else "BROKEN"))
        if self.not_run:
            groups = {}
            for n, w in self.not_run:
                groups.setdefault(w, []).append(n)
            for w, ns in groups.items():
                lines.append(f"  not run     {', '.join(ns)} ({w})")
        lines.append(f"  → answer    {a!r} — {self.why}")
        lines.append(f"  support     {self.support_line()}")
        lines.append(f"  safeguards  {self.safeguard_line()}")
        for e in self.safeguards:
            lines.append(f"              · {LABEL[e['kind']]}: {e['fact']} — {e['detail']}")
        return "\n".join(lines)

    def to_dict(self):
        from dataclasses import asdict
        d = asdict(self)
        d.update(deterministic=self.deterministic, fuzzy=self.fuzzy, share_deterministic=self.share_deterministic)
        return d


@dataclass
class Audit:
    answers: dict                   # question → AnswerAudit
    safeguards: list                # every event in the response
    model_outputs: int

    def __getitem__(self, q):
        return self.answers[q]

    def __str__(self):
        head = (f"audit: {len(self.answers)} answer(s), {self.model_outputs} model output(s), "
                f"{len(self.safeguards)} safeguard event(s)")
        return "\n".join([head] + [str(a) for a in self.answers.values()])

    def compact(self):
        """One line per answer: support shares and safeguards (what solvi.show prints)."""
        return "\n".join(f"{q}: {a.support_line()}; safeguards: {a.safeguard_line()}" for q, a in self.answers.items())

    def to_dict(self):
        return {"answers": {q: a.to_dict() for q, a in self.answers.items()}, "safeguards": self.safeguards,
                "model_outputs": self.model_outputs}


def build(res, question=None, catalog=None):
    catalog = catalog if catalog is not None else getattr(res, "catalog", None)
    events = res.safeguards if getattr(res, "safeguards", None) is not None else None
    if events is None:
        events, n_model = collect(res, catalog)
    else:
        n_model = res.model_outputs
    qs = [question] if isinstance(question, str) else list(question or res.results)
    return Audit({q: _one(res, q, events, catalog) for q in qs}, events, n_model)


def _one(res, q, events, catalog):
    tr, flow, a = res.trace, res.flow, res.results[q]
    init = tr.init
    by = {r.name: r for r in tr.records if r.kind != "head"}
    head = next((r for r in tr.records if r.kind == "head" and r.name == "answer:" + q), None)
    parts = {st.part.name: st.part for st in flow.steps}
    facts = set(flow.per_question.get(q, []))
    rule_part = parts.get("answer:" + q)
    au = AnswerAudit(q, a.answer, a.status, a.confidence, a.provenance, a.source, a.why)
    counts = {}

    def count(k):
        counts[k] = counts.get(k, 0) + 1
    reads = set()
    for f in facts:
        if f in parts and f in by:                    # inputs read by the parts that ran
            reads |= set(parts[f].inputs)
    if rule_part is not None:
        reads |= set(rule_part.inputs)
    if head is not None:
        reads |= set(head.inputs)
    for k in sorted(x for x in reads if x in init):
        au.given.append({"name": k, "value": init[k]})
        count("given")
    skipped = dict(tr.skipped)
    deciding = a.source if a.guard == "hard_check" else None
    for st in flow.steps:
        f = st.part.name
        if f not in facts:
            continue
        r = by.get(f)
        if r is None:
            if f in skipped:
                au.not_run.append((f, skipped[f]))
            continue
        m = _model(r.model)
        err = r.error if r.error else None
        origin = r.origin
        if r.kind == "check":
            p = st.part
            au.checks.append({"name": f, "value": None if _missing(r.value) else r.value, "hard": bool(p.hard),
                              "decides": f == deciding, "error": err})
            count("computed" if r.model is None else (origin if origin in FUZZY else "quoted_by_model"))
        elif origin == "quoted":
            s, e, src = r.quote if r.quote else (0, 0, "doc")
            text = snippet(init, r.quote) if r.quote else None
            full = init.get(src)[s:e] if r.quote and isinstance(init.get(src), str) else None
            au.quoted.append({"name": f, "value": None if _missing(r.value) else r.value, "start": s, "end": e, "source": src,
                              "text": text, "match": None if (_missing(r.value) or full is None) else matches(r.value, full),
                              "confidence": r.confidence, "model": m, "error": err, "producer": r.producer})
            count("quoted_by_model" if r.model is not None else "quoted")
        elif origin == "decided":
            au.decided.append({"name": f, "value": None if _missing(r.value) else r.value, "probs": r.probs, "model": m,
                               "error": err})
            count("decided")
        elif origin in ("learned", "proposed"):
            au.learned.append({"name": f, "value": None if _missing(r.value) else r.value, "model": m, "error": err,
                               "probs": r.probs, "provenance": origin})
            count(origin)
        else:
            au.computed.append({"name": f, "value": None if _missing(r.value) else r.value, "error": err})
            count("computed")
    rr = by.get("answer:" + q)
    if rr is not None:
        au.rule = {"name": rule_part.func.__name__ if rule_part is not None else rr.name, "provenance": rr.origin,
                   "model": _model(rr.model), "probs": rr.probs, "error": rr.error}
        count(rr.origin if rr.origin in FUZZY else "computed")
    if head is not None:
        au.learned.append({"name": f"answer head ({head.model['type'] if head.model else 'head'})", "value": head.value,
                           "model": _model(head.model), "probs": head.probs, "provenance": "learned"})
        count("learned")
    if catalog is not None:
        for c in catalog.constraints.values():
            if q in c.inputs and all(x in res.results for x in c.inputs):
                assign = {x: res.results[x].answer for x in c.inputs}
                try:
                    ok = any(v is None for v in assign.values()) or bool(c.func(**assign))
                except Exception:  # noqa: BLE001
                    ok = False
                au.constraints.append({"name": c.name, "satisfied": ok})
    au.safeguards = [e for e in events if q in e["questions"]]
    au.counts = counts
    return au
