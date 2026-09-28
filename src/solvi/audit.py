"""Audit of a response: for each answer, what it rests on — given inputs, computed facts, quotes (with offsets and the quoted
text), model decisions (model id, probabilities), learned parts, checks and constraints — and which safeguards fired
(grounding rejections, typed facts that failed their type, answers outside the options, low-confidence abstentions, hard
checks, constraint repairs, fallbacks).
Also a summary: how much of the answer's support is deterministic and how much comes from models."""
from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import FUZZY, classify, matches, snippet

STAT_KEYS = {"grounding": "grounding_rejected", "type_rejected": "type_rejected", "outside_options": "outside_options",
             "rule_abstained": "rule_abstained",
             "low_confidence": "low_confidence",
             "validator": "validator_rejected", "hard_check": "forced_by_hard_check",
             "constraint_repair": "constraint_repairs", "fallback": "fallbacks", "escalated": "model_escalated",
             "evidence_missing": "evidence_missing", "timeout": "timeouts"}
QUIET = {"evidence_missing", "timeout"}      # listed in safeguard_report only once they fire
STATS = ["asks", "answers", "abstained", "model_outputs"] + list(STAT_KEYS.values())

LABEL = {"grounding": "grounding rejected", "type_rejected": "type rejected", "outside_options": "outside the options",
         "rule_abstained": "rule abstained",
         "low_confidence": "low confidence",
         "validator": "rejected by validate", "hard_check": "hard check decided", "constraint_repair": "constraint repair",
         "fallback": "fallback producer", "escalated": "model escalated", "evidence_missing": "evidence missing",
         "timeout": "timed out"}


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
        if r.kind == "textin":                        # ask_text: the entry point / a field read from the text
            where = sorted(q for q in (r.extra or {}).get("candidates") or [r.value] if q in asked) \
                if r.name == "textin" else sorted(asked)
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
    for fact, why in getattr(res.trace, "rejected", None) or ():     # given facts that failed System(inputs=...)
        down = _downstream(catalog, fact)
        events.append({"kind": "type_rejected", "fact": fact, "detail": why,
                       "questions": sorted(q for q, fs in res.flow.unresolved.items() if set(fs) & down and q in asked)})
    for q, a in res.results.items():
        if a.guard in ("low_confidence", "escalated") and any(e["fact"] in ("answer:" + q, "textin") and q in e["questions"]
                                                             and e["kind"] == a.guard for e in events):
            pass                                      # the answer step itself was rejected: already counted once
        elif a.guard in ("hard_check", "outside_options", "rule_abstained", "low_confidence", "escalated", "grounding",
                         "type_rejected", "evidence_missing"):
            events.append({"kind": a.guard, "fact": "answer:" + q, "detail": a.why, "questions": [q]})
        if a.repaired:
            was, cons = a.repaired
            events.append({"kind": "constraint_repair", "fact": "answer:" + q, "questions": [q],
                           "detail": f"changed from {was!r} to satisfy {', '.join(cons)}"})
    return events, n_model


def _downstream(catalog, fact):
    """The fact and every fact computed from it (through the catalog's signatures)."""
    out = {fact}
    if catalog is None:
        return out
    grew = True
    while grew:
        grew = False
        for p in catalog.parts.values():
            if p.name not in out and any(x in out for x in p.inputs):
                out.add(p.name)
                grew = True
    return out


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


def _guarantee(au):
    """The promise of the escalation thresholds of the model decisions behind an answer: each distinct promise (from
    act_guard / calibrate_for), or "none" when a model decided without a calibrated threshold."""
    xs = [d.get("extra") or {} for d in au.decided] + ([au.rule.get("extra") or {}] if au.rule and au.rule.get("model")
                                                       else [])
    if not xs:
        return None
    got = sorted({f"{x['guarantee']['promise']} ({x['guarantee']['method']}, n = {x['guarantee']['n']})"
                  for x in xs if isinstance(x.get("guarantee"), dict)})
    if len(got) < len(xs):
        got.append("none for some decisions: their thresholds were not calibrated on your data (see act_guard)")
    return "; ".join(got)


def _model(m):
    return f"{m['type']} {m['id']} #{m['fp'][:8]}" if m else ""


def _extra_line(x):
    """A decision's recorded details: the act probability, an ordinal's expected level, the shared forward pass."""
    if not x:
        return ""
    out = []
    if x.get("act") is not None:
        out.append(f"act {x['act']:.2f}")
    if x.get("expected") is not None:
        out.append(f"expected {x['expected']:.2f}")
    if x.get("margin") is not None:
        out.append(f"margin {x['margin']:.2f}")
    if x.get("candidates"):
        out.append("candidates " + ", ".join(repr(c) for c in x["candidates"]))
    ps = x.get("pass")
    if isinstance(ps, dict):
        out.append(f"one pass with {', '.join(n for n in ps.get('with', []) if n)}" if ps.get("shared", True)
                   else "own pass (shared pass fell back)")
    return ("  " + "; ".join(out)) if out else ""


def _multi_lines(x, pad="              "):
    """A combination of models (solvi.multi): one line per proposal — each stage of a cascade, each vote, the routed part
    — with its value, its signal and what came of it; nested combinations indented."""
    if not isinstance(x, dict):
        return []
    from .multi import _shown

    def prop(e):
        s = f"{e['part']} [{e['model']}]" if e.get("model") else f"{e['part']} ({e.get('kind', '')})"
        s += f" {_shown(e.get('value'))}"
        if e.get("score") is not None:
            s += f"  {e.get('signal', 'signal')} {e['score']:.2f}"
        return s

    def esc(e):
        return f"escalated — {e['escalate']}" if e.get("escalate") else "answers alone"

    out = []
    if isinstance(x.get("stages"), list):
        by = x.get("answered_by")
        out.append(f"{pad}cascade     " + (f"stage {by + 1} answered" if by is not None else "every stage escalated")
                   + f"; {x.get('calls', len(x['stages']))} model(s) called")
        for i, e in enumerate(x["stages"]):
            out.append(f"{pad}  stage {i + 1}   {prop(e)} → " + ("answered" if i == by else esc(e)))
            out += _multi_lines(e, pad + "    ")
    if isinstance(x.get("votes"), list):
        agree = len({repr(e.get("value")) for e in x["votes"]}) == 1
        out.append(f"{pad}vote        rule {x.get('rule', 'all')}: " + ("all agree" if agree else "they disagree")
                   + f"; {x.get('calls', len(x['votes']))} model(s) called")
        for e in x["votes"]:
            out.append(f"{pad}  vote      {prop(e)} → {esc(e)}")
            out += _multi_lines(e, pad + "    ")
    if isinstance(x.get("route"), dict) and isinstance(x.get("routed"), dict):
        e = x["routed"]
        out.append(f"{pad}route       by {x['route'].get('by')} → {prop(e)} → {esc(e)}")
        out += _multi_lines(e, pad + "    ")
    return out


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
    guarantee: str | None = None                     # what the model thresholds behind the answer promise (act_guard …)
    counts: dict = field(default_factory=dict)       # support by provenance: given, computed, quoted, quoted_by_model, ...
    kind: str | None = None                          # the answer type's kind
    evidence: list = field(default_factory=list)     # [{"text", "start", "end", "source", "verified", "by_model"}] (a span first)
    extra: dict | None = None                        # an estimate's interval, a ranking's scores

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
        from .primitives import fmt
        a = "—" if self.answer is None else self.answer
        shown = repr(a) if self.answer is None else fmt(a, self.kind, self.extra)
        lines = [f"{self.question} = {shown}  [{self.status}]  confidence {self.confidence:.2f}"
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
                lines.append(f"  decided     {d['name']}: REJECTED — {d['error']}" + (f"  [{d['model']}]" if d.get("model") else "")
                             + _extra_line(d.get("extra")))
                lines += _multi_lines(d.get("extra"))
                continue
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((d["probs"] or {}).items(), key=lambda t: -t[1])[:4])
            lines.append(f"  decided     {d['name']} = {_short(d['value'])}" + (f"  ({p})" if p else "")
                         + (f"  [{d['model']}]" if d.get("model") else "") + _extra_line(d.get("extra")))
            lines += _multi_lines(d.get("extra"))
        for l_ in self.learned:
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((l_.get("probs") or {}).items(), key=lambda t: -t[1])[:4])
            lines.append(f"  learned     {l_['name']} = " + (f"— {l_['error']}" if l_.get("error") else _short(l_["value"]))
                         + (f"  ({p})" if p else "") + (f"  [{l_['model']}]" if l_.get("model") else ""))
            if l_.get("reads"):
                lines.append(f"              reads {', '.join(l_['reads'])}")
            if l_.get("ignored"):
                lines.append("              ignored " + "; ".join(f"{f} ({why})" for f, why in l_["ignored"].items()))
        for c in self.checks:
            tag = "hard" if c["hard"] else "soft"
            lines.append(f"  check       {c['name']} = {c['value']!r} ({tag}" + (", decides the answer" if c["decides"] else "") + ")"
                         + (f" — {c['error']}" if c.get("error") else ""))
        if self.rule:
            lines.append(f"  rule        {self.rule['name']} ({self.rule['provenance']})"
                         + (f"  [{self.rule['model']}]" if self.rule.get("model") else "") + _extra_line(self.rule.get("extra"))
                         + (f" — not computed: {self.rule['error']}" if self.rule.get("error") else ""))
            lines += _multi_lines(self.rule.get("extra"))
        for e in self.evidence:
            lines.append(f"  {'span' if e.get('span') else 'evidence':11s} {e['source']}[{e['start']}:{e['end']}] {e['text']!r}"
                         + (("  in the text; support not checked" if e["by_model"] else "  verified") if e["verified"]
                            else "  NOT IN THE TEXT") + ("  [model]" if e["by_model"] else ""))
        sc = (self.extra or {}).get("scores")
        if sc:
            lines.append("  scores      " + ", ".join(f"{k} {v:.2f}" for k, v in list(sc.items())[:6]))
        for c in self.constraints:
            lines.append(f"  constraint  {c['name']}: " + ("satisfied" if c["satisfied"] else "BROKEN"))
        if self.not_run:
            groups = {}
            for n, w in self.not_run:
                groups.setdefault(w, []).append(n)
            for w, ns in groups.items():
                lines.append(f"  not run     {', '.join(ns)} ({w})")
        lines.append(f"  → answer    {shown} — {self.why}")
        lines.append(f"  support     {self.support_line()}")
        if self.guarantee:
            lines.append(f"  guarantee   {self.guarantee}")
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
    overall: dict = None            # Response.overall: confidence, weakest answer, answered / abstained, complete, feasible

    def __getitem__(self, q):
        return self.answers[q]

    def __str__(self):
        head = (f"audit: {len(self.answers)} answer(s), {self.model_outputs} model output(s), "
                f"{len(self.safeguards)} safeguard event(s)")
        lines = [head]
        o = self.overall
        if o:
            w = f", weakest {o['weakest'][0]} {o['weakest'][1]:.2f}" if o["weakest"] else ""
            ab = f", abstained: {', '.join(o['abstained'])}" if o["abstained"] else ""
            ns = f", not stated: {', '.join(o['not_stated'])}" if o.get("not_stated") else ""
            lines.append(f"overall: confidence {o['confidence']:.2f}{w}; {o['answered']}/{o['answered'] + len(o['abstained'])} "
                         f"answered{ab}{ns}" + ("" if o["feasible"] else "; constraints violated"))
        return "\n".join(lines + [str(a) for a in self.answers.values()])

    def compact(self):
        """One line per answer: support shares and safeguards (what solvi.show prints)."""
        return "\n".join(f"{q}: {a.support_line()}; safeguards: {a.safeguard_line()}" for q, a in self.answers.items())

    def to_dict(self):
        return {"answers": {q: a.to_dict() for q, a in self.answers.items()}, "safeguards": self.safeguards,
                "model_outputs": self.model_outputs, "overall": self.overall}


def build(res, question=None, catalog=None):
    catalog = catalog if catalog is not None else getattr(res, "catalog", None)
    events = res.safeguards if getattr(res, "safeguards", None) is not None else None
    if events is None:
        events, n_model = collect(res, catalog)
    else:
        n_model = res.model_outputs
    qs = [question] if isinstance(question, str) else list(question or res.results)
    return Audit({q: _one(res, q, events, catalog) for q in qs}, events, n_model, getattr(res, "overall", None))


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
    read = {r.name[7:]: r for r in tr.records if r.kind == "textin" and r.name.startswith("textin:")}
    for k in sorted(x for x in reads if x in init):
        r = read.get(k)
        if r is not None and r.quote and r.provenance != "given":   # read from a text by a model (ask_text): not given
            s, e, src = r.quote
            full = init.get(src)[s:e] if isinstance(init.get(src), str) else None
            au.quoted.append({"name": k, "value": init[k], "start": s, "end": e, "source": src, "text": snippet(init, r.quote),
                              "match": None if full is None else matches(init[k], full), "confidence": r.confidence,
                              "model": _model(r.model), "error": None, "producer": "text in"})
            count("quoted_by_model")
            continue
        au.given.append({"name": k, "value": init[k]})
        count("given")
    entry = next((r for r in tr.records if r.kind == "textin" and r.name == "textin"), None)
    if entry is not None and (entry.value == q or q in ((entry.extra or {}).get("candidates") or ())):
        au.decided.append({"name": "entry point", "value": entry.value, "probs": entry.probs, "model": _model(entry.model),
                           "error": entry.error, "extra": None})
        count("decided" if entry.origin == "decided" else "given" if entry.origin == "given" else "computed")
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
                               "error": err, "extra": getattr(r, "extra", None)})
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
                   "model": _model(rr.model), "probs": rr.probs, "error": rr.error, "extra": getattr(rr, "extra", None)}
        count(rr.origin if rr.origin in FUZZY else "computed")
    if head is not None:
        entry = {"name": f"answer head ({head.model['type'] if head.model else 'head'})", "value": head.value,
                 "model": _model(head.model), "probs": head.probs, "provenance": "learned",
                 "reads": sorted(head.inputs)}
        hobj = getattr(res, "_heads", {}).get(q)
        if hobj is not None and getattr(hobj, "dropped", None):
            entry["ignored"] = dict(hobj.dropped)
        au.learned.append(entry)
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
    if a.evidence or a.kind is not None:
        au.kind, au.extra = a.kind, a.extra
        by_model = rr is not None and (rr.model is not None or rr.origin in FUZZY)
        for i, e in enumerate(a.evidence):
            text = init.get(e.source)
            ok = isinstance(text, str) and 0 <= e.start <= e.end <= len(text) and matches(str(e.value), text[e.start:e.end]) is not False
            au.evidence.append({"text": e.value, "start": e.start, "end": e.end, "source": e.source, "verified": bool(ok),
                                "by_model": by_model, "span": a.kind == "span" and i == 0})
            count("quoted_by_model" if by_model else "quoted")
    au.safeguards = [e for e in events if q in e["questions"]]
    au.guarantee = _guarantee(au)
    au.counts = counts
    return au
