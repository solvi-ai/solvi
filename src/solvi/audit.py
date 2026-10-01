"""Audit of a response: for each answer, what it rests on — given inputs, computed facts, quotes (with offsets and the quoted
text), model decisions (model id, probabilities), learned parts, checks and constraints — and which safeguards fired
(grounding rejections, typed facts that failed their type, answers outside the options, low-confidence abstentions, hard
checks, constraint repairs, fallbacks).
Also a summary: how much of the answer's support is deterministic and how much comes from models."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import i18n
from .provenance import FUZZY, classify, matches, snippet

STAT_KEYS = {"grounding": "grounding_rejected", "type_rejected": "type_rejected", "outside_options": "outside_options",
             "rule_abstained": "rule_abstained",
             "low_confidence": "low_confidence",
             "validator": "validator_rejected", "hard_check": "forced_by_hard_check",
             "constraint_repair": "constraint_repairs", "fallback": "fallbacks", "escalated": "model_escalated",
             "evidence_missing": "evidence_missing", "timeout": "timeouts", "instruction": "instruction_flips",
             "memory": "memory_disagreements"}
QUIET = {"evidence_missing", "timeout", "instruction", "memory"}      # listed in safeguard_report only once they fire
STATS = ["asks", "answers", "abstained", "model_outputs"] + list(STAT_KEYS.values())

LABEL = {k: i18n.label(k) for k in STAT_KEYS}      # safeguard → its English label (solvi.i18n has the other languages)
COLUMNS = ["col.given", "col.computed", "col.quoted", "col.decided", "col.learned", "col.check", "col.rule", "col.evidence",
           "col.span", "col.scores", "col.constraint", "col.not_run", "col.answer", "col.support", "col.guarantee",
           "col.safeguards"]


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
        if a.guard in ("low_confidence", "escalated", "instruction", "grounding", "memory") and any(
                e["fact"] in ("answer:" + q, "textin") and q in e["questions"] and e["kind"] == a.guard for e in events):
            pass                                      # the answer step itself was rejected: already counted once
        elif a.guard in ("hard_check", "outside_options", "rule_abstained", "low_confidence", "escalated", "grounding",
                         "type_rejected", "evidence_missing", "instruction", "memory"):
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
    if any(not isinstance(x.get("guarantee"), dict) for x in xs):     # two decisions may share one promise
        got.append("none for some decisions: their thresholds were not calibrated on your data (see act_guard)")
    return "; ".join(got)


def _model(m):
    return f"{m['type']} {m['id']} #{m['fp'][:8]}" if m else ""


def _extra_line(x, lang=None):
    """A decision's recorded details: the act probability, an ordinal's expected level, the shared forward pass."""
    if not x:
        return ""
    t = i18n.t
    out = []
    if x.get("act") is not None:
        out.append(t("x.act", lang, x=f"{x['act']:.2f}"))
    if x.get("expected") is not None:
        out.append(t("x.expected", lang, x=f"{x['expected']:.2f}"))
    if x.get("margin") is not None:
        out.append(t("x.margin", lang, x=f"{x['margin']:.2f}"))
    if x.get("candidates"):
        out.append(t("x.candidates", lang, x=", ".join(repr(c) for c in x["candidates"])))
    lg = x.get("long")
    if isinstance(lg, dict) and lg.get("mode") == "full" and not lg.get("fallback"):   # long="full": read whole
        out.append(f"read whole ({lg.get('tokens')} tokens, up to {lg.get('max_len')})")
    elif isinstance(lg, dict):                        # a long text: the sections the decider read (solvi.longdoc)
        out.append((f"longer than {lg.get('max_len')} tokens ({lg.get('tokens')}): " if lg.get("fallback") else "") +
                   f"read {lg.get('read')} of {lg.get('of')} sections ({lg.get('by')}): "
                   + ", ".join(f"[{a}:{b}]" + (f" {h[:30]!r}" if h else "") for a, b, h, _ in lg.get("sections") or []))
    tr = x.get("truncated")
    if isinstance(tr, dict):                          # an input that did not fit the pass: how much of it was read
        out.append(t("x.truncated", lang, read=tr.get("read_tokens"), of=tr.get("input_tokens")))
    ps = x.get("pass")
    if isinstance(ps, dict):
        out.append(t("x.pass", lang, x=", ".join(n for n in ps.get("with", []) if n)) if ps.get("shared", True)
                   else t("x.own_pass", lang))
    return ("  " + "; ".join(out)) if out else ""


def _multi_lines(x, pad=None, lang=None):
    """A combination of models (solvi.multi): one line per proposal — each stage of a cascade, each vote, the routed part
    — with its value, its signal and what came of it; nested combinations indented."""
    if not isinstance(x, dict):
        return []
    from .multi import _shown
    t = i18n.t
    pad = " " * (2 + i18n.width(COLUMNS, lang)) if pad is None else pad

    def prop(e):
        s = f"{e['part']} [{e['model']}]" if e.get("model") else f"{e['part']} ({e.get('kind', '')})"
        s += f" {_shown(e.get('value'))}"
        if e.get("score") is not None:
            s += f"  {e.get('signal', 'signal')} {e['score']:.2f}"
        return s

    def esc(e):
        return t("m.escalated", lang, e=i18n.msg(e["escalate"], lang)) if e.get("escalate") else t("m.alone", lang)

    out = []
    if isinstance(x.get("stages"), list):
        by = x.get("answered_by")
        out.append(pad + t("m.cascade", lang) + (t("m.stage_answered", lang, i=by + 1) if by is not None
                                                 else t("m.every_stage", lang))
                   + t("m.calls", lang, n=x.get("calls", len(x["stages"]))))
        for i, e in enumerate(x["stages"]):
            out.append(pad + t("m.stage", lang, i=i + 1) + f"{prop(e)} → " + (t("m.answered", lang) if i == by else esc(e)))
            out += _multi_lines(e, pad + "    ", lang)
    if isinstance(x.get("votes"), list):
        agree = len({repr(e.get("value")) for e in x["votes"]}) == 1
        out.append(pad + t("m.vote", lang, rule=x.get("rule", "all")) + t("m.all_agree" if agree else "m.disagree", lang)
                   + t("m.calls", lang, n=x.get("calls", len(x["votes"]))))
        for e in x["votes"]:
            out.append(pad + t("m.one_vote", lang) + f"{prop(e)} → {esc(e)}")
            out += _multi_lines(e, pad + "    ", lang)
    if isinstance(x.get("route"), dict) and isinstance(x.get("routed"), dict):
        e = x["routed"]
        out.append(pad + t("m.route", lang, by=x["route"].get("by")) + f"{prop(e)} → {esc(e)}")
        out += _multi_lines(e, pad + "    ", lang)
    if isinstance(x.get("memory"), dict):
        out += memory_lines(x["memory"], pad, lang)
    return out


def memory_lines(mem, pad="", lang=None):
    """A memory of corrections on a decision (solvi.memory): its proposal or why it abstained, what came of it, and the
    corrected cases it rests on (id, label, distance, source, who, the stored id)."""
    t = i18n.t
    near = mem.get("neighbours") or []
    head = pad + t("mem.head", lang, n=mem.get("n", 0), k=len(near))
    if mem.get("proposal") is None:
        head += t("mem.abstains", lang, why=i18n.msg(mem.get("abstain") or "", lang))
    else:
        head += t("mem.proposes", lang, v=repr(mem["proposal"]), s=f"{mem.get('strength', 0):.2f}",
                  a=f"{mem.get('agreement', 0):.0%}")
        act = mem.get("action", "")
        head += (t("mem.answered", lang, x=i18n.msg(mem.get("replaced", ""), lang)) if act == "answered" else
                 t({"agrees": "mem.agrees", "escalated": "mem.escalated",
                    "agrees (already escalated)": "mem.agrees_late"}.get(act, "mem.disagrees"), lang))
    out = [head]
    for c in near:
        src = ", ".join(str(v) for v in (c.get("source"), c.get("by"),
                                         f"#{c['stored_id']}" if c.get("stored_id") else None) if v)
        out.append(pad + t("mem.case", lang, id=c["id"][:8], v=repr(c["label"]), d=f"{c['distance']:.3f}", src=src))
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
    def memory(self):
        """The memories of corrections consulted for this answer (solvi.memory): [{"name", the recorded extra["memory"]:
        "proposal", "action", "neighbours" — the corrected cases it rested on — "fp", ...}]."""
        xs = [(d["name"], d.get("extra")) for d in self.decided] + ([(self.rule["name"], self.rule.get("extra"))]
                                                                   if self.rule else [])
        return [{"name": n, **x["memory"]} for n, x in xs if isinstance(x, dict) and isinstance(x.get("memory"), dict)]

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

    _lang = "en"                                     # the language str() renders in (not a field: to_dict() stays the same)

    def support_line(self, lang=None):
        lang = self._lang if lang is None else lang
        t = i18n.t
        c = self.counts
        parts = [t("au.kind", lang, n=c[k], kind=t("kind." + k, lang)) for k in ("given", "computed", "quoted",
                                                                               "quoted_by_model", "decided", "learned",
                                                                               "proposed") if c.get(k)]
        return (t("au.support", lang, n=sum(c.values()), parts=", ".join(parts) or t("au.none", lang),
                  share=f"{self.share_deterministic:.0%}")
                + (t("au.from_models", lang, n=self.fuzzy) if self.fuzzy else ""))

    def safeguard_line(self, lang=None):
        lang = self._lang if lang is None else lang
        if not self.safeguards:
            return i18n.t("au.none_fired", lang)
        n = {}
        for e in self.safeguards:
            n[e["kind"]] = n.get(e["kind"], 0) + 1
        return ", ".join(f"{i18n.label(k, lang)} ×{v}" for k, v in n.items())

    def __str__(self):
        return self.render()

    def render(self, lang=None):
        """The audit of this answer as text, in a language (solvi.i18n; default: the one it was built with, "en")."""
        from .primitives import fmt
        lang = i18n.check(self._lang if lang is None else lang)
        t, m = i18n.t, (lambda x: i18n.msg(x, lang))
        w = i18n.width(COLUMNS, lang)
        pad = " " * (2 + w)

        def col(key):
            return "  " + f"{t(key, lang):{w}s}"
        a = "—" if self.answer is None else self.answer
        shown = repr(a) if self.answer is None else fmt(a, self.kind, self.extra)
        lines = [t("au.answer", lang, q=self.question, shown=shown, status=i18n.status(self.status, lang),
                   c=f"{self.confidence:.2f}")
                 + (t("au.prov", lang, prov=i18n.provenance(self.provenance, lang)) if self.provenance else "")
                 + (t("au.by", lang, src=self.source) if self.source else "")]
        if self.given:
            lines.append(col("col.given") + "; ".join(f"{g['name']} = {_short(g['value'], 32)}" for g in self.given))
        for c in self.computed:
            lines.append(col("col.computed") + f"{c['name']} = "
                         + (t("au.failed", lang, err=m(c["error"])) if c.get("error") else _short(c["value"])))
        for q in self.quoted:
            if q.get("error"):
                lines.append(col("col.quoted") + f"{q['name']}: " + t("au.rejected", lang, err=m(q["error"]))
                             + (f"  [{q['model']}]" if q.get("model") else ""))
                continue
            how = t({True: "au.literal", False: "au.derived", None: "au.converted"}[q["match"]], lang)
            lines.append(col("col.quoted") + f"{q['name']} = {_short(q['value'])}  {q['source']}[{q['start']}:{q['end']}] "
                         f"{how} {q['text']!r}" + (t("au.confidence", lang, c=f"{q['confidence']:.2f}")
                                                   if q["confidence"] < 1 else "")
                         + (f"  [{q['model']}]" if q.get("model") else ""))
        for d in self.decided:
            if d.get("error"):
                lines.append(col("col.decided") + f"{d['name']}: " + t("au.rejected", lang, err=m(d["error"]))
                             + (f"  [{d['model']}]" if d.get("model") else "") + _extra_line(d.get("extra"), lang))
                lines += _multi_lines(d.get("extra"), pad, lang)
                continue
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((d["probs"] or {}).items(), key=lambda t_: -t_[1])[:4])
            lines.append(col("col.decided") + f"{d['name']} = {_short(d['value'])}" + (f"  ({p})" if p else "")
                         + (f"  [{d['model']}]" if d.get("model") else "") + _extra_line(d.get("extra"), lang))
            lines += _multi_lines(d.get("extra"), pad, lang)
        for l_ in self.learned:
            p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((l_.get("probs") or {}).items(), key=lambda t_: -t_[1])[:4])
            lines.append(col("col.learned") + f"{l_['name']} = " + (f"— {m(l_['error'])}" if l_.get("error")
                                                                  else _short(l_["value"]))
                         + (f"  ({p})" if p else "") + (f"  [{l_['model']}]" if l_.get("model") else ""))
            if l_.get("reads"):
                lines.append(pad + t("au.reads", lang, x=", ".join(l_["reads"])))
            if l_.get("ignored"):
                lines.append(pad + t("au.ignored", lang, x="; ".join(f"{f} ({m(why)})" for f, why in l_["ignored"].items())))
        for c in self.checks:
            tag = t("au.hard" if c["hard"] else "au.soft", lang)
            lines.append(col("col.check") + f"{c['name']} = {c['value']!r} ({tag}"
                         + (t("au.decides", lang) if c["decides"] else "") + ")"
                         + (f" — {m(c['error'])}" if c.get("error") else ""))
        if self.rule:
            lines.append(col("col.rule") + f"{self.rule['name']} ({i18n.provenance(self.rule['provenance'], lang)})"
                         + (f"  [{self.rule['model']}]" if self.rule.get("model") else "")
                         + _extra_line(self.rule.get("extra"), lang)
                         + (t("au.not_computed", lang, err=m(self.rule["error"])) if self.rule.get("error") else ""))
            lines += _multi_lines(self.rule.get("extra"), pad, lang)
        for e in self.evidence:
            lines.append(col("col.span" if e.get("span") else "col.evidence")
                         + f"{e['source']}[{e['start']}:{e['end']}] {e['text']!r}"
                         + ((t("au.unchecked", lang) if e["by_model"] else t("au.verified", lang)) if e["verified"]
                            else t("au.not_in_text", lang)) + (t("au.model", lang) if e["by_model"] else ""))
        sc = (self.extra or {}).get("scores")
        if sc:
            lines.append(col("col.scores") + ", ".join(f"{k} {v:.2f}" for k, v in list(sc.items())[:6]))
        for c in self.constraints:
            lines.append(col("col.constraint") + f"{c['name']}: " + t("au.satisfied" if c["satisfied"] else "au.broken", lang))
        if self.not_run:
            groups = {}
            for n, w_ in self.not_run:
                groups.setdefault(w_, []).append(n)
            for w_, ns in groups.items():
                lines.append(col("col.not_run") + f"{', '.join(ns)} ({m(w_)})")
        lines.append(col("col.answer") + f"{shown} — {m(self.why)}")
        lines.append(col("col.support") + self.support_line(lang))
        if self.guarantee:
            lines.append(col("col.guarantee") + m(self.guarantee))
        lines.append(col("col.safeguards") + self.safeguard_line(lang))
        for e in self.safeguards:
            lines.append(pad + f"· {i18n.label(e['kind'], lang)}: {e['fact']} — {m(e['detail'])}")
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
    overall: dict | None = None     # Response.overall: confidence, weakest answer, answered / abstained, complete, feasible

    def __getitem__(self, q):
        return self.answers[q]

    _lang = "en"                    # the language str() renders in (not a field: to_dict() stays the same)

    def __str__(self):
        return self.render()

    def render(self, lang=None):
        """The whole audit as text, in a language (solvi.i18n; default: the one it was built with, "en")."""
        lang = i18n.check(self._lang if lang is None else lang)
        t = i18n.t
        lines = [t("au.head", lang, n=len(self.answers), m=self.model_outputs, k=len(self.safeguards))]
        o = self.overall
        if o:
            w = t("au.weakest", lang, q=o["weakest"][0], c=f"{o['weakest'][1]:.2f}") if o["weakest"] else ""
            ab = t("au.abstained", lang, qs=", ".join(o["abstained"])) if o["abstained"] else ""
            ns = t("au.not_stated", lang, qs=", ".join(o["not_stated"])) if o.get("not_stated") else ""
            lines.append(t("au.overall", lang, c=f"{o['confidence']:.2f}", weakest=w, a=o["answered"],
                           t=o["answered"] + len(o["abstained"]), abstained=ab, not_stated=ns)
                         + ("" if o["feasible"] else t("au.infeasible", lang)))
        return "\n".join(lines + [a.render(lang) for a in self.answers.values()])

    def compact(self, lang=None):
        """One line per answer: support shares and safeguards (what solvi.show prints)."""
        lang = self._lang if lang is None else lang
        return "\n".join(i18n.t("au.compact", lang, q=q, support=a.support_line(lang), safeguards=a.safeguard_line(lang))
                         for q, a in self.answers.items())

    def to_dict(self):
        return {"answers": {q: a.to_dict() for q, a in self.answers.items()}, "safeguards": self.safeguards,
                "model_outputs": self.model_outputs, "overall": self.overall}


def build(res, question=None, catalog=None, lang=None):
    lang = i18n.check(lang)
    catalog = catalog if catalog is not None else getattr(res, "catalog", None)
    events = res.safeguards if getattr(res, "safeguards", None) is not None else None
    if events is None:
        events, n_model = collect(res, catalog)
    else:
        n_model = res.model_outputs
    qs = [question] if isinstance(question, str) else list(question or res.results)
    out = Audit({q: _one(res, q, events, catalog) for q in qs}, events, n_model, getattr(res, "overall", None))
    if lang != i18n.DEFAULT:
        out._lang = lang
        for a in out.answers.values():
            a._lang = lang
    return out


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
