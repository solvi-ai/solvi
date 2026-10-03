"""Response: the answers of one ask with the flow, the trace, every fact's value and the safeguards (System.ask returns
one; a TraceStorage loads one back). Moved out of solvi.system in 1.0, which re-exports it, so that the stores below the
System read a stored response without importing the System."""
from __future__ import annotations

from dataclasses import dataclass

from . import _deprecate
from .core import Serial
from .runtime import MISSING


@dataclass
class Response(Serial):
    """The answers of one ask, with the flow, the trace, every fact's value and the safeguards.

    A response keeps a strong reference to the System that answered it (for `report()`: the questions, the answer heads,
    the replay). Deliberately not a weak one: `System(cat, qs).ask(state).report()` is common
    and the temporary System would be gone before the report runs. A response you keep for long keeps its System (and its
    models) alive; keep `res.to_dict()` or the stored record instead, or drop the reference with `res._system = None`
    (then pass `system=` to report)."""
    results: dict
    flow: object
    trace: object
    values: dict
    ms: float
    feasible: bool = True               # do the answers satisfy every applicable constraint?
    violations: list | None = None      # names of the constraints still broken (fixed answers that conflict)
    catalog: object = None              # the catalog that answered (for the audit)
    safeguards: list | None = None      # safeguard events of this response (see solvi.audit.collect)
    model_outputs: int = 0              # outputs produced by models in this response
    stored_id = None                    # its id in a TraceStorage once saved (System(storage=...) saves every ask)
    read = None                         # ask_text: the solvi.textin.TextRead the question and state were read from
    textin = _deprecate.removed_attr("textin", "read", "Response")                # 0.7 name
    _system = None                      # the System that answered (reports; see the class docs)
    _heads = None                       # its answer heads (the audit)

    def __post_init__(self):
        if hasattr(self.trace, "records") and isinstance(self.results, dict):
            self.trace.answers = self.results      # replay(system) checks that these are the answers the trace gives

    def __getitem__(self, q):
        return self.results[q]

    @property
    def confidence(self):
        """Overall confidence: the probability that every answered question is right — the product of the answers'
        confidences (errors taken as independent, so it is conservative when answers agree through constraints). Abstained
        questions are left out; see `complete`. 1.0 when nothing was answered by a model or a learned part."""
        import math
        return math.prod(r.confidence for r in self.results.values() if r.status != "abstain")

    @property
    def complete(self):
        """Did every question get an answer (none abstained)?"""
        return all(r.status != "abstain" for r in self.results.values())

    @property
    def weakest(self):
        """(question, confidence) of the least confident answer, or None when every question abstained."""
        got = [(q, r.confidence) for q, r in self.results.items() if r.status != "abstain"]
        return min(got, key=lambda t: t[1]) if got else None

    @property
    def not_stated(self):
        """Questions answered "not stated" (solvi.Unknown): answered — they count in `confidence` and `complete` — but
        the text does not state the value."""
        from .core import Unknown
        return [q for q, r in self.results.items() if r.answer is Unknown and r.status != "abstain"]

    @property
    def overall(self):
        """The whole response in one line of data: confidence, weakest answer, answered / abstained, complete, feasible;
        the questions answered "not stated"; and per answer kind the answered count and the product of their confidences."""
        ab = [q for q, r in self.results.items() if r.status == "abstain"]
        kinds = {}
        for r in self.results.values():
            if r.status != "abstain":
                k = kinds.setdefault(r.kind or "?", [0, 1.0])
                k[0] += 1
                k[1] *= r.confidence
        return {"confidence": self.confidence, "weakest": self.weakest, "answered": len(self.results) - len(ab),
                "abstained": ab, "complete": not ab, "feasible": self.feasible, "not_stated": self.not_stated,
                "by_kind": {k: {"answered": n, "confidence": c} for k, (n, c) in sorted(kinds.items())}}

    def to_dict(self):
        """model_dump(mode="json"): answers, flow, trace, values, safeguards as JSON-ready data (see Serial)."""
        return self.model_dump("json")

    lang = "en"                         # the language of rendering (System(lang=...)); not a field: never serialized or hashed

    def signature(self, alg="syndrome"):
        """The signature of this response's trace (solvi.signature.sign: the input, then every record) — a few numbers to
        keep next to the stored id; solvi.signature.locate(res, sig) later names the one record that changed."""
        from .signature import sign
        return sign(self, alg)

    computed_state = _deprecate.removed_attr("computed_state", "state_text()", "Response")
    computed_state_text = _deprecate.removed_attr("computed_state_text()", "state_text(lang)", "Response")

    def state_text(self, lang=None):
        """The computed state for people: each computed fact with its value; a quote's offsets; for anything not
        computed by plain code, its provenance (quoted by a model, decided, learned) and the model; errors, including
        rejected (ungrounded) model outputs — in a language (solvi.i18n; default: the System's). The facts as data:
        `res.values`."""
        from . import i18n
        lang = i18n.check(self.lang if lang is None else lang)
        t = i18n.t
        lines = []
        for r in self.trace.records:
            if r.kind in ("rule", "head"):
                continue
            v = "—" if r.value is MISSING else repr(r.value)
            extra = t("cs.quote", lang, s=r.quote[0], e=r.quote[1]) if r.quote else ""
            if r.confidence < 1:
                extra += t("cs.confidence", lang, c=f"{r.confidence:.2f}")
            if r.origin not in ("computed", "quoted"):
                extra += f"   {i18n.provenance(r.origin, lang)}"
                if r.probs:
                    extra += " (" + ", ".join(f"{k} {float(p):.2f}" for k, p in sorted(r.probs.items(), key=lambda t_: -t_[1])[:3]) + ")"
            if r.model is not None:
                extra += t("cs.model", lang, m=f"{r.model['type']} {r.model['id']} #{r.model['fp'][:8]}")
            if r.error:
                extra += t("cs.error", lang, err=i18n.msg(r.error, lang))
            lines.append(f"{r.name:24s} = {v}{extra}")
        return "\n".join(lines)

    def audit(self, question=None, lang=None):
        """What each answer rests on and which safeguards fired: given inputs → computed facts → quotes (offsets and quoted
        text) → model decisions (model, probabilities) → learned parts → checks, rule, constraints → answer; plus the share
        of the support that is deterministic. `print(res.audit())`, or `res.audit("q").to_dict()` for data.
        lang: the language str() renders in (solvi.i18n; default: the System's, "en"); the data is the same in every one.
        → an Audit of every answer, or the AnswerAudit of one question when `question` is given."""
        from .audit import build
        a = build(self, question, lang=self.lang if lang is None else lang)
        return a[question] if isinstance(question, str) else a

    def report(self, format="md", question=None, system=None, replay="trusted"):
        """A human-readable report of this decision for an auditor or a customer: each answer, what it rests on, the quotes
        highlighted in the source text with their offsets, the safeguards that fired, the guarantee line, the models'
        fingerprints, the trace's hashes and the replay status. format: "md" (Markdown), "html" (one self-contained page,
        every value escaped) or "data" (a dict). replay: "trusted" (default: deterministic steps re-run, model outputs
        verified from the record — no model is called), "full" (models re-run too) or False. See solvi.report."""
        from .report import decision, render
        return render(decision(self, question, system, replay), format)


__all__ = ["Response"]
