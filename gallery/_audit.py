"""The audit check every gallery runner applies to every response (solvi 0.4.0 `Response.audit()`).

`check(res, system)` builds the audit, asserts its invariants and returns a small summary; `line(summary)` prints it in one
line; `Tally` adds the summaries of a run up. The invariants:

- every quote lies inside its source text, and the audit shows exactly `text[start:end]`;
- a quote from a part that must be literal (model-backed, or `exact=True`) is literally the text at its offsets;
- a model decision is one of its options; an answer with status ok/forced is a valid answer of its type (a subset for
  multi-label, one of the levels for ordinal); an abstention has no answer;
- a forced answer names the hard check that decided it, and a "hard check decided" safeguard fired for it;
- a repaired answer has a "constraint repair" safeguard; when the response is feasible every constraint shown is satisfied;
- the support counts add up to the items listed, and a catalog without models or learned parts is 100% deterministic
  with no model outputs.

Imported by the runners (`sys.path` gets this folder); not part of the playground presets."""
from __future__ import annotations

import json
from collections import Counter

from solvi.audit import LABEL


def _strict(catalog, fact, producer):
    part = catalog.parts.get(fact)
    if part is not None and part.alternatives is not None and producer:
        part = catalog.alternative(fact, producer)
    return part is not None and part.strict()


def _uses_models(catalog, system):
    parts = [a for p in catalog.parts.values() for a in (p.alternatives or [p])] + list(catalog.rules.values())
    return bool(system.heads) or any(p.model is not None or p.provenance in ("learned", "decided", "proposed")
                                     for p in parts)


def check(res, system):
    """Assert the audit's invariants for one response → summary dict."""
    cat, audit = system.catalog, res.audit()
    init = res.trace.init
    records = {r.name: r for r in res.trace.records if r.kind != "head"}
    kinds = Counter()
    seen = set()                                      # an event on a fact shared by several answers counts once
    items = det = 0
    for q, au in audit.answers.items():
        where = f"{q}: "
        for x in au.quoted:
            src = init.get(x["source"])
            assert isinstance(src, str) and 0 <= x["start"] <= x["end"] <= len(src), where + f"quote {x['name']} outside its text"
            if x["text"] is not None and len(src[x["start"]:x["end"]]) <= 60:
                assert x["text"] == src[x["start"]:x["end"]], where + f"audit shows {x['text']!r} for {x['name']}"
            r = records.get(x["name"])
            if not x.get("error") and x["value"] is not None and _strict(cat, x["name"], r.producer if r else None):
                assert x["match"] is True, where + f"{x['name']} = {x['value']!r} is not literally {x['text']!r}"
        for d in au.decided:
            if not d.get("error") and d["probs"]:
                assert d["value"] in d["probs"], where + f"decision {d['value']!r} outside its options"
        at = system.questions[q].answer
        if au.status in ("ok", "forced"):
            assert au.answer == at.normalize(au.answer), where + f"answer {au.answer!r} is not a valid {at.kind} answer"
        else:
            assert au.answer is None, where + "an abstention carries an answer"
        mine = Counter(e["kind"] for e in au.safeguards)
        if au.status == "forced":
            assert mine["hard_check"] and any(c["decides"] for c in au.checks), where + "forced without a deciding hard check"
        if res[q].repaired:
            assert mine["constraint_repair"], where + "repaired without a constraint-repair safeguard"
        if res.feasible:
            assert all(c["satisfied"] for c in au.constraints), where + "feasible response with a broken constraint"
        listed = (len(au.given) + len(au.computed) + len(au.quoted) + len(au.decided) + len(au.learned) + len(au.checks)
                  + (au.rule is not None))
        assert sum(au.counts.values()) == listed, where + f"support counts {au.counts} ≠ {listed} items listed"
        for e in au.safeguards:
            key = (e["kind"], e.get("fact"), e.get("detail"))
            if key not in seen:
                seen.add(key)
                kinds[e["kind"]] += 1
        items += sum(au.counts.values())
        det += au.deterministic
    if not _uses_models(cat, system):
        assert audit.model_outputs == 0 and det == items, "a catalog without models is not 100% deterministic"
    json.dumps(audit.to_dict(), default=str)                    # the audit is data: serializable as it is
    return {"answers": len(audit.answers), "items": items, "deterministic": det, "model_outputs": audit.model_outputs,
            "safeguards": kinds, "feasible": res.feasible, "violations": res.violations or []}


def line(s):
    share = s["deterministic"] / s["items"] if s["items"] else 1.0
    fired = ", ".join(f"{LABEL[k]} ×{v}" for k, v in sorted(s["safeguards"].items())) or "none fired"
    return (f"audit: {s['items']} support items, {share:.0%} deterministic, {s['model_outputs']} model outputs; "
            f"safeguards: {fired}" + ("" if s["feasible"] else f"; INFEASIBLE: {', '.join(s['violations'])}"))


class Tally:
    """Adds the per-case summaries up for the closing line of a run."""

    def __init__(self):
        self.n = self.items = self.det = self.models = 0
        self.kinds = Counter()

    def add(self, s):
        self.n += 1
        self.items += s["items"]
        self.det += s["deterministic"]
        self.models += s["model_outputs"]
        self.kinds.update(s["safeguards"])
        return s

    def __str__(self):
        fired = ", ".join(f"{LABEL[k]} ×{v}" for k, v in sorted(self.kinds.items())) or "none"
        return (f"audit invariants hold on {self.n}/{self.n} responses: {self.items} support items, "
                f"{self.det / max(self.items, 1):.0%} deterministic, {self.models} model outputs; safeguards fired: {fired}")
