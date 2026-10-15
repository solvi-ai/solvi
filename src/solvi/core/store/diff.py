"""solvi diff and shadow mode: which decisions change with a new catalog or model, and why.

diff(storage, system) re-runs stored decisions (a TraceStorage) with another System — a changed rule, a new model — and
reports, per decision, the questions whose answer, status, safeguard or confidence changes, with the steps that changed
the answer — those on a path of differing outputs from the answer back, where a difference starts — and why each
differs (the part's code or declarations changed, its model changed, it is new, or none of these: a non-deterministic
or external source). A step that was added or changed and gives nothing different downstream is not named.

Shadow(current, candidate) answers with the current system and runs the candidate on the same inputs, storing the
candidate's response with its differences; the answer returned is always the current system's. Labelled later
(`shadow.label(rid, truth)`), the paired outcomes feed an anytime-valid harm stop, overall and per slice."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ... import _deprecate


# --- comparing two responses to the same input
def _qstate(r):
    from . import plain
    return {"answer": plain(r.answer), "status": r.status, "guard": r.guard, "confidence": round(float(r.confidence), 6)}


def _events(resp, q):
    return sorted({(e["kind"], e["fact"]) for e in resp.safeguards or () if q in (e.get("questions") or ())})


def _deps(resp, q):
    """The facts an answer rests on in a response: from the answer (and every hard check that ran for its question, the
    deciding one included) back through the recorded inputs."""
    by = {}
    for r in resp.trace.records:
        by.setdefault(r.name, []).append(r)
    res = resp.results.get(q)
    todo = ["answer:" + q] + ([res.source] if res is not None and res.guard == "hard_check" and res.source in by else [])
    hard = {st.part.name for st in resp.flow.steps if st.part.kind == "check" and st.part.hard}
    todo += [c for c in (getattr(resp.flow, "per_question", None) or {}).get(q) or () if c in hard and c in by
             and c not in todo]
    seen = set(todo)
    while todo:
        n = todo.pop()
        for rec in by.get(n, ()):
            for x in rec.inputs:
                if x not in seen:
                    seen.add(x)
                    todo.append(x)
    return seen


def _out(r):
    """A record's output for comparison: (value hash or "missing", error, producer, quote)."""
    from ..runtime import MISSING, vhash
    return ("missing" if r.value is MISSING else vhash(r.value), r.error, r.producer,
            None if r.quote is None else list(r.quote))


def _show(r):
    from ..runtime import MISSING, srepr
    if r is None:
        return "not run"
    if r.value is MISSING:
        return f"no value ({r.error})" if r.error else "no value"
    s = srepr(r.value)
    return (s if len(s) <= 60 else s[:59] + "…") + (f" by {r.producer}" if r.producer else "")


def _model_of(r):
    """The model behind a record: the one used, else the one model-backed producer that ran and was rejected (a fact with
    alternative producers whose answer came from a later producer)."""
    if r is None:
        return {}
    if r.model:
        return r.model
    tm = r.tried_models or {}
    return dict(next(iter(tm.values()))) if len(tm) == 1 else {}


def _why(name, a, b, old, new):
    """One differing step as data: {"step", "name", "old", "new", "why"} — why its output differs between two responses."""
    ofp, nfp = old.trace.fingerprint or {}, new.trace.fingerprint or {}
    why = []
    op, np_ = (ofp.get("parts") or {}).get(name), (nfp.get("parts") or {}).get(name)
    if op and np_ and op != np_:
        why.append("its code or declarations changed")
    am, bm = _model_of(a), _model_of(b)
    if am.get("fp") != bm.get("fp") and (am or bm):
        why.append(f"its model changed (#{am.get('fp', '—')} → #{bm.get('fp', '—')})")
    elif am and a is not None and b is not None and (a.model is None) != (b.model is None):
        why.append(f"its model (#{am.get('fp')}) was {'used' if a.model is not None else 'rejected'} and is now "
                   f"{'used' if b.model is not None else 'rejected'}")
    if a is not None and b is not None:
        ch = sorted(x for x in set(a.inputs) | set(b.inputs) if a.inputs.get(x) != b.inputs.get(x))
        if ch:
            why.append("inputs changed: " + ", ".join(ch))
    if a is None:
        why.append("new step (not run before)")
    elif b is None:
        why.append("no longer runs" + (f" ({dict(new.trace.skipped).get(name)})" if name in dict(new.trace.skipped) else ""))
    if not why:
        if not op or not np_:
            why.append("the part's code at the time was not recorded")
        else:
            why.append("same code, model and inputs: a non-deterministic or external source")
    return {"step": (b or a).step, "name": name, "old": _show(a), "new": _show(b), "why": "; ".join(why)}


def _records(resp):
    by = {}
    for r in resp.trace.records:
        by.setdefault(r.name, r)
    return by


def causes(old, new, question):
    """The steps that changed `question`'s answer between two responses to the same input, in flow order → [{"step",
    "name", "old", "new", "why"}]. From the answer step (and a hard check that decided it) back through the recorded
    inputs, only through steps whose output differs: a step that gives the same output as before stops the walk — what
    changed above it did not reach the answer — so a step that was added (or changed) and changes nothing downstream
    is not a cause. Of the steps on such a path, the causes are those where a difference starts: its own code,
    declarations or model changed, it is new or no longer runs, or none of what it reads differs. [] when the answer
    step itself gives the same output (see first_difference for what is reported then)."""
    olds, news = _records(old), _records(new)
    ofp, nfp = (old.trace.fingerprint or {}).get("parts") or {}, (new.trace.fingerprint or {}).get("parts") or {}

    def differs(n):
        a, b = olds.get(n), news.get(n)
        return (a is None) != (b is None) or (a is not None and _out(a) != _out(b))
    roots = ["answer:" + question]
    for resp, by in ((old, olds), (new, news)):
        res = resp.results.get(question)
        if res is not None and res.guard == "hard_check" and res.source in by:
            roots.append(res.source)
    path, todo = set(), [n for n in dict.fromkeys(roots) if differs(n)]
    reads = {}
    while todo:
        n = todo.pop()
        if n in path:
            continue
        path.add(n)
        reads[n] = [x for r in (olds.get(n), news.get(n)) if r is not None for x in r.inputs if differs(x)]
        todo.extend(reads[n])
    order = [r.name for r in new.trace.records] + [r.name for r in old.trace.records if r.name not in news]
    out = []
    for n in dict.fromkeys(order):
        if n not in path:
            continue
        a, b = olds.get(n), news.get(n)
        own = a is None or b is None or (ofp.get(n) and nfp.get(n) and ofp[n] != nfp[n]) \
            or _model_of(a).get("fp") != _model_of(b).get("fp") or (a.model is None) != (b.model is None)
        if own or not reads[n]:
            out.append(_why(n, a, b, old, new))
    return out


def first_difference(old, new, question=None):
    """With a question: the first of the steps (in flow order) that changed its answer between two responses to the same
    input — see causes; when the answer step itself gives the same output (a safeguard or the confidence changed), the
    first step among the facts the answer rests on whose output differs. Without a question: the first step whose
    output differs at all. → {"step", "name", "old", "new", "why"} or None when every step gives the same output."""
    if question is not None:
        found = causes(old, new, question)
        if found:
            return found[0]
    olds, news = _records(old), _records(new)
    scope = None if question is None else (_deps(old, question) | _deps(new, question))
    order = [r.name for r in new.trace.records] + [r.name for r in old.trace.records if r.name not in news]
    ofp, nfp = old.trace.fingerprint or {}, new.trace.fingerprint or {}
    for name in dict.fromkeys(order):
        if scope is not None and name not in scope:
            continue
        a, b = olds.get(name), news.get(name)
        if a is not None and b is not None and _out(a) == _out(b):
            continue
        return _why(name, a, b, old, new)
    if question is not None:                          # the facts agree: the answer step itself or a joint repair
        ro, rn = old.results.get(question), new.results.get(question)
        if ro is not None and rn is not None and (ro.repaired or rn.repaired):
            return {"step": None, "name": "answer:" + question, "old": ro.why, "new": rn.why,
                    "why": "a constraint between answers repaired it"}
        if ofp.get("questions") and nfp.get("questions") and ofp["questions"] != nfp["questions"]:
            return {"step": None, "name": "question:" + question, "old": "", "new": "",
                    "why": "every step agrees; the questions changed (answer type, min_confidence, calibration)"}
    return None


def compare(old, new, confidence=0.01):
    """The differences between two responses to the same input → {question: {"old", "new", "changed", "first_step",
    "causes"}} for the questions whose answer, status, safeguard (the guard that settled it, or the safeguard events
    concerning it) differ — or whose confidence moved by more than `confidence` (None: ignore confidence). "causes": the
    steps that changed the answer (see causes), "first_step" the first of them in flow order (see first_difference).
    Questions asked in one response only are listed with the other side None."""
    out = {}
    for q in list(dict.fromkeys(list(old.results) + list(new.results))):
        ro, rn = old.results.get(q), new.results.get(q)
        if ro is None or rn is None:
            out[q] = {"old": None if ro is None else _qstate(ro), "new": None if rn is None else _qstate(rn),
                      "changed": ["question"], "first_step": None, "causes": []}
            continue
        a, b = _qstate(ro), _qstate(rn)
        changed = [k for k in ("answer", "status", "guard") if a[k] != b[k]]
        ea, eb = _events(old, q), _events(new, q)
        if ea != eb:
            changed.append("safeguards")
            a["safeguards"], b["safeguards"] = [list(e) for e in ea], [list(e) for e in eb]
        if confidence is not None and abs(a["confidence"] - b["confidence"]) > confidence:
            changed.append("confidence")
        if changed:
            found = causes(old, new, q)
            out[q] = {"old": a, "new": b, "changed": changed, "first_step": found[0] if found else
                      first_difference(old, new, q), "causes": found}
    return out


# --- solvi diff
@dataclass
class DiffReport:
    """What changes when stored decisions are re-run with another system. A compact stored decision (see
    TraceStorage record=) is compared on its answers only: its change has "record": "compact" and no causes."""
    total: int = 0                                # stored decisions re-run
    changed: list = field(default_factory=list)   # [{"id", "seq", "time", "questions": {question: change}}]
    errors: list = field(default_factory=list)    # [{"id", "seq", "error"}] decisions that could not be re-run
    catalog: dict = field(default_factory=dict)   # {"new": fp, "stored": {fp: count}, "changed_parts", "added_parts"}
    only: list | None = None                      # diff(only=): the decisions the change is allowed to change
    extraneous: list = field(default_factory=list)   # ids of the changed decisions not in `only` (in `changed` order)
    exact: bool = False                           # `only` given, nothing outside it changed, every decision re-ran

    @property
    def ok(self):
        """No answer changed and every decision re-ran."""
        return not self.changed and not self.errors

    def by_question(self):
        """question → {"changed": n, "fields": {field: n}, "transitions": {"old → new": n}}."""
        out = {}
        for c in self.changed:
            for q, ch in c["questions"].items():
                s = out.setdefault(q, {"changed": 0, "fields": {}, "transitions": {}})
                s["changed"] += 1
                for f in ch["changed"]:
                    s["fields"][f] = s["fields"].get(f, 0) + 1
                if "answer" in ch["changed"] or "status" in ch["changed"]:
                    t = f"{_ans(ch['old'])} → {_ans(ch['new'])}"
                    s["transitions"][t] = s["transitions"].get(t, 0) + 1
        return out

    def to_dict(self):
        return {"total": self.total, "changed": self.changed, "errors": self.errors, "catalog": self.catalog,
                "by_question": self.by_question(), "only": self.only, "extraneous": self.extraneous, "exact": self.exact}

    def __str__(self):
        n = len(self.changed)
        lines = [f"diff: {self.total} stored decision(s) re-run, {n} changed, {len(self.errors)} could not be re-run"]
        cp = self.catalog.get("changed_parts")
        if cp:
            lines.append("  parts changed since they were stored: " + ", ".join(cp))
        ap = self.catalog.get("added_parts")
        if ap:
            lines.append("  parts that ran now and not then: " + ", ".join(ap))
        if self.only is not None:
            lines.append("  exact: only the allowed decisions change" if self.exact else
                         f"  {len(self.extraneous)} decision(s) changed outside the allowed ones: "
                         + ", ".join(map(str, self.extraneous[:10])) + (" …" if len(self.extraneous) > 10 else ""))
        for q, s in sorted(self.by_question().items()):
            lines.append(f"  {q}: {s['changed']} changed (" + ", ".join(f"{k} {v}" for k, v in s["fields"].items()) + ")")
            for t, k in sorted(s["transitions"].items(), key=lambda kv: -kv[1]):
                lines.append(f"      {t}: {k}")
        for c in self.changed:
            lines.append(f"- {c['id']} (#{c['seq']})")
            for q, ch in c["questions"].items():
                lines.append(f"    {q}: {_ans(ch['old'])} → {_ans(ch['new'])}  [{', '.join(ch['changed'])}]")
                fs = ch.get("first_step")
                if fs:
                    lines.append(f"      first difference: {fs['name']}: {fs['old']} → {fs['new']} — {fs['why']}")
                for c in (ch.get("causes") or [])[1:]:
                    lines.append(f"      and: {c['name']}: {c['old']} → {c['new']} — {c['why']}")
        for e in self.errors:
            lines.append(f"! {e['id']} (#{e['seq']}): {e['error']}")
        return "\n".join(lines)


def _ans(s):
    if s is None:
        return "not asked"
    a = "—" if s["answer"] is None else repr(s["answer"])
    return a if s["status"] == "ok" else f"{a} ({s['status']})"


COMPACT_STEP = {"step": None, "name": "—", "old": "", "new": "",
                "why": "a compact record keeps no intermediate steps: only the answers are compared"}


def _compact_change(s, system, confidence):
    """A compact stored decision against `system` on its recorded input: its kept answers (answer, status, guard,
    confidence) against the new ones → compare's {question: change}, with no causes (the steps are not kept)."""
    from ..response import Response
    from ..runtime import Result, Trace
    from ..schema import load
    from . import view
    v = view(s.data)
    tr = load(Trace, v["trace"], system)
    names = [q for q in v["results"] if q in system.questions]
    new = system.ask(dict(tr.init), names, store=False)
    results = {q: load(Result, r) for q, r in v["results"].items()}
    old = Response(results, None, tr, {}, 0.0, safeguards=list(v.get("safeguards") or []))
    out = {}
    for q in dict.fromkeys(list(old.results) + list(new.results)):
        ro, rn = old.results.get(q), new.results.get(q)
        if ro is None or rn is None:
            out[q] = {"old": None if ro is None else _qstate(ro), "new": None if rn is None else _qstate(rn),
                      "changed": ["question"], "first_step": None, "causes": []}
            continue
        a, b = _qstate(ro), _qstate(rn)
        changed = [k for k in ("answer", "status", "guard") if a[k] != b[k]]
        if _events(old, q) != _events(new, q):
            changed.append("safeguards")
        if confidence is not None and abs(a["confidence"] - b["confidence"]) > confidence:
            changed.append("confidence")
        if changed:
            out[q] = {"old": a, "new": b, "changed": changed, "first_step": dict(COMPACT_STEP), "causes": []}
    return out


def diff(storage, system, confidence=0.01, limit=None, only=None, **filters):
    """Re-run stored decisions with `system` (e.g. a new catalog or model) and report what changes → DiffReport.
    Each stored decision is loaded (typed values restored with `system`), its recorded input is asked again for the same
    questions (store=False: nothing is written to the system's own storage), and the two responses are compared (see
    compare). A stored decision whose input did not come back as it was (see solvi.core.schema: an untyped enum or object, an
    untyped date stored by solvi ≤ 0.7.1) is listed under `errors` — "could not be re-run" — not as a changed one. filters: TraceStorage.query filters (question=, since=, ...) to pick the decisions; limit: at most this many.
    only: the ids of the stored decisions the change is meant to change (an exception for one corrected decision) —
    `extraneous` lists the ids of the changed decisions outside it, `exact` is True when there are none and every
    decision re-ran (a decision that could not be re-run, listed in `errors`, may have changed: then `exact` is False).
    A cause whose part (in `system`) cites what it implements (`clause=` / `cite=` on its decorator) carries "cites":
    [{"clause", "quote"}] — the chain clause → part → decision.
    The system's learned parts may learn from these asks as from any other (System(learn=...))."""
    from ..provenance import catalog_fingerprint
    rep = DiffReport()
    now = catalog_fingerprint(system.catalog)
    rep.catalog = {"new": now["fp"], "stored": {}, "changed_parts": [], "added_parts": []}
    changed_parts, added_parts = set(), set()
    rows = storage.query(**filters) if filters else storage.iter()
    for i, s in enumerate(rows):
        if limit is not None and i >= limit:
            break
        rep.total += 1
        if getattr(s, "compact", False):
            try:
                ch = _compact_change(s, system, confidence)
            except Exception as e:  # noqa: BLE001
                rep.errors.append({"id": s.id, "seq": s.seq, "error": f"{type(e).__name__}: {e}"})
                continue
            if ch:
                rep.changed.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": ch, "record": "compact"})
            continue
        try:
            old = s.response(system)
            fp = old.trace.fingerprint or {}
            key = fp.get("catalog") or "unrecorded"
            rep.catalog["stored"][key] = rep.catalog["stored"].get(key, 0) + 1
            changed_parts |= {n for n, h in (fp.get("parts") or {}).items() if now["parts"].get(n) != h}
            names = [q for q in old.results if q in system.questions]
            lost = sorted(k for k in getattr(old.trace, "unrestored", None) or () if k in old.trace.init)
            if lost:                                  # a re-run on text where the decision read a date is not a change
                raise ValueError("the stored input was not restored (" + ", ".join(lost) + ": stored without a type "
                                 "the dump restores, and none is declared) — declare the types (System(input_model=...) or "
                                 "annotations) to re-run it")
            new = system.ask(dict(old.trace.init), names, store=False)
        except Exception as e:  # noqa: BLE001
            rep.errors.append({"id": s.id, "seq": s.seq, "error": f"{type(e).__name__}: {e}"})
            continue
        was = {r.name for r in old.trace.records} | {n for n, _ in old.trace.skipped}
        added_parts |= {r.name for r in new.trace.records if r.name not in was and r.kind not in ("head", "plan", "textin", "guard", "then")}
        ch = compare(old, new, confidence)
        if ch:
            _add_cites(ch, system.catalog, new)
            rep.changed.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": ch})
    rep.catalog["changed_parts"] = sorted(changed_parts)
    rep.catalog["added_parts"] = sorted(added_parts)
    if only is not None:
        allowed = set(only)
        rep.only = list(only)
        rep.extraneous = [i for i in dict.fromkeys(c["id"] for c in rep.changed) if i not in allowed]
        rep.exact = not rep.extraneous and not rep.errors
    return rep


def part_cites(cat, name, producer=None):
    """The citations of the part behind a step (Part.cites: the decorators' clause= / cite=) → [{"clause", "quote"}] or
    None. name: a step's name (a fact, or "answer:<question>" for a rule); producer: the alternative producer that gave
    a grouped fact its value (else the citations of all its producers)."""
    p = cat.rules.get(name[len("answer:"):]) if name.startswith("answer:") else cat.parts.get(name)
    if p is None:
        return None
    alts = p.alternatives or []
    chosen = [a for a in alts if a.name == producer] or alts or [p]
    cites = [c for a in chosen for c in (a.cites or ())]
    return [{"clause": c, "quote": q} for c, q in cites] or None


def _add_cites(changes, cat, new):
    """Each cause (and first step) of compare's changes gains "cites" when its part has citations."""
    recs = _records(new)
    for d in changes.values():
        for c in list(d.get("causes") or []) + ([d["first_step"]] if d.get("first_step") else []):
            r = recs.get(c.get("name") or "")
            cites = part_cites(cat, c.get("name") or "", getattr(r, "producer", None))
            if cites:
                c["cites"] = cites


# --- shadow mode
_BETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


class _PairedHarm:
    """A test martingale against "the candidate is the wrong one in at most half of the discordant pairs" (one where
    exactly one of current / candidate is right). Per pair x = 1 when the candidate is the wrong one; each bet b
    multiplies its wealth by 1 + b·(2x − 1), whose mean is ≤ 1 under that null, and the e-value is the mean wealth over
    the bets in _BETS (a mixture of test martingales is one). Ville: the chance it ever reaches 1/alpha is ≤ alpha, so
    it may be checked after every label. update(x) → True when this pair takes it to 1/alpha (it fires once)."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.log_w = [0.0] * len(_BETS)
        self.n = self.harm = 0
        self.fired = False

    @property
    def e_value(self):
        return sum(math.exp(w) for w in self.log_w) / len(_BETS)

    def update(self, cand_wrong):
        self.n += 1
        self.harm += bool(cand_wrong)
        sign = 1.0 if cand_wrong else -1.0
        self.log_w = [w + math.log1p(b * sign) for w, b in zip(self.log_w, _BETS)]
        if not self.fired and self.e_value >= 1.0 / self.alpha:
            self.fired = True
            return True
        return False

    def to_dict(self):
        return {"discordant": self.n, "candidate_wrong": self.harm, "e_value": round(self.e_value, 4),
                "threshold": round(1.0 / self.alpha, 4), "halted": self.fired}


class Shadow:
    """Answer with `current`, run `candidate` on the same input, keep what differs; with labels, stop a harmful candidate.

        shadow = Shadow(current, candidate, storage=SQLiteStorage("shadow.db"), slice_of=lambda st: st["region"])
        res = shadow.ask(state, rid="req-17")  # current's response, exactly as current.ask(state) (saved to current's storage)
        shadow.label("req-17", truth)          # when the outcome is known → None, or the event that halted the candidate

    The candidate's response is saved to `storage` (when given) with meta {"shadow_of": the current response's stored id,
    "current_catalog", "diff": compare(current, candidate)}; the candidate never writes to its own storage. A candidate that
    fails is counted and recorded, never raised. stats: {"asks", "agree", "differ", "errors"}; changed: the last `keep`
    differing inputs [{"stored_id", "shadow_id", "questions"}]; summary() as text.

    Harm stop. Both systems answer every input, so each labelled one is a paired comparison. Among the labelled
    question answers where exactly one of the two is right, a test martingale bets that the candidate is the wrong one
    (see _PairedHarm): overall at alpha/2 and per slice at alpha/(2·max_slices) — the chance of ever halting a candidate
    that is no more often the wrong one is ≤ alpha, at any stopping time, so checking after every label is allowed. The
    first test to reach its level halts: `halted` holds the event {"event": "halted", "scope": "overall" | "slice",
    "slice", "question", "why", "e_value", "labels"}; the shadow keeps running and answering with `current` (switching
    is yours); later labels still count but nothing halts twice. A halt needs evidence: harm on a tenth of the discordant
    pairs takes hundreds of them. The outcome must not depend on which answer was served (no feedback loop).

    slice_of: fn(init_state) → a slice name (None: one slice). The first max_slices − 1 names have their own test; the
    rest share one ("(other)"). alpha: the harm stop's error (default 0.05). correct: fn(answer, truth) → bool (default
    equality; a question left unanswered counts as not right). max_pending: the answered, not yet labelled requests kept
    (the oldest go first). rid: given to ask(), else the current response's stored_id."""

    def __init__(self, current, candidate, storage=None, confidence=0.01, keep=1000, *, slice_of=None, alpha=0.05,
                 max_slices=20, correct=None, max_pending=10_000):
        from . import open_storage
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha is a probability strictly between 0 and 1")
        if int(max_slices) < 1:
            raise ValueError("max_slices is at least 1")
        self.current, self.candidate = current, candidate
        self.storage = open_storage(storage, candidate)
        if self.storage is not None and self.storage.catalog is None:
            self.storage.catalog = candidate
        self.confidence, self.keep = confidence, keep
        self.stats = {"asks": 0, "agree": 0, "differ": 0, "errors": 0}
        self.changed, self.errors = [], []
        self.last = None                              # (candidate response or None, diff) of the last ask
        self.slice_of, self.correct = slice_of, correct or (lambda answer, truth: answer == truth)
        self.alpha, self.max_slices, self.max_pending = float(alpha), int(max_slices), int(max_pending)
        self._overall = _PairedHarm(self.alpha / 2)
        self._slices = {}
        self._pending = {}                            # rid → (slice, {question: (current answer, candidate answer)})
        self.labels = {"labels": 0, "discordant": 0, "current_wrong": 0, "candidate_wrong": 0}
        self.halted = None

    @property
    def harm(self):
        """The harm stop's state: {"overall": {...}, "slices": {name: {...}}, "halted": the event or None}."""
        return {"overall": self._overall.to_dict(), "slices": {k: v.to_dict() for k, v in self._slices.items()},
                "halted": self.halted}

    def _remember(self, rid, init_state, resp, cand):
        if rid is None:
            return
        sl = None
        if self.slice_of is not None:
            try:
                sl = self.slice_of(init_state)
            except Exception as e:  # noqa: BLE001
                sl = f"(slice_of failed: {type(e).__name__})"
        pairs = {q: (r.answer, cand.results[q].answer if q in cand.results else None) for q, r in resp.results.items()}
        self._pending.pop(rid, None)
        self._pending[rid] = (sl, pairs)
        while len(self._pending) > self.max_pending:
            del self._pending[next(iter(self._pending))]

    def _slice_test(self, sl):
        if sl not in self._slices:
            if len(self._slices) >= self.max_slices - 1 and sl != "(other)":
                sl = "(other)"
            self._slices.setdefault(sl, _PairedHarm(self.alpha / (2 * self.max_slices)))
        return sl, self._slices[sl]

    def label(self, rid, truth, question=None):
        """The outcome of an answered request: truth for `question`, or {question: truth}. → None, or the event when this
        label halts the candidate. A request is labelled once; an unknown (or forgotten) rid gives None."""
        if rid is None:
            raise ValueError("label needs the rid given to ask() (or the current response's stored_id)")
        if rid not in self._pending:
            return None
        sl, pairs = self._pending[rid]
        if isinstance(truth, dict):
            truths = truth
        elif question is not None:
            truths = {question: truth}
        elif len(pairs) == 1:
            truths = {next(iter(pairs)): truth}
        else:
            raise ValueError(f"the request answered {sorted(pairs)}: name the question or pass {{question: truth}}")
        del self._pending[rid]
        event = None
        for q, t in truths.items():
            if q not in pairs:
                continue
            a_cur, a_cand = pairs[q]
            cw, kw = not self.correct(a_cur, t), not self.correct(a_cand, t)
            self.labels["labels"] += 1
            self.labels["current_wrong"] += cw
            self.labels["candidate_wrong"] += kw
            if cw == kw:
                continue
            self.labels["discordant"] += 1
            tests = [("overall", None, self._overall)]
            if self.slice_of is not None:
                name, test = self._slice_test(sl)
                tests.append(("slice", name, test))
            for scope, name, test in tests:
                if test.update(kw) and self.halted is None:
                    where = "overall" if scope == "overall" else f"in slice {name!r}"
                    event = self.halted = {
                        "event": "halted", "scope": scope, "slice": name, "question": q,
                        "why": f"candidate worse {where}: e-value {test.e_value:.1f} ≥ {1 / test.alpha:.0f} "
                               f"({test.harm} of {test.n} discordant pairs the candidate's error)",
                        "e_value": test.e_value, "labels": self.labels["labels"]}
        return event

    @_deprecate.removed_kwargs(names="questions")
    def ask(self, init_state, questions=None, *, rid=None, **kw):
        """current.ask(init_state, questions, **kw), the candidate run beside it. rid: the key a later label() uses
        (default: the current response's stored_id; with neither, the request cannot be labelled)."""
        resp = self.current.ask(init_state, questions, **kw)
        self.stats["asks"] += 1
        try:
            cand = self.candidate.ask(init_state, questions, store=False)
            ch = compare(resp, cand, self.confidence)
        except Exception as e:  # noqa: BLE001
            self.stats["errors"] += 1
            self.errors = (self.errors + [{"stored_id": resp.stored_id, "error": f"{type(e).__name__}: {e}"}])[-self.keep:]
            self.last = (None, None)
            return resp
        self.stats["differ" if ch else "agree"] += 1
        sid = None
        if self.storage is not None:
            try:
                sid = self.storage.save(cand, meta={"shadow_of": resp.stored_id,
                                                    "current_catalog": (resp.trace.fingerprint or {}).get("catalog"),
                                                    "diff": ch})
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                self.errors = (self.errors + [{"stored_id": resp.stored_id, "error": f"saving: {type(e).__name__}: {e}"}]
                               )[-self.keep:]
        if ch:
            self.changed = (self.changed + [{"stored_id": resp.stored_id, "shadow_id": sid, "questions": ch}])[-self.keep:]
        self._remember(resp.stored_id if rid is None else rid, init_state, resp, cand)
        self.last = (cand, ch)
        return resp

    report = _deprecate.removed_attr("report()", "summary()", "Shadow")

    def summary(self):
        """What the shadow saw, as text: asks, agreements, differences per question and answer, failures."""
        st = self.stats
        lines = [f"shadow: {st['asks']} ask(s), candidate agrees on {st['agree']}, differs on {st['differ']}, "
                 f"failed on {st['errors']}"]
        per = {}
        for c in self.changed:
            for q, ch in c["questions"].items():
                t = f"{q}: {_ans(ch['old'])} → {_ans(ch['new'])}"
                per[t] = per.get(t, 0) + 1
        for t, n in sorted(per.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {t}  ×{n}")
        lb = self.labels
        if lb["labels"]:
            lines.append(f"labels: {lb['labels']}, current wrong {lb['current_wrong']}, candidate wrong "
                         f"{lb['candidate_wrong']}, discordant {lb['discordant']}; harm e-value "
                         f"{self._overall.e_value:.2f} (halts at {1 / self._overall.alpha:.0f})")
        if self.halted:
            lines.append("halted: " + self.halted["why"])
        return "\n".join(lines)


__all__ = ["causes", "compare", "diff", "DiffReport", "part_cites", "Shadow"]
