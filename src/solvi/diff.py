"""solvi diff and shadow mode: which decisions change with a new catalog or model, and why.

diff(storage, system) re-runs stored decisions (a TraceStorage) with another System — a changed rule, a new model — and
reports, per decision, the questions whose answer, status, safeguard or confidence changes, with the steps that changed
the answer — those on a path of differing outputs from the answer back, where a difference starts — and why each
differs (the part's code or declarations changed, its model changed, it is new, or none of these: a non-deterministic
or external source). A step that was added or changed and gives nothing different downstream is not named.

Shadow(current, candidate) answers with the current system and runs the candidate on the same inputs, storing the
candidate's response with its differences; the answer returned is always the current system's."""
from __future__ import annotations

from dataclasses import dataclass, field


# --- comparing two responses to the same input
def _qstate(r):
    from .storage import plain
    return {"answer": plain(r.answer), "status": r.status, "guard": r.guard, "confidence": round(float(r.confidence), 6)}


def _events(resp, q):
    return sorted({(e["kind"], e["fact"]) for e in resp.safeguards or () if q in (e.get("questions") or ())})


def _deps(resp, q):
    """The facts an answer rests on in a response: from the answer (and a deciding hard check) back through the recorded
    inputs."""
    by = {}
    for r in resp.trace.records:
        by.setdefault(r.name, []).append(r)
    res = resp.results.get(q)
    todo = ["answer:" + q] + ([res.source] if res is not None and res.guard == "hard_check" and res.source in by else [])
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
    from .runtime import MISSING, vhash
    return ("missing" if r.value is MISSING else vhash(r.value), r.error, r.producer,
            None if r.quote is None else list(r.quote))


def _show(r):
    from .runtime import MISSING, srepr
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
    """What changes when stored decisions are re-run with another system."""
    total: int = 0                                # stored decisions re-run
    changed: list = field(default_factory=list)   # [{"id", "seq", "time", "questions": {question: change}}]
    errors: list = field(default_factory=list)    # [{"id", "seq", "error"}] decisions that could not be re-run
    catalog: dict = field(default_factory=dict)   # {"new": fp, "stored": {fp: count}, "changed_parts", "added_parts"}

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
                "by_question": self.by_question()}

    def __str__(self):
        n = len(self.changed)
        lines = [f"diff: {self.total} stored decision(s) re-run, {n} changed, {len(self.errors)} could not be re-run"]
        cp = self.catalog.get("changed_parts")
        if cp:
            lines.append("  parts changed since they were stored: " + ", ".join(cp))
        ap = self.catalog.get("added_parts")
        if ap:
            lines.append("  parts that ran now and not then: " + ", ".join(ap))
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


def diff(storage, system, confidence=0.01, limit=None, **filters):
    """Re-run stored decisions with `system` (e.g. a new catalog or model) and report what changes → DiffReport.
    Each stored decision is loaded (typed values restored with `system`), its recorded input is asked again for the same
    questions (store=False: nothing is written to the system's own storage), and the two responses are compared (see
    compare). A stored decision whose input did not come back as it was (see solvi.schema: an untyped enum or object, an
    untyped date stored by solvi ≤ 0.7.1) is listed under `errors` — "could not be re-run" — not as a changed one. filters: TraceStorage.query filters (question=, since=, ...) to pick the decisions; limit: at most this many.
    The system's learned parts may learn from these asks as from any other (System(learn=...))."""
    from .provenance import catalog_fingerprint
    rep = DiffReport()
    now = catalog_fingerprint(system.catalog)
    rep.catalog = {"new": now["fp"], "stored": {}, "changed_parts": [], "added_parts": []}
    changed_parts, added_parts = set(), set()
    rows = storage.query(**filters) if filters else storage.iter()
    for i, s in enumerate(rows):
        if limit is not None and i >= limit:
            break
        rep.total += 1
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
                                 "the dump restores, and none is declared) — declare the types (System(inputs=...) or "
                                 "annotations) to re-run it")
            new = system.ask(dict(old.trace.init), names, store=False)
        except Exception as e:  # noqa: BLE001
            rep.errors.append({"id": s.id, "seq": s.seq, "error": f"{type(e).__name__}: {e}"})
            continue
        was = {r.name for r in old.trace.records} | {n for n, _ in old.trace.skipped}
        added_parts |= {r.name for r in new.trace.records if r.name not in was and r.kind not in ("head", "plan", "textin")}
        ch = compare(old, new, confidence)
        if ch:
            rep.changed.append({"id": s.id, "seq": s.seq, "time": s.time, "questions": ch})
    rep.catalog["changed_parts"] = sorted(changed_parts)
    rep.catalog["added_parts"] = sorted(added_parts)
    return rep


# --- shadow mode
class Shadow:
    """Answer with `current`, run `candidate` on the same input, keep what differs.

        shadow = Shadow(current, candidate, storage=SQLiteStorage("shadow.db"))
        res = shadow.ask(state)        # current's response, exactly as current.ask(state) (saved to current's storage)

    The candidate's response is saved to `storage` (when given) with meta {"shadow_of": the current response's stored id,
    "current_catalog", "diff": compare(current, candidate)}; the candidate never writes to its own storage. A candidate that
    fails is counted and recorded, never raised. stats: {"asks", "agree", "differ", "errors"}; changed: the last `keep`
    differing inputs [{"stored_id", "shadow_id", "questions"}]; report() as text."""

    def __init__(self, current, candidate, storage=None, confidence=0.01, keep=1000):
        from .storage import open_storage
        self.current, self.candidate = current, candidate
        self.storage = open_storage(storage, candidate)
        if self.storage is not None and self.storage.catalog is None:
            self.storage.catalog = candidate
        self.confidence, self.keep = confidence, keep
        self.stats = {"asks": 0, "agree": 0, "differ": 0, "errors": 0}
        self.changed, self.errors = [], []
        self.last = None                              # (candidate response or None, diff) of the last ask

    def ask(self, init_state, names=None, **kw):
        resp = self.current.ask(init_state, names, **kw)
        self.stats["asks"] += 1
        try:
            cand = self.candidate.ask(init_state, names, store=False)
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
        self.last = (cand, ch)
        return resp

    def report(self):
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
        return "\n".join(lines)
