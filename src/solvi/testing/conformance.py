"""Conformance checks for the extension points of solvi.core: what the docs say you get for free, tested on your class.

    from solvi.testing.conformance import check_storage, check_slow_path

    def test_my_store(tmp_path):
        check_storage(lambda: MyStorage(tmp_path / "decisions"), reopen=lambda: MyStorage(tmp_path / "decisions"))

Each check runs the implementation through the parts of solvi that rely on it and raises ConformanceError (an
AssertionError, so pytest shows it as a failed assertion) naming the guarantee that does not hold; it returns a dict of
what it checked. solvi's own test suite runs every check on every built-in (tests/test_conformance.py).

    check_storage(make, reopen=None)            TraceStorage: chain verifies, records round-trip, replay, query, redact,
                                                tampering is caught, a reopened store is the same store
    check_slow_path(path, states, ...)          SlowPath: Thought of its mode, replay matches, records round-trip through
                                                JSON, cost recomputes, budget respected, decisions replay from a store
    check_strategist(strategist, catalog, ...)  Strategist: a Flow in an executable order, deterministic, hard checks in
                                                the flow, decisions replay, the plan recorded when it says so
    check_head(make, options, rows, answers)    Head: probabilities over its options, deterministic, teach changes the
                                                fingerprint, System.fit(head=) answers from it and replays
    check_decider(decider, inputs, ...)         Decider: Decisions within its options, probabilities, deterministic,
                                                identity recorded, replay
    check_extractor(extractor, system, texts)   Extractor: literal quotes at their offsets, deterministic, read fields
                                                replay
    check_monitor(make, stream, changed=None)   Monitor: reports, flagged, a change flagged, reset forgets it
    check_environment(make, seeds=(0, 1))       Environment: the same seed and actions give the same outcomes
    check_action_model(make, transitions, ...)  ActionModel: Predictions, determinism, the fingerprint moves with what it
                                                learned, never contradicts an outcome it observed; held-out comparison
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path


class ConformanceError(AssertionError):
    """An implementation does not keep a guarantee its protocol promises (the message says which)."""


def _ok(cond, what):
    if not cond:
        raise ConformanceError(what)


def _vh(v):
    from ..core.runtime import vhash
    from ..core.store import plain
    return vhash(plain(v))


def _json(d):
    return json.loads(json.dumps(d, ensure_ascii=False, default=str))


# ------------------------------------------------------------------------------------------------ TraceStorage
def _size_system(storage=None):
    from ..core import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.rule("size")
    def size(amount):
        return "big" if amount > 100 else "small"
    return System(cat, [Question("size", "Is the amount big?", Answer.choice(["small", "big"]))], storage=storage)


AMOUNTS = (5.0, 250.0, 99.5, 1000.0, 42.0)


def check_storage(make, *, reopen=None):
    """make: a function () → a fresh, empty store (a TraceStorage subclass); reopen: () → the same store opened again
    (its records read back from where make() wrote them), when the backend persists. → what was checked."""
    from ..core.store import TraceStorage
    store = make()
    _ok(isinstance(store, TraceStorage), f"{type(store).__name__} is not a TraceStorage subclass")
    _ok(store.head()["count"] == 0, "make() must give an empty store")
    try:
        system = _size_system(store)
        ids = []
        for a in AMOUNTS:
            res = system.ask({"amount": a})
            _ok(isinstance(res.stored_id, str) and res.stored_id, "save() → the record's id (System(storage=) sets "
                                                                  "res.stored_id)")
            ids.append(res.stored_id)
        _ok(len(set(ids)) == len(ids), "every record gets its own id")
        cid = store.save_correction("size", {"amount": 7.0}, "big", by="conformance")
        head = store.head()
        _ok(head["count"] == len(ids) + 1 and isinstance(head["hash"], str), f"head() counts every chained record: {head}")
        v = store.verify()
        _ok(v["ok"] and not v["problems"], f"a fresh chain verifies: {v['problems'][:3]}")
        _ok([s.id for s in store.iter()] == ids, "iter() gives the decisions in stored order")
        for a, i in zip(AMOUNTS, ids):
            back = store.get(i, system)
            _ok(back["size"].answer == ("big" if a > 100 else "small"), f"get({i}) gives the stored answer back")
            _ok(store.record(i)["id"] == i, f"record({i}) is the stored dict")
        _ok(len(store.query(question="size", answer="big")) == sum(a > 100 for a in AMOUNTS), "query(answer=) filters")
        _ok(store.corrections() and store.corrections()[0]["id"] == cid, "corrections() lists the correction")
        _ok(store.replay_all(system) == [], "every stored decision replays")
        anchor = store.head()
        # tampering: an answer edited in place (through the backend's own _rewrite) is caught, and undone verifies again
        rec = store.record(ids[1])
        bad = _json(rec)
        ans = bad.get("answers", {}).get("size")
        _ok(isinstance(ans, list) and ans, "a decision record keeps its answers")
        bad["answers"]["size"][0] = "small" if ans[0] == "big" else "big"
        store._rewrite(bad)
        v = store.verify(anchor=anchor)
        _ok(not v["ok"] and any(p[1] == ids[1] for p in v["problems"]),
            "an answer edited in a stored record breaks verify() at that record")
        store._rewrite(_json(rec))
        _ok(store.verify(anchor=anchor)["ok"], "the record written back verifies again")
        # erasure keeps the chain whole
        rid = store.redact(ids[0], by="conformance", note="erasure check")
        _ok(isinstance(rid, str), "redact() → the redaction record's id")
        v = store.verify(anchor=anchor)
        _ok(v["ok"], f"the chain verifies after an erasure: {v['problems'][:3]}")
        _ok(ids[0] not in [s.id for s in store.iter()], "a redacted record is passed over by iter()")
        _ok(store.replay_all(system) == [], "the decisions left replay after an erasure")
        final = store.head()
    finally:
        store.close()
    out = {"records": final["count"], "verify": True, "tamper": True, "redact": True, "replay": True}
    if reopen is not None:
        again = reopen()
        try:
            _ok(again.head() == final, f"a reopened store has the same head: {again.head()} ≠ {final}")
            _ok(again.verify(anchor=anchor)["ok"], "a reopened store verifies")
            _ok([s.id for s in again.iter()] == ids[1:], "a reopened store holds the same records")
            _ok(again.replay_all(_size_system()) == [], "a reopened store's decisions replay")
        finally:
            again.close()
        out["reopen"] = True
    return out


# ------------------------------------------------------------------------------------------------ SlowPath
def _cost_key(c):
    return (None if c.usd is None else round(c.usd, 9), c.calls, c.input_tokens, c.output_tokens)


def check_slow_path(path, states, *, question=None, price=None, system1=None):
    """path: a SlowPath; states: inputs to think about; question: the question (default: the path's, or its System's
    only one); price: as Dispatcher's; system1: a System 1 for the same question — then every state is also dispatched
    (supervise=1: the path checks every answer) through a store, and the stored decisions must replay. → what was
    checked."""
    from ..core.costs import Budget, Cost, cost_of
    from ..core.dispatch import Dispatcher, SlowPath, Thought
    _ok(isinstance(path, SlowPath), f"{type(path).__name__} is not a SlowPath")
    _ok(isinstance(getattr(type(path), "mode", None), str), "a SlowPath names its mode")
    _ok(issubclass(type(path), SlowPath.path_of(path.mode)), f"mode {path.mode!r} is registered to "
                                                               f"{SlowPath.path_of(path.mode).__qualname__}")
    q = question or path.question
    if q is None:
        qs = list(path.system.questions)
        _ok(len(qs) == 1, "say which question with question=")
        q = qs[0]
    fp = path.fingerprint()
    _ok(isinstance(fp, str) and fp, "fingerprint() is a non-empty str")
    n = 0
    for st in states:
        th = path.run(dict(st), q, price=price)
        _ok(isinstance(th, Thought) and th.mode == path.mode, "run() → a Thought of the path's mode")
        _ok(not th.accepted or th.record is not None, "an accepted answer comes with its record")
        rep = path.replay(th)
        _ok(rep["ok"], f"a fresh Thought replays: {rep['mismatches'][:3]}")
        back = Thought.from_dict(_json(th.to_dict()), path.system)
        _ok(back.mode == th.mode and back.accepted == th.accepted and _vh(back.answer) == _vh(th.answer),
            "a Thought round-trips through JSON (to_dict / from_dict by its mode)")
        rep = path.replay(back)
        _ok(rep["ok"], f"a stored Thought replays: {rep['mismatches'][:3]}")
        c = cost_of(back.responses, price, back.cost.ms, back.generated())
        _ok(_cost_key(c) == _cost_key(back.cost), "the recorded cost is what the record's model outputs give")
        if th.record is not None:
            flipped = Thought(th.mode, th.answer, not th.accepted, th.why, th.record, th.cost, th.stopped)
            _ok(not path.replay(flipped)["ok"], "replay notices a Thought whose acceptance is not its record's")
        if getattr(path, "steps", 1) > 1:
            lean = path.run(dict(st), q, price=price, budget=Budget(calls=0), expected_round=Cost(None, 1))
            _ok(len(lean.responses) <= 1 or lean.stopped, "with no budget for another step the path stops after one")
        n += 1
    _ok(path.fingerprint() == fp, "thinking does not change the path's fingerprint")
    out = {"states": n, "replay": True, "round_trip": True, "cost": True}
    if system1 is not None:
        from ..core.store import JSONLStorage
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONLStorage(Path(tmp) / "dispatch.jsonl")
            d = Dispatcher(system1, path, question=q, price=price, supervise=1.0, storage=store)
            for st in states:
                d.ask(dict(st))
            _ok(d.replay_all() == [], f"stored decisions replay: {d.replay_all()[:2]}")
            _ok(store.verify()["ok"], "the dispatcher's store verifies")
            store.close()
        none = Dispatcher(system1, path, question=q, total=Budget(calls=0))
        _ok(all(none.ask(dict(st)).s2 is None for st in states), "no slow-path run once the total budget is used up")
        out.update(dispatch=True, budget=True)
    return out


# ------------------------------------------------------------------------------------------------ Strategist
def check_strategist(strategist, catalog, questions, states):
    """strategist: the planner; catalog, questions: a System's; states: inputs (dicts of given facts). The System is
    built with System(catalog, questions, strategist=strategist) (which refuses a planner that leaves a hard check's
    `then` out of its question's flow). → what was checked."""
    from ..core.plan.strategist import Strategist
    from ..core.runtime import Flow
    from ..core.system import System
    _ok(isinstance(strategist, Strategist), f"{type(strategist).__name__} has no plan(catalog, questions, init_keys, "
                                            "heads)")
    system = System(catalog, list(questions), strategist=strategist)
    qs = list(system.questions.values())
    hard = [(p.name, qn) for p in catalog.parts.values() if p.kind == "check" and p.hard for qn in (p.then or {})
            if qn in system.questions]
    n = 0
    for st in states:
        keys = list(st)
        flow = strategist.plan(catalog, qs, keys, system.heads)
        _ok(isinstance(flow, Flow), "plan() → a solvi.core.runtime.Flow")
        again = strategist.plan(catalog, qs, keys, system.heads)
        _ok([s.part.name for s in again.steps] == [s.part.name for s in flow.steps], "the same input gives the same plan")
        have = set(keys)
        for s in flow.steps:
            p = s.part
            alts = p.alternatives or [p]
            if p.kind == "rule" and flow.unresolved.get(getattr(p, "question", None)):
                continue                              # an unresolved question's rule: it abstains, it does not run
            _ok(any(all(x in have for x in a.inputs) for a in alts),
                f"step {p.name} runs before its inputs are computed ({p.inputs})")
            have.add(p.name)
        for c, qn in hard:
            _ok(c in flow.per_question.get(qn, ()) or flow.unresolved.get(qn),
                f"hard check {c} (then= for {qn!r}) is in {qn!r}'s flow")
        res = system.ask(dict(st))
        rep = res.trace.replay(system)
        _ok(rep["ok"], f"a decision planned by it replays: {rep['mismatches'][:3]}")
        recorded = any(r.name == "plan:strategy" for r in res.trace.records)
        if getattr(flow, "strategy", None) is not None and getattr(strategist, "record", True):
            _ok(recorded, "a flow with a strategy is recorded in the trace (plan:strategy)")
        n += 1
    return {"states": n, "flow": True, "hard_checks": len(hard), "replay": True}


# ------------------------------------------------------------------------------------------------ Head
def check_head(make, options, rows, answers, features=None):
    """make: options → a fresh, unfitted head; rows: [{fact: value}]; answers: the right option per row; features:
    the facts it may read (default: the first row's keys). → what was checked."""
    from ..core import Answer, Catalog, Question, System
    from ..core.deciders.protocols import Head
    features = list(features if features is not None else rows[0])
    h = make(list(options))
    _ok(isinstance(h, Head), f"{type(h).__name__} lacks a method of the Head protocol")
    h.fit(rows, list(answers), features)
    _ok(set(h.features) <= set(features), "a fitted head reads only the features it was given")
    twin = make(list(options))
    twin.fit(rows, list(answers), features)
    fp = h.fingerprint()
    _ok(isinstance(fp, str) and fp == twin.fingerprint(), "the same fit gives the same fingerprint")
    for r in rows:
        p = h.predict(r)
        _ok(set(p) == set(options), f"predict() gives a probability per option: {sorted(p)}")
        _ok(all(math.isfinite(x) and -1e-9 <= x <= 1 + 1e-9 for x in p.values()) and abs(sum(p.values()) - 1) < 1e-6,
            f"predict() is a distribution: {p}")
        _ok(_vh(p) == _vh(twin.predict(r)), "the same fit gives the same predictions")
        _ok(set(h.contributions(r)) <= set(h.features), "contributions() are per feature")
    i = next((k for k, a in enumerate(answers) if a != max(h.predict(rows[k]), key=h.predict(rows[k]).get)), 0)
    other = [o for o in options if o != answers[i]][0]
    ms = h.teach(rows[i], other)
    _ok(isinstance(ms, (int, float)) and ms >= 0, "teach() → the milliseconds it took")
    _ok(h.fingerprint() != fp, "teach() changes the fingerprint (the head's answers may have changed)")
    # in a System: System.fit(head=) installs it, the System answers from it, the trace replays
    system = System(Catalog(), [Question("answer", "?", Answer.choice(list(options)))])
    system.fit("answer", [(dict(r), a) for r, a in zip(rows, answers)], features, select=False, head=make)
    _ok(isinstance(system.heads["answer"], type(h)), "System.fit(head=) installs the head")
    for r in rows:
        res = system.ask(dict(r))
        p = system.heads["answer"].predict({f: r[f] for f in system.heads["answer"].features})
        if res["answer"].status == "ok":
            _ok(res["answer"].answer == max(p, key=p.get), "the System answers with the head's most probable option")
        rep = res.trace.replay(system)
        _ok(rep["ok"], f"an answer from the head replays: {rep['mismatches'][:3]}")
    return {"rows": len(rows), "distribution": True, "teach": True, "system": True}


# ------------------------------------------------------------------------------------------------ Decider
def check_decider(decider, inputs, *, system=None, states=None):
    """decider: a Decider (called with the facts it reads); inputs: [{fact: value}]; system, states: a System using it
    and inputs to ask it — its decisions must replay and record the decider's fingerprint. → what was checked."""
    from ..core import Decision, Unknown
    from ..core.deciders.protocols import Decider
    _ok(isinstance(decider, Decider), f"{type(decider).__name__} lacks options / fingerprint() / __call__")
    fp = decider.fingerprint()
    _ok(isinstance(fp, str) and fp, "fingerprint() is a non-empty str")
    opts = decider.options
    for x in inputs:
        d = decider(**x)
        _ok(isinstance(d, Decision), f"a call → a solvi.Decision, not {type(d).__name__}")
        vals = d.value if isinstance(d.value, (list, tuple, set)) else [d.value]
        if opts is not None and not d.escalate:
            _ok(all(v in opts or v is Unknown for v in vals), f"{d.value!r} is one of the options {opts}")
        if d.probs:
            ps = list(d.probs.values())
            _ok(all(math.isfinite(p) and -1e-9 <= p <= 1 + 1e-9 for p in ps), f"probabilities in [0, 1]: {d.probs}")
        if getattr(decider, "deterministic", True):
            e = decider(**x)
            _ok(_vh(e.value) == _vh(d.value) and _vh(e.probs) == _vh(d.probs), "a deterministic decider decides the "
                                                                                "same input the same way")
    _ok(decider.fingerprint() == fp, "deciding does not change the fingerprint")
    out = {"inputs": len(inputs), "closed_set": opts is not None}
    if system is not None:
        models = system.fingerprint()["models"]
        _ok(fp in models.values(), "the System records the decider's fingerprint among its models")
        for st in states or []:
            res = system.ask(dict(st))
            rep = res.trace.replay(system)
            _ok(rep["ok"], f"a decision replays: {rep['mismatches'][:3]}")
            _ok(any((r.model or {}).get("fp") == fp for r in res.trace.records),
                "the decider's identity is recorded with the step it produced")
        out["system"] = True
    return out


# ------------------------------------------------------------------------------------------------ Extractor
def check_extractor(extractor, system, texts, *, question=None, **textin):
    """extractor: an Extractor; system: a System whose entry point (question=, or its only one) reads the fields;
    texts: messages to read; textin: TextIn's other options (today=, patterns=, synonyms=, ...). → what was
    checked."""
    from ..core.catalog import Quote
    from ..core.textin import Extractor, TextIn
    _ok(isinstance(extractor, Extractor), f"{type(extractor).__name__} lacks find(text, field) / fingerprint()")
    fp = extractor.fingerprint()
    _ok(isinstance(fp, str) and fp, "fingerprint() is a non-empty str")
    tin = TextIn(system, extractor=extractor, **textin)
    q = question or (next(iter(tin.entry_points)) if len(tin.entry_points) == 1 else None)
    _ok(q is not None, "say which entry point with question=")
    quotes = read = 0
    for text in texts:
        for name in tin.entry_points[q].fields:
            fs = tin.spec(q, name)
            got = extractor.find(text, fs)
            _ok(isinstance(got, list) and all(isinstance(x, Quote) for x in got), "find() → [Quote]")
            for x in got:
                _ok(0 <= x.start <= x.end <= len(text) and x.value == text[x.start:x.end],
                    f"a quote is the text at its offsets: {x.value!r} vs {text[x.start:x.end]!r}")
                _ok(0 <= x.confidence <= 1, "a quote's confidence is in [0, 1]")
                quotes += 1
            _ok([(x.start, x.end, x.value) for x in extractor.find(text, fs)] == [(x.start, x.end, x.value) for x in got],
                "find() is deterministic")
        r = tin.read(text, q)
        for f in r.fields.values():
            if f.status == "read":
                _ok(f.quote.value == text[f.quote.start:f.quote.end], "a field read is quoted literally")
                read += 1
        if not r.missing:
            res = system.ask_text(r)
            rep = res.trace.replay(system)
            _ok(rep["ok"], f"a decision on a read text replays: {rep['mismatches'][:3]}")
    _ok(extractor.fingerprint() == fp, "reading does not change the fingerprint")
    return {"texts": len(texts), "quotes": quotes, "fields_read": read}


# ------------------------------------------------------------------------------------------------ Monitor
def check_monitor(make, stream, *, changed=None):
    """make: () → a fresh monitor; stream: decisions of an unchanged stream (Responses, Results, Decisions or dicts the
    monitor reads); changed: decisions after a change — when given, the monitor must flag it. → what was checked."""
    from ..core.guarantees.monitor import Monitor
    m = make()
    _ok(isinstance(m, Monitor), f"{type(m).__name__} lacks observe / flagged / reset")

    def run(mon, xs):
        out = []
        for x in xs:
            rep = mon.observe(x)
            _ok(isinstance(rep, dict) and isinstance(rep.get("drift"), bool) and isinstance(rep.get("why"), list),
                "observe() → a report with drift (bool) and why (list)")
            _ok(mon.flagged == rep["drift"], "flagged is the last report's drift")
            out.append(rep["drift"])
        return out
    first = run(m, stream)
    flagged_on_change = None
    if changed is not None:
        after = run(m, changed)
        flagged_on_change = any(after)
        _ok(flagged_on_change, "the monitor flags the changed stream")
    m.reset()
    _ok(not m.flagged, "reset() clears the flag")
    again = run(m, stream)
    if not any(first):
        _ok(not any(again), "after reset() the unchanged stream is not flagged (the change was forgotten)")
    _ok(run(make(), stream) == first, "a fresh monitor reports the same stream the same way")
    return {"stream": len(first), "changed": flagged_on_change, "reset": True}


# ------------------------------------------------------------------------------------------------ Environment
def check_environment(make, *, seeds=(0, 1), steps=30, policy=None):
    """make: () → a fresh environment; policy: (state, actions) → the action to take (default: the first one).
    → what was checked."""
    from ..core.environment import Environment, Outcome
    pick = policy or (lambda state, acts: acts[0])
    _ok(isinstance(make(), Environment), "the environment lacks reset / actions / step")

    def episode(seed):
        env = make()
        state = env.reset(seed)
        trail = [_vh(state)]
        for _ in range(steps):
            acts = list(env.actions(state))
            if not acts:
                break
            o = env.step(pick(state, acts))
            _ok(isinstance(o, Outcome), f"step() → an Outcome, not {type(o).__name__}")
            if not o.accepted:
                _ok(_vh(o.state) == _vh(state), "a refused action leaves the state as it was")
            trail.append((_vh(o.state), o.accepted, _vh(o.effect), o.done))
            state = o.state
            if o.done:
                break
        return trail
    n = 0
    for seed in seeds:
        a, b = episode(seed), episode(seed)
        _ok(a == b, f"seed {seed}: the same seed and actions give the same outcomes")
        n += len(a) - 1
    return {"seeds": len(seeds), "steps": n, "deterministic": True}


# ------------------------------------------------------------------------------------------------ ActionModel
def check_action_model(make, transitions, *, held_out=None):
    """make: () → a fresh action model (solvi.core.knowledge.ActionModel); transitions: what an environment did —
    [(state, action, args, accepted, effect)], observed in order; held_out: more of them, only predicted (the result
    reports how the predictions compare). Checks: the protocol's methods; every prediction is a Prediction with a
    verdict in accept / refuse / unknown, a risk ≥ 0 and an integer support, plain JSON in to_dict(); learning is
    deterministic (two fresh models fed the same transitions give the same predictions and the same fingerprint);
    what it learned changes its fingerprint; and it never contradicts an outcome it observed (on a (state, action, args)
    the environment answered one way every time, it predicts that answer or "unknown" — never the other). → what was
    checked, with the held-out comparison (precision and recall of refusals over answered ones, abstention)."""
    from ..core.knowledge.actions import VERDICTS, ActionModel, Prediction
    transitions = list(transitions)
    m = make()
    _ok(isinstance(m, ActionModel), f"{type(m).__name__} lacks observe / predict / fingerprint")
    fresh_fp = m.fingerprint()
    _ok(isinstance(fresh_fp, str) and fresh_fp, "fingerprint() → a non-empty string")

    def valid(p):
        _ok(isinstance(p, Prediction), f"predict() → a Prediction, not {type(p).__name__}")
        _ok(p.verdict in VERDICTS, f"a verdict in {VERDICTS}, not {p.verdict!r}")
        _ok(isinstance(p.risk, (int, float)) and math.isfinite(p.risk) and p.risk >= 0, "risk is a finite number ≥ 0")
        _ok(isinstance(p.support, int) and not isinstance(p.support, bool), "support is an integer (-1: a spec's rate)")
        _ok(isinstance(p.hard, bool), "hard is True or False")
        _json(p.to_dict())
        return p.verdict

    for state, action, args, accepted, effect in transitions:
        valid(m.predict(state, action, args))
        m.observe(state, action, args, accepted, effect)
    again = make()
    for state, action, args, accepted, effect in transitions:
        again.observe(state, action, args, accepted, effect)
    _ok(again.fingerprint() == m.fingerprint(), "the same transitions give the same fingerprint")
    if transitions:
        _ok(m.fingerprint() != fresh_fp, "what the model learned changes its fingerprint")
    seen = {}
    for state, action, args, accepted, _ in transitions:
        seen.setdefault(_vh([state, action, args]), (state, action, args, set()))[3].add(bool(accepted))
    wrong = abstained = 0
    for state, action, args, outcomes in seen.values():
        v = valid(m.predict(state, action, args))
        _ok(v == valid(again.predict(state, action, args)), "the same transitions give the same predictions")
        if len(outcomes) != 1:
            continue
        ok = outcomes.pop()
        abstained += v == "unknown"
        if (v == "accept" and not ok) or (v == "refuse" and ok):
            wrong += 1
    _ok(wrong == 0, f"the model contradicts {wrong} outcome(s) it observed")
    out = {"transitions": len(transitions), "distinct": len(seen), "abstained_on_own_data": abstained,
           "deterministic": True}
    if held_out is not None:
        rows = [(valid(m.predict(s, a, x)), bool(acc)) for s, a, x, acc, _ in held_out]
        ans = [(v, ok) for v, ok in rows if v != "unknown"]
        tp = sum(1 for v, ok in ans if v == "refuse" and not ok)
        fp = sum(1 for v, ok in ans if v == "refuse" and ok)
        fn = sum(1 for v, ok in ans if v == "accept" and not ok)
        out["held_out"] = {"n": len(rows), "abstained": len(rows) - len(ans),
                           "refusal_precision": tp / (tp + fp) if tp + fp else None,
                           "refusal_recall": tp / (tp + fn) if tp + fn else None}
    return out


__all__ = ["check_action_model", "check_decider", "check_environment", "check_extractor", "check_head",
           "check_monitor", "check_slow_path", "check_storage", "check_strategist", "ConformanceError"]
