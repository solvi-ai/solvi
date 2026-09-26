"""Flow execution: computed_state with provenance and a hash chain, answers, hard checks, independent replay."""
from __future__ import annotations

import hashlib
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .core import Quote


def _canon(v):
    if isinstance(v, Quote):
        return {"quote": [_canon(v.value), v.start, v.end, v.source]}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, float):
        return round(v, 9)
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)


def vhash(v) -> str:
    return hashlib.sha256(json.dumps(_canon(v), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


MISSING = object()


@dataclass
class Record:
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

    def body(self):
        return {"step": self.step, "kind": self.kind, "name": self.name, "inputs": self.inputs, "value": vhash(self.value),
                "quote": self.quote, "error": self.error, "prev": self.prev}


@dataclass
class Result:
    answer: Any
    confidence: float
    why: str
    status: str = "ok"             # ok | forced | abstain
    probs: dict = field(default_factory=dict)


@dataclass
class Trace:
    init_hash: str
    records: list
    init: dict
    skipped: list = field(default_factory=list)     # [(part, why)] steps of the flow that did not need to run

    def value(self, name):
        for r in self.records:
            if r.name == name:
                return r.value
        return self.init.get(name, MISSING)

    def replay(self, catalog, flow=None):
        """Independent replay: recompute every step from its recorded inputs, verify the value, the quote, the error and the
        hash chain, and that the chain starts from the hash of the recorded input. With `flow` (res.flow) it also checks the
        trace is complete: every planned step is either recorded or listed as skipped at run time.
        Limits: a trace rebuilt honestly from a *different* input is internally consistent — compare `init_hash` with a
        receipt you published elsewhere to catch that."""
        vals = dict(self.init)
        prev = self.init_hash
        bad = []
        if vhash(self.init) != self.init_hash:
            bad.append((0, "init", "init_hash does not match the recorded input"))
        for r in self.records:
            if r.prev != prev:
                bad.append((r.step, r.name, "hash chain broken"))
            if vhash(r.body()) != r.hash:
                bad.append((r.step, r.name, "record modified after execution"))
            prev = r.hash
            part = catalog.rules[r.name[7:]] if r.kind == "rule" else catalog.parts[r.name]
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
            try:
                v = part.func(**{x: (a.value if isinstance(a, Quote) else a) for x, a in args.items()})
            except Exception as e:  # noqa: BLE001
                if r.error is None:
                    bad.append((r.step, r.name, f"recompute failed: {type(e).__name__}"))
                vals[r.name] = r.value
                continue
            if r.error is not None and not r.error.startswith("quote outside"):
                bad.append((r.step, r.name, f"recorded error {r.error!r}, but the step recomputes fine"))
                vals[r.name] = r.value
                continue
            if isinstance(v, Quote):
                src = self.init.get(v.source, "")
                if not (isinstance(src, str) and 0 <= v.start <= v.end <= len(src)):
                    bad.append((r.step, r.name, "quote outside the text"))
                v = v.value
            if r.error is None and vhash(v) != vhash(r.value):
                bad.append((r.step, r.name, f"value {r.value!r} ≠ recomputed {v!r}"))
            vals[r.name] = r.value
        if flow is not None:
            seen = {r.name for r in self.records} | {n for n, _ in self.skipped}
            for st in flow.steps:
                if st.part.name not in seen:
                    bad.append((0, st.part.name, "planned step missing from the trace"))
        return {"ok": not bad, "steps": len(self.records), "mismatches": bad}


_THREADS = sys.platform != "emscripten"         # no threads in the browser (Pyodide): steps then run one by one


def _run_step(p, vals, init_state):
    """Evaluate one part on the current facts → (value, quote, confidence, error, input hashes)."""
    args = {x: vals.get(x, MISSING) for x in p.inputs}
    hashes = {x: vhash(v) for x, v in args.items() if v is not MISSING}
    if any(v is MISSING for v in args.values()):
        return MISSING, None, 1.0, "missing inputs: " + ", ".join(x for x, v in args.items() if v is MISSING), hashes
    try:
        v = p.func(**args)
    except Exception as e:  # noqa: BLE001
        return MISSING, None, 1.0, f"{type(e).__name__}: {str(e)[:120]}", hashes
    if isinstance(v, Quote):
        src = init_state.get(v.source, "")
        err = None if isinstance(src, str) and 0 <= v.start <= v.end <= len(src) else "quote outside the text"
        return v.value, (v.start, v.end, v.source), v.confidence, err, hashes
    return v, None, 1.0, None, hashes


def execute(catalog, flow, init_state, workers=1, early_exit=True):
    """Run the flow. Hard checks and what they depend on run first; a failed hard check settles the questions whose flow contains
    it, and the steps only those questions needed are skipped (early exit). Steps that do not depend on each other run in
    parallel when workers > 1 (threads: suits I/O-bound parts such as API calls and model inference). Records are always written
    in flow order, so the hash chain and replay do not depend on scheduling."""
    vals = dict(init_state)
    steps = flow.steps
    names = [st.part.name for st in steps]
    index = {n: i for i, n in enumerate(names)}
    done = {}
    live = set(flow.per_question)
    settled_by = {}

    def run(idxs):
        if workers <= 1 or len(idxs) <= 1 or not _THREADS:
            for i in idxs:                                  # idxs are in topological (flow) order
                done[i] = _run_step(steps[i].part, vals, init_state)
                if done[i][0] is not MISSING:
                    vals[names[i]] = done[i][0]
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
                    running[ex.submit(_run_step, steps[i].part, dict(vals), init_state)] = i
                finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in finished:
                    i = running.pop(fut)
                    done[i] = fut.result()
                    if done[i][0] is not MISSING:
                        vals[names[i]] = done[i][0]

    hard = [i for i, st in enumerate(steps) if st.part.kind == "check" and st.part.hard]
    if early_exit and hard:
        first = set()

        def up(i):
            if i not in first:
                first.add(i)
                for x in steps[i].part.inputs:
                    if x in index:
                        up(index[x])
        for i in hard:
            up(i)
        run(sorted(first))
        for i in hard:
            if done[i][0] is False:
                part = steps[i].part
                checkpoint_of = {r.split(" ", 1)[1] for r in steps[i].reasons if r.startswith("checkpoint ")}
                for q in [q for q in live if names[i] in flow.per_question.get(q, ())
                          and (not part.then or q in part.then or q in checkpoint_of)]:
                    live.discard(q)
                    settled_by[q] = names[i]

    def needed(i):
        p = steps[i].part
        if not early_exit:
            return True
        if p.kind == "rule":
            return p.question in live
        return any(p.name in flow.per_question.get(q, ()) for q in live)
    run([i for i in range(len(steps)) if i not in done and needed(i)])

    init_hash = vhash(init_state)
    prev, recs, skipped = init_hash, [], []
    for i, st in enumerate(steps, 1):
        if i - 1 not in done:
            by = sorted({settled_by[q] for q in settled_by if st.part.name in flow.per_question.get(q, ()) or
                         (st.part.kind == "rule" and st.part.question == q)})
            skipped.append((st.part.name, "not needed: hard check " + ", ".join(by) + " failed" if by else "not needed"))
            continue
        value, quote, conf, err, hashes = done[i - 1]
        rec = Record(step=i, kind=st.part.kind, name=st.part.name, inputs=hashes, value=value, quote=quote, confidence=conf,
                     error=err, prev=prev)
        rec.hash = vhash(rec.body())
        prev = rec.hash
        recs.append(rec)
    final = {k: v for k, v in vals.items()}
    return Trace(init_hash, recs, dict(init_state), skipped), final


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
        c = r.confidence if r.kind == "extract" else 1.0
        if part is not None:
            for x in part.inputs:
                c = min(c, conf(x))
        memo[f] = c
        return c
    return min([conf(f) for f in facts] or [1.0])


def now_ms():
    return time.perf_counter() * 1000
