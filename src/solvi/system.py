"""System: catalog + questions → ask(init_state) → answers with confidence, flow, computed_state, trace; fit / teach / journal."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .core import Catalog
from .heads import Head
from .runtime import MISSING, Result, execute, now_ms, path_confidence
from .strategist import computable, plan


@dataclass
class Response:
    results: dict
    flow: object
    trace: object
    values: dict
    ms: float

    def __getitem__(self, q):
        return self.results[q]

    @property
    def computed_state(self):
        lines = []
        for r in self.trace.records:
            if r.kind == "rule":
                continue
            v = "—" if r.value is MISSING else repr(r.value)
            extra = f"   quote [{r.quote[0]}:{r.quote[1]}]" if r.quote else ""
            if r.kind == "extract" and r.confidence < 1:
                extra += f" confidence {r.confidence:.2f}"
            if r.error:
                extra += f"   ERROR: {r.error}"
            lines.append(f"{r.name:24s} = {v}{extra}")
        return "\n".join(lines)


class System:
    def __init__(self, catalog: Catalog, questions, journal: str | None = None, workers: int = 1):
        self.catalog = catalog
        self.questions = {q.name: q for q in questions}
        self.heads: dict[str, Head] = {}
        self.journal = Path(journal) if journal else None
        self.workers = workers                    # >1: independent steps run in parallel threads
        self.calib: dict[str, tuple] = {}         # question → (a, b): confidence' = σ(a·logit(confidence) + b)
        self.learned_rules = {}                   # question → RuleList (readable rules learned from examples)

    # --- answers
    def ask(self, init_state, names=None, workers=None):
        t0 = now_ms()
        qs = [self.questions[n] for n in (names or self.questions)]
        flow = plan(self.catalog, qs, init_state.keys(), self.heads)
        trace, vals = execute(self.catalog, flow, init_state, workers=workers or self.workers)
        by = {r.name: r for r in trace.records}
        results = {}
        for q in qs:
            r = self._answer(q, flow, trace, vals, by)
            if q.name in self.calib and r.status == "ok":
                r.confidence = _platt(r.confidence, *self.calib[q.name])
            results[q.name] = r
        resp = Response(results, flow, trace, vals, now_ms() - t0)
        if self.journal:
            self._log(init_state, resp)
        return resp

    def _answer(self, q, flow, trace, vals, by):
        facts = flow.per_question.get(q.name, [])
        # hard checks: a false hard check in the question's flow decides the answer (the model cannot override it)
        for f in facts:
            part = self.catalog.parts.get(f)
            r = by.get(f)
            if part is not None and part.kind == "check" and part.hard and r is not None and r.value is False:
                if q.name in part.then:
                    return Result(q.answer.normalize(part.then[q.name]), 1.0, f"hard check {f} is false", "forced")
                return Result(None, 0.0, f"hard check {f} is false and no answer is set for it", "abstain")
        missing = [f for f in facts if f in by and by[f].value is MISSING]
        if flow.unresolved.get(q.name):
            return Result(None, 0.0, "cannot compute: " + ", ".join(flow.unresolved[q.name]), "abstain")
        rule = self.catalog.rules.get(q.name)
        soft_failed = [f for f in facts if self.catalog.parts.get(f) is not None and self.catalog.parts[f].kind == "check"
                       and not self.catalog.parts[f].hard and by.get(f) is not None and by[f].value is False]
        if rule is not None:
            r = by.get(rule.name)
            if r is None or r.value is MISSING:
                return Result(None, 0.0, "rule not computed: " + (r.error if r else "no step") +
                              (f"; missing {', '.join(missing)}" if missing else ""), "abstain")
            conf = path_confidence(self.catalog, trace, rule.inputs)
            why = "; ".join(f"{x} = {vals.get(x)!r}" for x in rule.inputs)
            try:
                return Result(q.answer.normalize(r.value), conf, why)
            except ValueError:
                return Result(None, 0.0, f"rule returned {r.value!r}, not one of the answer options; {why}", "abstain")
        head = self.heads.get(q.name)
        if head is None:
            return Result(None, 0.0, "no rule and the answer head is not fitted (fit)", "abstain")
        lost = [f for f in head.features if f not in vals]
        if lost:                                      # a head never guesses from facts that could not be computed
            return Result(None, 0.0, "head features not computed: " + ", ".join(lost), "abstain")
        p = head.predict(vals)
        a = max(p, key=p.get)
        contrib = head.contributions(vals)
        why = ", ".join(f"{f} = {vals.get(f)!r} ({c:+.2f})" for f, c in sorted(contrib.items(), key=lambda t: -abs(t[1]))[:4])
        if soft_failed:
            why += "; failed checks: " + ", ".join(soft_failed)
        conf = p[a] * path_confidence(self.catalog, trace, head.features)
        return Result(a, conf, why, probs=p)

    # --- task-specific training
    def facts_for(self, init_state):
        """All computable facts (for head training): a "compute everything" flow without rules."""
        from .core import Question
        from .strategist import plan as _plan
        q = Question("__all__", "", None)
        flow = _plan(self.catalog, [q], init_state.keys())
        flow.steps = [s for s in flow.steps if s.part.kind != "rule"]
        _, vals = execute(self.catalog, flow, init_state, early_exit=False)
        return vals

    def fit(self, question, examples):
        """examples: [(init_state, answer)] → the answer head and its features (and hence the question's flow)."""
        q = self.questions[question]
        rows = [self.facts_for(s) for s, _ in examples]
        ans = [q.answer.normalize(a) for _, a in examples]
        keys = set(examples[0][0].keys())
        cands = sorted(f for f in computable(self.catalog, keys) - keys)
        self.heads[question] = Head(q.answer.options).fit(rows, ans, cands)
        return self.heads[question]

    def fit_fast(self, question, examples, features=None, lam=None):
        """Fast answer head (closed-form ridge, milliseconds): examples — [(init_state, answer)]. Features: the given facts
        (numbers, booleans, categories or vectors such as a document embedding), by default every computed fact. Unlike fit it
        keeps all features and learns online: every `teach` for this question updates it instantly."""
        from .fast import FastHead
        import time
        q = self.questions[question]
        t0 = time.perf_counter()
        rows = [self.facts_for(s) for s, _ in examples]
        if features is None:
            keys = set(examples[0][0].keys())
            features = sorted(f for f in computable(self.catalog, keys) - keys)
        head = FastHead(q.answer.options, lam=lam).fit(rows, [q.answer.normalize(a) for _, a in examples], list(features))
        head.fit_ms = (time.perf_counter() - t0) * 1000
        self.heads[question] = head
        return head

    def learn_rule(self, question, examples, facts, **kw):
        """An answer rule learned from examples (solvi.rules.RuleList): a readable "if feature then answer" list, installed in the
        catalog as a regular rule (deterministic, replayable)."""
        from .core import Part
        from .rules import RuleList
        q = self.questions[question]
        rows = [self.facts_for(s) for s, _ in examples]
        rl = RuleList(facts, **kw).fit(rows, [q.answer.normalize(a) for _, a in examples])

        def learned(**args):
            return rl.predict(args)[0]
        learned.__name__ = f"learned_{question}"
        self.catalog.rules[question] = Part(kind="rule", name="answer:" + question, inputs=list(facts), func=learned,
                                            doc=str(rl), question=question)
        self.learned_rules[question] = rl
        return rl

    def calibrate(self, question, examples, truth):
        """Calibrate answer confidence (Platt scaling) on held-out examples: examples is [init_state], truth is [correct answer]."""
        import numpy as np
        xs, ys = [], []
        for st, y in zip(examples, truth):
            r = self.ask(st, [question])[question]
            if r.status != "ok":
                continue
            c = min(max(r.confidence, 1e-4), 1 - 1e-4)
            xs.append(np.log(c / (1 - c)))
            ys.append(float(r.answer == y))
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 10 or ys.min() == ys.max():
            self.calib[question] = (1.0, float(np.log((ys.mean() + 1e-3) / (1 - ys.mean() + 1e-3))) if len(xs) else 0.0)
            return self.calib[question]
        a, b = 1.0, 0.0
        for _ in range(500):
            p = 1 / (1 + np.exp(-(a * xs + b)))
            ga, gb = ((p - ys) * xs).mean(), (p - ys).mean()
            a, b = a - 0.5 * ga, b - 0.5 * gb
        self.calib[question] = (float(a), float(b))
        return self.calib[question]

    def teach(self, question, init_state, correct):
        """Human correction. A fast head (fit_fast) absorbs it at once; any head gets it in the journal for the next fit.
        Returns the update time in ms for a fast head, else None."""
        from .fast import FastHead
        ms = None
        head = self.heads.get(question)
        if isinstance(head, FastHead):
            ms = head.update(self.facts_for(init_state), self.questions[question].answer.normalize(correct))
        if self.journal:
            with open(self.journal, "a") as fh:
                fh.write(json.dumps({"teach": question, "init": _jsonable(init_state), "answer": correct}, ensure_ascii=False) + "\n")
        return ms

    def _log(self, init_state, resp):
        with open(self.journal, "a") as fh:
            fh.write(json.dumps({"init_hash": resp.trace.init_hash,
                                 "answers": {q: [r.answer, round(r.confidence, 4), r.status] for q, r in resp.results.items()},
                                 "flow": [s.part.name for s in resp.flow.steps],
                                 "records": [[r.step, r.name, r.hash] for r in resp.trace.records]}, ensure_ascii=False) + "\n")


def _platt(c, a, b):
    import math
    c = min(max(c, 1e-4), 1 - 1e-4)
    return 1 / (1 + math.exp(-(a * math.log(c / (1 - c)) + b)))


def _jsonable(d):
    return json.loads(json.dumps(d, default=repr))
