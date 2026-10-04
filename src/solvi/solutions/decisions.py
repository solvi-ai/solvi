"""One entry point for System 1 and System 2: a question, labelled examples, a promise and (optionally) a slow path in;
a fitted, calibrated, stored System and Dispatcher out — with a plain account of who answers what and what is promised.

    import solvi

    s = solvi.build(Question("team", "Which team?", Answer.choice(TEAMS)), examples, catalog=cat, max_risk=0.02,
              slow=model, price=(0.04, 0.17), storage="decisions.jsonl")
    res = s.ask({"email": "my parcel never came"})       # a solvi.core.dispatch.Dispatched: answer, by, reasons, cost
    print(s.explain())                                   # who answers which slice, what is promised, on what data
    print(s.report())                                    # what it did, read from the store

Nothing here learns anything new: it composes what the library has, with defaults, and records every choice.

System 1, from what is given:
  - the catalog has a rule (or a decision part) answering the question → it answers as it is;
  - learner= (a function [(state, answer)] → a decision part, e.g. your classifier wrapped as a solvi part) → fitted on
    the fit share of the examples;
  - otherwise → `System.fit`: a ridge head over every fact the catalog computes from the examples' inputs (the given
    keys included), with its feature selection.

The examples are shuffled (seed=) and split: a fit share when something is fitted or a signal is chosen (3/4 to fit a
head or a learner; 1/3 to choose a signal for a rule), the rest for calibration — all of it for System 1's guarantee,
or half for it and half for the dispatcher when there is a slow path, since the dispatcher must be calibrated on
examples System 1's guarantee did not see. `split=(fit, guarantee, dispatch)` sets the shares.

The signal System 1's guarantee reads: signal= when given; else the act probability of a decision part that has one;
else the answer's confidence; and when that does not vary (a rule's answer is always sure), the number the catalog
computes that best separates right from wrong answers on the fit share (AUROC), recorded with its AUROC.

New kinds of input (novel="auto"): when the answer is a choice among more than two options and System 1 is fitted here
(a head or learner=), answers no example shows can come — an intent nobody labelled. The leave-options-out simulation
(`solvi.core.guarantees.openset.leave_out`: System 1 fitted three times without a third of the options) sizes an `OpenSetGate` that
keeps max_error (or max_risk, as the stricter max_error at the same level) on a stream with a share of such inputs and
flags the change. novel=False: a plain guarantee; novel=True: the gate or an error.

The promise: max_risk= (P(answered alone and wrong) ≤ it, conformal risk control) or max_error= (the error among the
answers given alone ≤ it with probability ≥ 1 − delta, learn-then-test) — on System 1 by `System.guarantee`, and with a
slow path on all the answers given alone together by `Dispatcher.calibrate`, which picks per slice System 1 hands over
its own would-be answer, the slow path's, the slow path's when it agrees, or a person.

The slow path (slow=): a SlowPath or a System as they are; a decision part (`model.decision(...)`); a model from
`solvi.core.deciders.llm` (a decision part is made from the question's text and options, reading `reads` — default: every given key
of the examples; with the open-set gate on, "not stated" is an option and goes to a person); a function
state → answer; a compiled specification (a Compiled from `solvi.experimental.compile.compile_spec`, compiled from the
written text, never from the examples; a Spec is compiled first — build does not import the experimental compiler). Inputs the open-set gate holds back go to a person, not to
the slow path: a slice calibrated on known kinds of input says nothing about a new kind (an LLM given unseen intents
put most of them on a known one). After the gate flags a change of the stream, System 1's answers are checked by the
slow path and a disagreement goes to a person.

What it does not do: it does not make either path more accurate, it does not teach System 1 from the slow path, and
its promises hold for inputs like the examples (the gate stretches that to new options like the left-out ones, not
further). One question per build."""
from __future__ import annotations

import copy
import math
import random

from ..core.catalog import Answer, Catalog, Question

SPLIT_FIT = 0.75            # the share fitted on when System 1 is fitted here
SPLIT_SELECT = 1 / 3        # the share a signal is chosen on when a rule answers and its confidence does not vary


class DecisionSystem:
    """What build() returns. system: System 1 (a solvi.System); dispatcher: the solvi.core.dispatch.Dispatcher that asks it;
    gate: the OpenSetGate, or None; slow: the SlowPath, or None; choices: every choice made, with its numbers (what
    explain() prints); calibration: the reports of System.guarantee / OpenSetGate / Dispatcher.calibrate."""

    def __init__(self, question, system, dispatcher, gate, slow, choices, calibration):
        self.question, self.system, self.dispatcher, self.gate, self.slow = question, system, dispatcher, gate, slow
        self.choices, self.calibration = choices, calibration

    def ask(self, state):
        """One input → a Dispatched (answer, by "s1" / "s2" / "human", action, reasons, cost); stored when the build
        was given storage=."""
        return self.dispatcher.ask(state)

    def replay(self, res, trust_models=False):
        """Re-check a Dispatched without calling a model → {"ok", "mismatches"} (Dispatcher.replay)."""
        return self.dispatcher.replay(res, trust_models=trust_models)

    def replay_all(self, trust_models=False):
        """Replay every stored decision → the ones that failed (Dispatcher.replay_all; needs storage=)."""
        return self.dispatcher.replay_all(trust_models=trust_models)

    @property
    def storage(self):
        return self.dispatcher.storage

    def report(self, since=None, until=None, **options):
        """What the system did over a stored period (solvi.core.store.sysreport.SystemReport: print it) — who answered, the cost,
        the promise in force against the stored labels, drift — read from the store alone."""
        from ..core.store.sysreport import system_report
        if self.dispatcher.storage is None:
            raise ValueError("a report reads the stored decisions: build(..., storage=path)")
        options.setdefault("price", self.dispatcher.price)
        return system_report(self.dispatcher.storage, since, until, **options)

    def explain(self):
        """The chosen setup in plain text: what System 1 is, what it reads, how the examples were split, the
        promise and its threshold, the open-set gate, who answers each slice System 1 hands over, the budget, the
        store, and what is not covered."""
        c = self.choices
        lines = [f"Question {self.question!r}: {c['examples']} labelled examples, shuffled with seed {c['seed']}: "
                 + ", ".join(f"{v} {k}" for k, v in c["split"].items() if v) + "."]
        lines.append(f"System 1: {c['system1']}.")
        lines.append(f"Its signal: {c['signal']}.")
        g = c["guarantee"]
        lines.append(f"Its promise: {g['promise']}. {g['how']}")
        if c.get("novel"):
            lines.append(f"New kinds of input: {c['novel']}.")
        if self.slow is None:
            lines.append("No slow path: every input System 1 does not answer alone goes to a person, with System 1's "
                         "would-be answer and the reason.")
        else:
            lines.append(f"Slow path: {c['slow']}.")
            if c.get("probe"):
                lines.append(f"Is a slice open to it? {c['probe'][0].upper() + c['probe'][1:]}.")
            for k, v in c.get("slices", {}).items():
                lines.append(f"  slice {k!r} ({v['n']} calibration examples): {v['text']}")
            if c.get("dispatch_promise"):
                lines.append(f"All answers given alone together: {c['dispatch_promise']}.")
                lines.append(f"Woken by: {', '.join(c['wake'])}; held back by the open-set gate → a person; after a drift "
                             "flag System 1's answers are checked by the slow path and a disagreement goes to a person.")
            else:
                lines.append("Every input System 1 does not answer alone goes to a person; after a drift flag System 1's "
                             "answers are checked by the slow path and a disagreement goes to a person.")
        lines.append(f"Budget: {c['budget']}.")
        lines.append(f"Record: {c['storage']}.")
        lines.append("Not covered: inputs unlike the examples" + (" beyond new options like the left-out ones"
                                                                if self.gate is not None else "")
                     + "; the promise holds for inputs like the examples, so calibrate again when the inputs change.")
        return "\n".join(lines)

    def __str__(self):
        return self.explain()

    def __repr__(self):
        return f"DecisionSystem({self.question!r}, system1={self.choices['mode']}, slow={self.slow!r})"


# ------------------------------------------------------------------------------------------------ helpers
def _judge(q, correct):
    """correct(answer, label) → bool, or equality of the normalized answers."""
    from ..core.runtime import vhash

    def right(ans, label):
        if correct is not None:
            return bool(correct(ans, label))
        try:
            return vhash(ans) == vhash(q.answer.normalize(label))
        except ValueError:
            return vhash(ans) == vhash(label)
    return right


def _rows(system, qname, examples, signal, correct):
    """System 1 (no guarantee) on labelled examples → (signals, right, lost) as System.guarantee reads them."""
    from ..core.guarantees.guarantee import QuestionGuard, _add_checkpoints, _calibration_rows, _drop_checkpoints
    guard = QuestionGuard(qname, None, signal)
    _add_checkpoints(system, guard)
    try:
        corr = None if correct is None else (lambda result, label: correct(result.answer, label))
        sig, ok, _, lost = _calibration_rows(system, qname, examples, guard, corr)
    finally:
        _drop_checkpoints(system, guard)
    return sig, ok, lost


def _choose_signal(system, qname, examples, correct, q):
    """The confidence does not vary: the numeric fact that best separates right from wrong answers → (name, auroc,
    {candidate: auroc})."""
    from ..core.guarantees.guarantee import separation
    right = _judge(q, correct)
    facts, ok = [], []
    for st, y in examples:
        res = system.ask(st, [qname], store=False)
        r = res[qname]
        if r.status == "abstain":
            continue
        facts.append(system.facts_for(st))
        ok.append(right(r.answer, y))
    names = sorted({k for f in facts for k, v in f.items()
                    if isinstance(getattr(v, "value", v), (int, float)) and not isinstance(getattr(v, "value", v), bool)})
    scores = {}
    for n in names:
        s = [getattr(f.get(n), "value", f.get(n)) for f in facts]
        s = [float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else -math.inf for v in s]
        if len(set(s)) < 2:
            continue
        a = separation(s, ok)["auroc"]
        if a is not None:
            scores[n] = round(a, 4)
    if not scores:
        return None, None, scores
    best = max(sorted(scores), key=lambda n: scores[n])
    return best, scores[best], scores


def _constant(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return len(xs) < 2 or max(xs) - min(xs) < 1e-12


def _slow_path(slow, q, reads, novel_on, writer, inputs):
    """slow= → (SlowPath, description)."""
    from ..core.dispatch import AskPath, SlowPath
    from ..core.system import System
    if isinstance(slow, SlowPath):
        return slow, f"the given SlowPath ({slow.mode})"
    if isinstance(slow, System):
        return AskPath(slow, question=q.name if q.name in slow.questions else None), "the given System"
    kind = type(slow).__name__
    if kind == "Spec" or writer is not None or inputs is not None:
        raise TypeError("a Spec as the slow path is compiled first (since 1.0 build does not compile it): c = "
                        "solvi.experimental.compile.compile_spec(spec, [question], inputs, writer); build(..., slow=c)")
    if hasattr(slow, "system") and hasattr(slow, "accepted"):
        c = slow
        if not c.accepted:
            raise ValueError(f"the compiled specification was not accepted: {c.reason}")
        return AskPath(c.system()), "the rules compiled from the written specification (solvi.experimental.compile)"
    cat = Catalog()
    if hasattr(slow, "question") and hasattr(slow, "decide"):          # a decision part
        sq = slow.question(cat, name=q.name, text=q.text)
        return AskPath(System(cat, [sq])), f"the decision part {getattr(slow, '__name__', q.name)!r}"
    if hasattr(slow, "decision") and hasattr(slow, "scorer"):         # a model (solvi.core.deciders.llm, a DecideModel)
        opts = list(q.answer.options or [])
        boolean = sorted(map(str, opts)) == ["no", "yes"]
        part = slow.decision(q.name, q.text, reads if len(reads) > 1 else reads[0], () if boolean else opts,
                             type=bool if boolean else None, not_stated=bool(novel_on and not boolean))
        sq = part.question(cat, name=q.name, text=q.text)
        what = getattr(slow, "model_id", None) or type(slow).__name__
        return AskPath(System(cat, [sq])), (f"the model {what} asked the question, reading {', '.join(reads)}"
                                             + ("; it may say \"not stated\" (→ a person)" if novel_on and not boolean else ""))
    if callable(slow):
        from ..core.catalog import Part
        fn = slow

        def answer(**given):
            return fn(given)
        answer.__name__ = f"slow_{q.name}"
        cat.replace_rule(Part(kind="rule", name="answer:" + q.name, inputs=list(reads), func=answer, question=q.name,
                              doc=(getattr(fn, "__doc__", "") or "").strip()))
        return AskPath(System(cat, [Question(q.name, q.text, q.answer)])), \
            f"the function {getattr(fn, '__name__', 'slow')} over {', '.join(reads)}"
    raise TypeError("slow= takes a SlowPath, a System, a decision part, a model (solvi.core.deciders.llm), a function state → answer, "
                    "or a compiled specification")


# ------------------------------------------------------------------------------------------------ build
def build(question, examples, *, catalog=None, learner=None, slow=None, reads=None, max_risk=None, max_error=None,
          signal=None, correct=None, novel="auto", budget=None, total=None, price=None, storage=None, split=None,
          seed=0, delta=0.10, min_slice=20, writer=None, inputs=None):
    """Fit System 1, calibrate its guarantee and the dispatcher, wire the store → a DecisionSystem (see the module docs).

    question: a Question, or the name of one a rule of `catalog` (or learner=) answers. examples: [(state, correct
    answer)] — a state is the dict `ask` takes. catalog: the parts System 1 computes facts with (and its answer rule, if
    it has one). learner: a function [(state, answer)] → a decision part answering the question (fitted here, and
    refitted without some options for the open-set gate). slow: the slow path (module docs); reads: the given keys a
    model slow path reads. max_risk= or max_error=: the promise. signal: the guarantee's signal (default: chosen, see
    the module docs). correct: a function (answer, label) → bool when a label is not simply the right answer (a label
    that says "right" / "wrong" of System 1's answer, a quote that overlaps). novel: "auto", True or False. budget /
    total: solvi.core.dispatch.Budget per decision / for the dispatcher's life; price: dollars per million input and output
    tokens (needed for dollars). storage: a path or a TraceStorage — every decision and the policy, hash-chained.
    split: (fit, guarantee, dispatch) shares. seed: the shuffle and the open-set simulation. delta: learn-then-test's
    confidence. min_slice: a slice of the slow path with fewer calibration examples goes to a person. writer, inputs:
    gone in 1.0 (they compiled a Spec here): a TypeError says to compile it first."""
    from ..core.dispatch import SIGNALS, THINK, Dispatcher
    from ..core.guarantees.guarantee import promise_text
    from ..core.store import open_storage
    from ..core.system import System
    if (max_risk is None) == (max_error is None):
        raise ValueError("give the promise as max_risk= (P(answered alone and wrong)) or max_error= (the error among the "
                         "answers given alone)")
    if novel not in ("auto", True, False):
        raise ValueError('novel is "auto", True or False')
    examples = [tuple(e) for e in examples]
    if not examples or not all(len(e) == 2 and isinstance(e[0], dict) for e in examples):
        raise ValueError("examples are [(state dict, correct answer), ...]")
    cat = catalog if catalog is not None else Catalog()
    qname = question if isinstance(question, str) else question.name
    if learner is not None:
        mode = "learner"
    elif qname in cat.rules:
        mode = "given"
    else:
        mode = "head"
        if isinstance(question, str):
            raise ValueError(f"no rule answers {qname!r} in the catalog: give the Question (its answer type) so a head "
                             "can be fitted, or learner=")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    shuffled = [examples[i] for i in order]
    n = len(shuffled)
    has_slow = slow is not None

    # --- the split: a fit share when System 1 is fitted here, the rest for calibration
    if split is not None:
        if len(split) != 3 or any(x < 0 for x in split) or sum(split) <= 0:
            raise ValueError("split is (fit, guarantee, dispatch) shares ≥ 0")
        fit_share = split[0] / float(sum(split))
        g_ratio = split[1] / max(1e-12, float(split[1] + split[2]))
    else:
        fit_share = SPLIT_FIT if mode != "given" else 0.0
        g_ratio = 0.5 if has_slow else 1.0
    n_fit = int(round(n * fit_share))
    fit_ex, rest = shuffled[:n_fit], shuffled[n_fit:]
    if mode != "given" and not fit_ex:
        raise ValueError("System 1 is fitted here and the fit share is empty: more examples, or split=")

    # --- System 1
    cat_question = None
    if mode != "learner":
        cat_question = question if isinstance(question, Question) else Question(qname, qname, _rule_answer(cat, qname))

    def make_system(ex):
        """System 1 fitted on `ex` → (System, Question)."""
        if mode == "learner":
            c = copy.copy(cat)
            c.parts, c.rules, c.constraints = dict(cat.parts), dict(cat.rules), dict(cat.constraints)
            c.types, c.readers = dict(cat.types), {k: dict(v) for k, v in cat.readers.items()}
            assert learner is not None
            part = learner(ex)
            q1 = part.question(c, name=qname, text=None if isinstance(question, str) else question.text)
            return System(c, [q1]), q1
        s = System(cat, [cat_question])
        if mode == "head":
            s.fit(qname, ex)
        return s, cat_question
    system, q = make_system(fit_ex)
    if mode == "learner":
        desc1 = f"the given learner's decision part, fitted on {len(fit_ex)} examples"
    elif mode == "head":
        feats = list(getattr(system.heads.get(qname), "features", []) or [])
        desc1 = (f"a ridge head (System.fit) fitted on {len(fit_ex)} examples, reading {len(feats)} facts"
                 + (f": {', '.join(feats[:12])}" + (f" and {len(feats) - 12} more" if len(feats) > 12 else "")
                    if feats else " (none: every input gets the same answer)"))
    else:
        desc1 = f"the catalog's rule for {qname!r}, as written"
    if mode == "given" and fit_ex:              # nothing is fitted on them: they calibrate
        rest, fit_ex = fit_ex + rest, []

    # --- the signal the guarantee reads
    sig_name, sig_desc, choose_on = signal, f"{signal!r} (given)", []
    if signal is None:
        probe = (fit_ex or rest)[:20]
        if mode != "head" and not _rows(system, qname, probe, "act", correct)[2]:
            sig_name, sig_desc = "act", "the act probability of System 1's decision (P(its answer is right))"
        elif not _constant(_rows(system, qname, probe, "confidence", correct)[0]):
            sig_name, sig_desc = "confidence", "the answer's confidence"
        else:
            if fit_ex:
                choose_on = fit_ex
            else:
                k = int(round(len(rest) * SPLIT_SELECT))
                choose_on, rest = rest[:k], rest[k:]
            best, auc, scores = _choose_signal(system, qname, choose_on, correct, q)
            if best is None:
                sig_name, sig_desc = "confidence", ("the answer's confidence (it does not vary, and no number the catalog "
                                                    "computes does either)")
            else:
                sig_name = best
                others = ", ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if k != best)
                sig_desc = (f"the fact {best!r}, chosen on {len(choose_on)} examples as the number that best separates right "
                            f"from wrong answers (AUROC {auc:.2f}" + (f"; others: {others}" if others else "")
                            + "); the answer's confidence does not vary")
            if choose_on is fit_ex:
                choose_on = []                  # counted as the fit share
    k = int(round(len(rest) * g_ratio))
    g_ex, d_ex = rest[:k], rest[k:]
    if has_slow and not d_ex:
        raise ValueError("a slow path is calibrated on examples System 1's guarantee did not see: none are left")
    if not g_ex:
        raise ValueError("no examples are left to calibrate System 1's guarantee on")

    # --- the promise on System 1 (open-set gate or plain guarantee)
    level = max_error if max_error is not None else max_risk
    opts = list(q.answer.options or [])
    can_novel = q.answer.kind == "choice" and len(opts) > 2 and mode in ("learner", "head")
    if novel is True and not can_novel:
        raise ValueError("novel=True needs a choice among more than two options and System 1 fitted here (a head or "
                         "learner=): the simulation refits it without some options")
    want_gate = novel is True or (novel == "auto" and can_novel)

    def promise_s1(cal_ex):
        """Calibrate System 1's promise on cal_ex → (gate or None, its description, novel description, reports)."""
        system.guarantee(qname, False)
        rep_all, ndesc = {}, None
        if want_gate:
            try:
                g, info = _open_set(make_system, system, q, qname, fit_ex, cal_ex, sig_name, correct, level, delta, seed)
                rep_all["openset"] = g.report
                system.guarantee(qname, promise=g, signal=sig_name)
                ndesc = (f"System 1 was refitted 3 times without a third of the options ({info['groups']} left out); its "
                         f"signal on their examples stands in for inputs no option fits. An open-set gate keeps the error "
                         f"among the answers given alone ≤ {level:g} on streams with a share of such inputs (estimated from "
                         f"the recent signals, at least {g.min_share:g}) and flags a change of the stream")
                if max_risk is not None:
                    ndesc += f" (max_risk {max_risk:g} is kept as the stricter max_error {max_risk:g})"
                r0 = g.report.get(f"at_share_{g.min_share:g}") or {}
                gd = {"promise": g.promise,
                      "how": (f"Calibrated on {len(cal_ex)} examples ({info['known']} known, {info['novel']} simulated new); "
                              f"at the minimum share the threshold is {r0.get('threshold', math.inf):.4g} and answers "
                              f"{r0.get('answered_known', 0):.1%} of the known examples alone."
                              if r0 else "No threshold keeps the promise: everything goes to a person.")}
                return g, gd, ndesc, rep_all
            except ValueError as e:
                if novel is True:
                    raise
                ndesc = f"not simulated: {e}"
        rep = system.guarantee(qname, cal_ex, max_risk=max_risk, max_error=max_error, signal=sig_name, delta=delta,
                               weak="warn", correct=None if correct is None else (lambda result, label, f=correct: f(result.answer, label)))
        rep_all["guarantee"] = rep
        how = (f"Calibrated on {rep['n']} examples: threshold {rep['threshold']:.4g}, answered alone {rep['answered']:.1%}, "
               f"error among them {rep['error']:.2%}, P(alone and wrong) {rep['risk']:.2%}"
               + (f"; AUROC of the signal {rep['separation']['auroc']:.2f}" if rep.get("separation", {}).get("auroc") is not None else "")
               + (f". {rep['why']}" if rep.get("why") else "."))
        return None, {"promise": rep["promise"], "how": how}, ndesc, rep_all

    gate, gdesc, novel_desc, calib = promise_s1(g_ex)
    novel_on = gate is not None
    wake = tuple(s for s in SIGNALS if s != "openset")
    slice_open, probe_desc = True, None
    if has_slow:
        # a dev-only probe: does System 1 hand the slow path anything? Its share of the examples goes through System 1 as
        # calibrated; the inputs it holds back on a signal that wakes the slow path are counted. Fewer than min_slice: no
        # slice is open to the slow path, and that share calibrates System 1 instead.
        probe = Dispatcher(system, None, question=qname, wake=wake)
        woken = {}
        for st, _ in d_ex:
            kinds = [k for k, _ in probe.signals(system.ask(st, [qname], store=False)) if k in THINK]
            if kinds:
                woken[kinds[0]] = woken.get(kinds[0], 0) + 1
        if gate is not None:
            gate.reset()
        n_open = sum(v for k, v in woken.items() if k in wake)
        held = ", ".join(f"{v} on the {k!r} slice" for k, v in sorted(woken.items())) or "none"
        if n_open < min_slice:
            slice_open = False
            moved = len(d_ex)
            if split is None:
                g_ex, d_ex = g_ex + d_ex, []
                gate, gdesc, novel_desc, calib = promise_s1(g_ex)
                novel_on = gate is not None
            probe_desc = (f"no slice is open to the slow path: of the {moved} examples set aside for the dispatcher, "
                          f"System 1 held back {held}, and only {n_open} on a signal that wakes the slow path (fewer than "
                          f"min_slice {min_slice})" + ("; that share calibrated System 1 instead, and the slow path only "
                                                        "checks System 1's answers after a drift flag" if split is None else
                                                        "; split= was given, so the shares stay as given"))
        else:
            probe_desc = f"the probe on the dispatcher's {len(d_ex)} examples: System 1 held back {held}"

    # --- the dispatcher
    slow_path, slow_desc = None, None
    if has_slow:
        if reads is None:
            reads = list(examples[0][0].keys())
        reads = [reads] if isinstance(reads, str) else list(reads)
        slow_path, slow_desc = _slow_path(slow, q, reads, novel_on, writer, inputs)
    if not slice_open:
        wake = ("drift", "supervise")
    d = Dispatcher(system, slow_path, question=qname, budget=budget, total=total, price=price, wake=wake,
                   unknown="human" if novel_on else "answer", on_disagree="human", storage=open_storage(storage))
    choices = {"examples": n, "seed": seed, "mode": mode, "system1": desc1, "signal": sig_desc or str(sig_name),
               "split": {"to fit System 1": len(fit_ex), "to choose the signal": len(choose_on),
                         "to calibrate System 1's guarantee": len(g_ex), "to calibrate the dispatcher": len(d_ex)},
               "guarantee": gdesc, "novel": novel_desc, "slow": slow_desc, "wake": list(wake),
               "budget": ("none" if budget is None and total is None else
                          "; ".join(x for x in (f"per decision {budget.to_dict()}" if budget is not None else "",
                                                f"in total {total.to_dict()}" if total is not None else "") if x))
               + (f"; price {price}" if price is not None else ""),
               "storage": (f"every decision and the policy, hash-chained, in {type(d.storage).__name__}"
                           if d.storage is not None else "not stored (storage= keeps every decision)")}
    choices["probe"] = probe_desc
    if has_slow and slice_open:
        rep = d.calibrate(d_ex, max_risk=max_risk, max_error=max_error, delta=delta, min_slice=min_slice,
                          correct=None if correct is None else (lambda ans, label, res: correct(ans, label)))
        calib["dispatch"] = rep
        if gate is not None:
            gate.reset()                         # the calibration examples are not the stream
        who = {"s1": "System 1's own answer", "s2": "the slow path's answer", "agree": "the slow path's answer when it "
               "agrees with System 1", "human": "a person"}
        short = {"s1": "System 1", "s2": "slow path", "agree": "slow path agreeing with System 1"}
        sl = {}
        for k, v in rep["slices"].items():
            acc = ", ".join(f"{short[a]} {x:.0%}" for a, x in v["accuracy"].items())
            text = who[v["answer"]] + (f" when its confidence ≥ {v['threshold']:.4g}" if v["threshold"] is not None else "")
            text += f" (right on this slice: {acc})" if acc else ""
            if v["answer"] == "human":
                text += f" — {v['why']}"
            sl[k] = {"n": v["n"], "text": text}
        choices["slices"] = sl
        choices["dispatch_promise"] = (f"{promise_text(rep['method'], rep['level'], rep['delta'])}; on the {rep['n']} "
                                       f"calibration examples: answered alone {rep['answered']:.1%}, P(alone and wrong) "
                                       f"{rep['risk']:.2%}" + (f" ({rep['why']})" if rep.get("why") else ""))
    return DecisionSystem(qname, system, d, gate, slow_path, choices, calib)


def _rule_answer(cat, qname):
    """The answer type of a rule's question when only its name is given: the rule's options, else yes/no."""
    opts = getattr(cat.rules[qname], "options", None)
    return Answer.choice(list(opts)) if opts else Answer.yes_no()


def _open_set(make_system, system, q, qname, fit_ex, g_ex, sig_name, correct, level, delta, seed, folds=3):
    """The leave-options-out simulation and the gate → (OpenSetGate, info)."""
    from ..core.guarantees.openset import OpenSetGate
    norm = q.answer.normalize
    ks, kr, _ = _rows(system, qname, g_ex, sig_name, correct)
    keep = [i for i, s in enumerate(ks) if math.isfinite(s)]
    ks, kr = [ks[i] for i in keep], [kr[i] for i in keep]
    labels = sorted({str(norm(y)) for _, y in fit_ex + g_ex})
    if len(labels) < folds:
        raise ValueError(f"fewer options in the examples ({len(labels)}) than folds ({folds})")
    random.Random(seed).shuffle(labels)
    groups = [labels[i::folds] for i in range(folds)]
    us = []
    for grp in groups:
        gs = set(grp)
        sub = [(s, y) for s, y in fit_ex if str(norm(y)) not in gs]
        out = [(s, y) for s, y in g_ex if str(norm(y)) in gs]
        if not sub or not out:
            continue
        sk, _ = make_system(sub)
        sig, _, _ = _rows(sk, qname, out, sig_name, lambda answer, label: False)   # no option of theirs is known
        us += [s for s in sig if math.isfinite(s)]
    gate = OpenSetGate.calibrate(ks, kr, us, max_error=level, delta=delta, seed=seed)
    return gate, {"known": len(ks), "novel": len(us), "groups": " / ".join(str(len(g)) for g in groups)}


def __getattr__(name):
    if name == "AutoSystem":                       # the 0.9 name, until 1.1
        from .._deprecate import _warn_from_caller
        _warn_from_caller("solvi.auto.AutoSystem was renamed in 1.0: use solvi.solutions.decisions.DecisionSystem (what "
                          "solvi.build returns); the old name is removed in 1.1", skip=1)
        return DecisionSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DecisionSystem", "build"]
