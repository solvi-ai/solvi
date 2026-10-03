"""DecisionPart: a decider bound to one question — a catalog part with its gates, guarantees and files. (Part of
solvi.core.deciders, which re-exports every name.)"""
from __future__ import annotations

import dataclasses
import inspect
import math
import warnings

import numpy as np

from ... import _deprecate
from ..catalog import Decision, Quote, Unknown, gone_in_1_0
from ..provenance import ESCALATED, INSTRUCTION
from ..provenance import digest
from ..catalog import Answer, Question
from .state import _is_text, _single, _text, jsonable, state_text
from .wire import Logits, _respan
from .backends import LONG_CPU_TOKENS, SECTION_TOKENS
from .adapt import ADAPTERS, _Spec, lora_key
from .gate import Facts, GroupBy, _group_guard, _group_promise, _shown, _threshold, _vkey, confidence_source, group_record, guard_promise, no_separation, one_source


class DecisionPart:
    """A decider bound to one question (name, task, options, kind, the facts it reads). Callable as a catalog function; it
    is also the model recorded in the trace: `fingerprint()` covers the checkpoint and this question's adaptation and
    thresholds only, so teaching one decision does not mark the others as changed."""

    # the thresholds by the question-level names (decision(min_confidence=, min_act=)); stored as escalate_below /
    # act_threshold, the keys of calibration files and snapshots
    min_confidence = property(lambda self: self.escalate_below,
                              lambda self, v: setattr(self, "escalate_below", v),
                              doc="Escalate below this calibrated confidence (stored as `escalate_below`).")
    min_act = property(lambda self: self.act_threshold, lambda self, v: setattr(self, "act_threshold", v),
                       doc="Escalate below this act probability (stored as `act_threshold`).")
    long_len = _deprecate.removed_attr("long_len", "max_len_long", "DecisionPart")

    def __init__(self, model, name, task, text_fact, options, descriptions=None, multi=False, other=None, *, kind=None,
                 as_bool=False, escalate_below=None, act_threshold=None, use_act=None, score_value="median",
                 option_order="canonical", permutations=4, min_margin=None, long=None, top_k=None, rerank=False,
                 perturb=0, retrieve_query=None, **prim):
        self.model = model
        self.long, self.top_k, self.rerank = long, None if top_k is None else max(1, int(top_k)), bool(rerank)
        self.retrieve_query = str(retrieve_query).strip() or None if retrieve_query is not None else None
        if self.retrieve_query and long is None:
            raise ValueError('retrieve_query is what long="retrieve" (or long="full" beyond its length) searches by: '
                             "set long=")
        self.max_len_long = getattr(model, "max_len_long", None) if long == "full" else None   # read whole up to (long="full")
        if long == "full" and self.max_len_long is None:
            raise ValueError('long="full" needs a checkpoint with a long-input length ("max_len_long"); use long="retrieve"')
        self.spec = _Spec(task, options, descriptions, multi, other, kind, as_bool, score_value, **prim)
        sp = self.spec
        self._shown = None if sp.values is None else list(sp.values)   # the caller's order, for options and probs
        self._shown_labels = list(sp.options)
        if option_order == "canonical" and sp.kind in ("choice", "multi") and len(sp.real) > 1:
            ordered = sorted(sp.real, key=str) + ([sp.other] if sp.other is not None else [])
            if ordered != list(sp.options):         # the part asks, adapts and answers in the sorted order
                options, descriptions, other = ordered, dict(sp.descriptions), (sp.other if sp.other is not None else False)
                self.spec = _Spec(task, options, descriptions, multi, other, kind, as_bool, score_value, **prim)
        self._spec_args = (task, descriptions, multi, other, kind, as_bool, score_value, prim)
        if self.spec.kind not in ("choice", "multi") or len(self.spec.real) < 2:
            option_order = "given"                  # scores, numbers, rankings: the order is the meaning
        self.option_order, self.permutations, self.min_margin = option_order, max(1, int(permutations)), min_margin
        self.perturb = max(0, int(perturb or 0))  # re-ask without instruction-like sentences (solvi.core.deciders.perturb), up to k times
        self._orders = self._option_orders()
        self.facts = [text_fact] if isinstance(text_fact, str) else list(text_fact)
        self.escalate_below, self.act_threshold, self.use_act = escalate_below, act_threshold, use_act
        self.guarantee = None                   # what the escalation threshold promises (act_guard / calibrate_for)
        self.conformal_set = None               # the answer-set quantile (conformal)
        self.groups = None                      # thresholds per group (act_guard(groups=...)): {"by", "nodes", "signal"}
        self.correction_memory = None           # a solvi.core.knowledge.memory.CorrectionMemory consulted on every decision (solvi.core.knowledge.memory.attach)
        self.asked = 0                          # decisions made by this part on its own (calls())
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = task
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts])
        self.__solvi_model__ = self
        self.__solvi_provenance__ = "decided"
        self.__solvi_options__ = None if self._shown is None else list(self._shown)
        self.__solvi_decision__ = self

    # identity recorded in the trace
    @property
    def model_id(self):
        return self.model.model_id

    @property
    def deterministic(self):
        """Does the model give the same output for the same input (replay re-runs it)? False for an LLM (solvi.core.deciders.llm):
        replay then checks the recorded output instead."""
        return getattr(self.model, "deterministic", True)

    @property
    def available(self):
        return getattr(self.model, "available", True)

    @property
    def options(self):
        """The values this decision can take (a bool question: True, False; with "not stated": also Unknown; None for a
        number or a span)."""
        return None if self._shown is None else list(self._shown)

    @property
    def labels(self):
        """The options as the model reads them (a bool question: "yes", "no")."""
        return list(self.spec.options)

    @property
    def kind(self):
        return self.spec.kind

    @property
    def multi(self):
        return self.spec.multi

    @property
    def task(self):
        return self.spec.task

    @property
    def adaptation(self):
        return self.model.adaptations.get(self.spec.key)

    @property
    def lora(self):
        """This question's adapter (its Adapter slot: a solvi.experimental.lora.LoraAdapter, experimental) or None."""
        loras = getattr(self.model, "loras", None)
        return loras.get(lora_key(self.spec)) if loras else None

    def _batch_group(self):
        """(model, facts) when this part may share a forward pass with other questions on the same facts (the strategist's
        batches, solvi.core.runtime.plan_batches); None when it needs a pass of its own (a model that cannot batch, the
        pointer)."""
        if not self.model.batchable or self.spec.pointer:
            return None
        return self.model, self.facts

    def fingerprint(self):
        a = self.adaptation
        th = {k: v for k, v in (("escalate_below", self.escalate_below), ("act_threshold", self.act_threshold),
                                ("use_act", self.use_act), ("guarantee", self.guarantee),
                                ("conformal", self.conformal_set), ("min_margin", self.min_margin),
                                ("perturb", self.perturb or None),
                                ("groups", None if self.groups is None else
                                 (self.groups["by"].describe(), sorted((list(k), v["threshold"])
                                                                       for k, v in self.groups["nodes"].items()))),
                                ("option_order", None if self.option_order != "average" else
                                 (self.option_order, self.permutations)),
                                ("long", self.long_key()),
                                ("memory", None if self.correction_memory is None else self.correction_memory.fingerprint()),
                                ("lora", None if self.lora is None else self.lora.hash))
              if v is not None}
        if th:
            return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None, th)
        return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None)

    def text_of(self, facts):
        """The input this part reads, from a dict of facts: text facts as they are (joined by new lines), a state by
        state_text (several facts: {fact: value})."""
        vals = [facts[f] for f in self.facts]
        if all(_is_text(v) for v in vals):
            return "\n".join(_text(v) for v in vals)
        if len(vals) == 1:
            return self.model.text(vals[0])
        return state_text({f: (v.value if isinstance(v, Quote) else v) for f, v in zip(self.facts, vals)},
                          self.model.state_format)

    def __call__(self, *args, **kw):
        vals = dict(zip(self.__signature__.parameters, args))
        vals.update(kw)
        text = self.text_of(vals)
        self.asked += 1
        return self._bind(self._one(text, self._ctx(text, vals=vals)), vals)

    def _ctx(self, text, vals=None, raw=None):
        """What _finish needs besides the model's output: the input's text and, with thresholds per group, its group."""
        c = {"text": text}
        if self.groups is not None:
            c["path"] = self.groups["by"].path(vals, raw)
        return c

    def _input_text(self, x):
        return self.text_of(x) if isinstance(x, Facts) else self.model.text(x)

    def _bind(self, d, vals):
        """Point a span's / the evidence's quotes into the fact they were read from: a decision reading one given text fact
        (the pointer's offsets are into that text); otherwise a span escalates and evidence is dropped."""
        if not (isinstance(d.value, Quote) or d.evidence):
            return d
        f = self.facts[0] if len(self.facts) == 1 else None
        v = vals.get(f) if f is not None else None
        if not isinstance(v, str):
            d.evidence = []
            if isinstance(d.value, Quote) and d.escalate is None:
                d.escalate = f"a span is read from one text fact; {self.__name__} reads {self.facts}"
            return d
        if isinstance(d.value, Quote):
            d.value = dataclasses.replace(d.value, source=f)
        d.evidence = [dataclasses.replace(e, source=f) for e in d.evidence]
        return d

    def in_pass(self, siblings, args, names=None):
        """Called by the runtime for a part in a shared pass (flow.batches): score every sibling's question about the same
        input in one forward pass (cached, so the siblings read it) and return this part's decision; the decision's extra
        records the pass."""
        text = self.text_of(args)
        ctx = self._ctx(text, vals=args)
        m = self.model
        self.asked += 1
        if self.long is not None and self._too_long(text):   # a long text: this part retrieves and decides on its own
            d = self._bind(self._one(text, ctx), args)
            d.extra["pass"] = {"with": list(names) if names else [s.__name__ for s in siblings], "shared": False}
            return d
        if (len(siblings) > 1 and m.batchable and all(s.model is m for s in siblings)
                and all(s.option_order != "average" for s in siblings)):
            z, a, shared = m._raw_pass([s.spec for s in siblings], text)[siblings.index(self)]
        else:
            (z, a), shared = self._raw([text])[0], False
        if shared:
            ctx["specs"] = [s.spec for s in siblings]             # the pass read the input after all their questions
        d = self._bind(self._finish(m._decision(self.spec, z), a, ctx=ctx), args)
        d.extra["pass"] = {"with": list(names) if names else [s.__name__ for s in siblings], "shared": shared}
        return d

    def __repr__(self):
        return f"DecisionPart({self.__name__!r}, {self.spec.kind}, options={self.options}, model={self.model.model_id!r})"

    def _group_threshold(self, ctx):
        """With thresholds per group: (the input's group path, the node whose threshold applies, its info) — None when
        the input does not say which group it is in."""
        path = (ctx or {}).get("path")
        if path is None:
            return None
        from ..calibration import node_of
        node = node_of(path, self.groups["nodes"])
        return path, node, self.groups["nodes"][node]

    def _finish(self, d, act, threshold=None, ctx=None):
        """Act or escalate. threshold: a combination's shared threshold (solvi.core.deciders.combine) in place of this part's own
        act_threshold / escalate_below — on the part's signal (see _signal); −inf applies only the other safeguards.
        ctx: the input (see _ctx) — with thresholds per group, the threshold of the input's group applies."""
        grp = None
        if threshold is None:
            eb, at = self.escalate_below, self.act_threshold
            if self.groups is not None:
                grp = self._group_threshold(ctx)
                if grp is not None and self.groups["signal"] == "act":
                    at = grp[2]["threshold"]
                elif grp is not None:
                    eb = grp[2]["threshold"]
            d = self.model._finish(self.spec, d, act, eb, at, self.use_act)
            if self.groups is not None and grp is None:          # no group, no threshold that holds for it
                d.escalate = (f"group unknown: the thresholds are per group ({self.groups['by'].label()}) and this input "
                              f"does not give {self.groups['by'].names}; would have answered {d.value!r}")
        else:
            d = self.model._finish(self.spec, d, act, -math.inf, -math.inf, self.use_act)
        if self.option_order == "canonical" and d.probs:       # probabilities in the caller's order of the options
            at = {o: n for n, o in enumerate(self._shown_labels)}
            d.probs = dict(sorted(d.probs.items(), key=lambda kv: at.get(kv[0], len(at))))
            if self.spec.multi and isinstance(d.value, tuple):
                d.value = tuple(sorted(d.value, key=lambda o: at.get(o, len(at))))
        if self.min_margin is not None and not self.spec.multi and len(d.probs) > 1:
            (a1, p1), (a2, p2) = sorted(d.probs.items(), key=lambda kv: -kv[1])[:2]
            d.extra["margin"] = p1 - p2
            if d.escalate is None and p1 - p2 < self.min_margin:
                d.escalate = (f"margin {p1 - p2:.2f} < {self.min_margin:g} between {a1!r} ({p1:.2f}) and {a2!r} ({p2:.2f}); "
                              f"would have answered {d.value!r}")
        if threshold is not None:
            name, s = self._signal(d)
            if d.escalate is None and s < threshold:
                d.escalate = (f"{ESCALATED}: " if name == "act" else "") + \
                    f"{name} {s:.2f} < {threshold:.2f} (shared threshold); would have answered {d.value!r}"
        elif self.guarantee is not None:
            d.extra["guarantee"] = dict(self.guarantee) if grp is None else group_record(self.guarantee, *grp)
            want, got = self.guarantee.get("probabilities"), confidence_source(d)
            if want is not None and got is not None and d.escalate is None and (got == "logprobs") != (want == "logprobs"):
                d.escalate = (f"probabilities from {'log-probabilities' if got == 'logprobs' else 'the numbers the model wrote'}"
                              f", but the threshold was calibrated on "
                              f"{'log-probabilities' if want == 'logprobs' else 'the numbers the model wrote'}: the "
                              f"guarantee does not cover it; would have answered {_shown(d.value)!r}")
        if self.long is None and (ctx or {}).get("text") is not None:
            self._cut(d, self.model.truncation(ctx.get("specs") or self.spec, ctx["text"]))
        if self.perturb and d.escalate is None and (ctx or {}).get("text") is not None:
            quiet = {**ctx, "text": None}             # a variant is gated like the input itself: same thresholds and
            self._perturbed(d, ctx["text"], lambda dv, a: self._finish(dv, a, threshold, quiet))   # group, no re-asking
        if self.correction_memory is not None and (ctx or {}).get("text") is not None:
            self.correction_memory.apply(d, ctx["text"], alone=threshold is None and not ctx.get("combined"))
        if self.conformal_set is not None and d.probs:
            cands = self.candidates(d)
            if (d.extra.get("memory") or {}).get("action") == "answered" and d.value not in cands:
                cands.append(d.value)                 # the memory's answer is a candidate, whatever the model's set
            d.extra["candidates"] = cands
            if d.escalate:
                d.escalate += f"; candidates at {self.conformal_set['coverage']:.0%}: {cands!r}"
        return d

    def _cut(self, d, cut):
        """An input that did not fit the pass was read cut: say so in the decision (extra["truncated"]) and warn once per
        part. The answer stands — a classification often needs only the start of a text — but a fact beyond the cut was
        not read, and the trace now shows that it could not have been."""
        if not cut:
            return
        d.extra["truncated"] = cut
        self.model._warn_once(("truncated", self.__name__),
                              f"{self.__name__}: the input has {cut['input_tokens']} tokens and one pass reads "
                              f"{cut['read_tokens']} of them (max_len {cut['max_len']}, the question takes "
                              f"{cut['question_tokens']}): the rest was cut, and the decision's extra[\"truncated\"] says "
                              "so. long=\"retrieve\" reads a long input by its relevant sections.")

    def _perturbed(self, d, text, gate=None):
        """The perturb=k safeguard: ask again on up to k variants of the input without its instruction-like sentences
        (solvi.core.deciders.perturb.variants — deterministic rules); escalate when an answer differs, or when the answer is the same
        but the model would not have given it alone without those sentences (`gate`: the part's own act / confidence
        gate applied to a variant's decision — an instruction can leave the answer and lift the model's confidence in
        it). Records extra["perturb"]: {"variants", "calls" (extra forward passes), "removed" (per variant), "answers",
        "flipped", "unsure" (a variant escalated)}. An input without such sentences has no variants and costs nothing;
        an input that is nothing but such sentences has no variant to compare with and escalates
        (extra["perturb"]["only_instruction"])."""
        from .perturb import variants
        vs = variants(text, self.perturb)
        if not vs:
            from .perturb import instruction_spans
            only = [text[a:b] for a, b in instruction_spans(text)] if isinstance(text, str) else []
            if only:                                  # nothing is left without them: no answer to compare with
                d.extra["perturb"] = {"variants": 0, "calls": 0, "removed": [only], "answers": [], "flipped": False,
                                      "unsure": False, "only_instruction": True}
                d.escalate = (f"{INSTRUCTION}: " + "; ".join(repr(r) for r in only) + " (the input is nothing else: no "
                              f"answer without it to compare with); would have answered {_shown(d.value)!r}")
            return d
        outs = self._read([v.text for v in vs])
        base, flip, unsure, answers = _vkey(d.value), None, None, []
        for v, (z, a) in zip(vs, outs):
            dv = self.model._decision(self.spec, z)
            val = dv.value
            answers.append(val)
            if flip is None and _vkey(val) != base:
                flip = (v, val)
            elif gate is not None and flip is None and unsure is None:
                why = gate(dv, a).escalate
                if why:
                    unsure = (v, why)
        d.extra["perturb"] = {"variants": len(vs), "calls": len(vs) * len(self._orders if self.option_order == "average"
                                                                            else [0]),
                              "removed": [v.removed for v in vs], "answers": [jsonable(_shown(a)) for a in answers],
                              "flipped": flip is not None, "unsure": flip is None and unsure is not None}
        if flip is not None:
            d.escalate = (f"{INSTRUCTION}: " + "; ".join(repr(r) for r in flip[0].removed)
                          + f" (without it: {_shown(flip[1])!r}); would have answered {_shown(d.value)!r}")
        elif unsure is not None:
            d.escalate = (f"{INSTRUCTION}: " + "; ".join(repr(r) for r in unsure[0].removed)
                          + f" (without it the model does not answer alone: {unsure[1].split('; would have answered')[0]})"
                          + f"; would have answered {_shown(d.value)!r}")
        return d

    def _signal(self, d):
        """The signal a threshold applies to for a finished decision → ("act", the act probability) when the model gave
        one and the part uses it, else ("confidence", the calibrated confidence) — as act_guard's signal="auto"."""
        if d.extra.get("act") is not None and self.use_act is not False:
            return "act", float(d.extra["act"])
        return "confidence", float(d.conf)

    def candidates(self, d):
        """The conformal answer set of a decision (after conformal(...)): the answers that cannot be ruled out at the
        calibrated coverage, most probable first — a short list for the person who handles an escalation. Never empty:
        when no answer passes (an unsure decision — the one that escalates), the most probable answer is listed; a
        larger set only covers more."""
        from ..calibration import set_scores
        keys = list(d.probs)
        s = set_scores([d.probs[k] for k in keys], self.conformal_set["ordinal"], Unknown in keys)
        keep = sorted((i for i in range(len(keys)) if s[i] <= self.conformal_set["quantile"]),
                      key=lambda i: -d.probs[keys[i]])
        if not keep and keys:
            keep = [max(range(len(keys)), key=lambda i: d.probs[keys[i]])]
        return [keys[i] if keys[i] is Unknown else self.spec.out(keys[i]) for i in keep]

    def _option_orders(self):
        """The option lists the model is asked with: [the given one], [sorted], or `permutations` rotations."""
        real = list(self.spec.real)
        if self.option_order == "canonical":
            return [sorted(real, key=str)]
        if self.option_order == "average":
            k, n = len(real), min(self.permutations, len(real))
            return [real[s:] + real[:s] for s in sorted({(i * k) // n for i in range(n)})]
        return [real]

    def _raw(self, texts):
        """[text] → [(logits in the part's option order, act logit)], asked with each of the part's option orders and
        averaged (the act logit too); the question's adaptation applies afterwards, to the part's own spec. long="full":
        a text that does not fit an ordinary pass is read whole, up to the checkpoint's long-input length."""
        texts = list(texts)
        if self.max_len_long and texts:
            n = [self.model.count_tokens(t) for t in texts]
            big = [i for i, x in enumerate(n) if x > self._pass_budget()]
            if big:
                self._cpu_warning(max(n))
                out = [None] * len(texts)
                small = [i for i in range(len(texts)) if i not in set(big)]
                for ix, rl in ((big, self.max_len_long), (small, 0)):
                    if ix:
                        for i, r in zip(ix, self._raw_at([texts[i] for i in ix], rl)):
                            out[i] = r
                return out
        return self._raw_at(texts, 0)

    def _cpu_warning(self, tokens):
        m = self.model
        if tokens > LONG_CPU_TOKENS and callable(getattr(m, "on_cpu", None)) and m.on_cpu():
            m._warn_once("cpu", (
                f"{m.model_id}: long=\"full\" reads a {tokens}-token text whole on a CPU: a pass costs more than in "
                "proportion to its length, so this is many times a 512-token pass. Use a GPU, or long=\"retrieve\" "
                "with a larger max_len (e.g. max_len=2048), which reads the text's relevant sections."))

    def _raw_at(self, texts, read_len=0):
        if self.option_order != "average":           # given, or canonical (the spec itself is in sorted order)
            return self.model._raw_full([(self.spec, t) for t in texts], read_len=read_len)
        task, desc, multi, other, kind, as_bool, sv, prim = self._spec_args
        opts = (lambda o: o + [self.spec.other] if self.spec.other is not None and self.spec.other not in o else o)
        specs = [_Spec(task, opts(list(o)), desc, multi, other, kind, as_bool, sv, **prim) for o in self._orders]
        idx = {o: i for i, o in enumerate(self.spec.real)}
        runs = [self.model._raw_full([(sp, t) for t in texts], read_len=read_len) for sp in specs]
        out = []
        for j in range(len(texts)):
            zs, us, acts = [], [], []
            for sp, run in zip(specs, runs):
                z, a = run[j]
                back = np.empty(len(self.spec.real))
                for pos, o in enumerate(sp.real):
                    back[idx[o]] = z[pos]
                zs.append(back)
                us.append(getattr(z, "unknown", None))
                acts.append(a)
            m = np.mean(zs, axis=0).view(Logits)
            m.unknown = None if any(u is None for u in us) else float(np.mean(us))
            m.pointer = getattr(runs[0][j][0], "pointer", None)
            m.escalate = next((w for w in (getattr(run[j][0], "escalate", None) for run in runs) if w), None)
            out.append((m, None if any(a is None for a in acts) else float(np.mean(acts))))
        return out

    def _one(self, text, ctx=None):
        ctx = ctx if ctx is not None else self._ctx(text)
        if self.long is not None and self._too_long(text):
            return self._retrieve(text, ctx)
        z, a = self._raw([text])[0]
        return self._finish(self.model._decision(self.spec, z), a, ctx=ctx)

    def _read(self, texts):
        """[text] → [(logits, act logit)] as a decision reads them: with long="retrieve", a text over the budget by the
        window of its retrieved sections (the signal the part answers on) — so calibration, fit / teach / adapt, the
        memory's features and the perturb re-asks see what a decision sees."""
        texts = list(texts)
        if self.long is not None:
            texts = [self.long_input(t) if self._too_long(t) else t for t in texts]
        return self._raw(texts)

    # --- long texts (long="retrieve" / "full", solvi.core.deciders.longdoc)
    def long_key(self):
        """How this part reads long texts, as its fingerprint records it: None, ("retrieve", top_k, rerank) or ("full",
        max_len_long, top_k, rerank) — top_k resolved (top_k=None: from the budget)."""
        if self.long is None:
            return None
        q = (("query", self.retrieve_query),) if self.retrieve_query else ()     # absent: the key is what it was
        if self.long == "full":
            return ("full", self.max_len_long, self.sections_k(), self.rerank) + q
        return (self.long, self.sections_k(), self.rerank) + q

    def sections_k(self):
        """The sections retrieve reads: top_k, or with top_k=None budget / 170 (at least 3) — sections of ≈ 170 tokens."""
        if self.top_k is not None:
            return self.top_k
        return max(3, round(self.budget() / SECTION_TOKENS))

    def long_input(self, text):
        """The text a decision reads for an input that does not fit an ordinary pass: the whole text (long="full", when
        it fits max_len_long), else the window of its retrieved sections."""
        if self.long == "full" and self.model.count_tokens(text) <= self.budget():
            return text
        return self._window(text)[2].text
    def _prompt_tokens(self):
        sp = self.spec
        return self.model.count_tokens(" ".join([sp.task] + [str(o) for o in sp.options] +
                                                [str(v) for v in (sp.descriptions or {}).values() if v])) + len(sp.options) + 8

    def budget(self):
        """The tokens of input this decision can read in one pass: max_len (long="full": max_len_long) minus its
        question."""
        return max(32, (self.max_len_long or self.model.max_len) - self._prompt_tokens())

    def _pass_budget(self):
        return max(32, self.model.max_len - self._prompt_tokens())

    def _too_long(self, text):
        """Does the text not fit an ordinary pass (max_len minus the question)? Then long= decides how it is read."""
        return self.model.count_tokens(text) > self._pass_budget()

    def _window(self, text):
        """The top_k sections of a long text → (the LongDocument, [(section, score)], the window read, reranked?)."""
        from .longdoc import LongDocument
        budget = self.budget()
        k = self.sections_k()
        doc = LongDocument(text, max_tokens=max(16, budget // k), count=self.model.count_tokens)
        sp = self.spec
        query = self.retrieve_query or " ".join([sp.task] + [str(o) for o in sp.real] +
                                                [str(v) for v in (sp.descriptions or {}).values() if v])
        rr = self._relevance if self.rerank else None
        sel = doc.select(query, k=k, budget=budget, rerank=rr)
        return doc, sel, doc.window([s for s, _ in sel]), rr is not None

    def _retrieve(self, text, ctx=None):
        """Decide on the top_k sections of a long text; spans and evidence mapped back into the text; the sections read
        in extra["long"]. ctx: as for _finish (the group, and the whole text for perturb and the memory)."""
        d, a = self._windowed(text)
        return self._finish(d, a, ctx=ctx if ctx is not None else self._ctx(text))

    def _initial(self, text):
        """→ (the model's decision before any safeguard, its act logit), as the part reads the text: a long text
        (long="retrieve") by its retrieved window, spans and evidence mapped back. What a combination (solvi.core.deciders.combine) and
        DecideModel.decide_pass start from."""
        if self.long is not None and self._too_long(text):
            return self._windowed(text)
        z, a = self._raw([text])[0]
        return self.model._decision(self.spec, z), a

    def _windowed(self, text):
        """The decision on a long text's window, before the safeguards → (Decision, act logit). long="full": the whole
        text when it fits max_len_long (offsets are the text's own), else its window within max_len_long (the fallback
        is recorded)."""
        sp = self.spec
        if self.long == "full":
            n = self.model.count_tokens(text)
            if n <= self.budget():
                z, a = self._raw([text])[0]
                d = self.model._decision(sp, z)
                d.extra["long"] = {"mode": "full", "tokens": n, "max_len": self.max_len_long}
                return d, a
        doc, sel, win, rr = self._window(text)
        z, a = self._raw([win.text])[0]
        d = self.model._decision(sp, z)
        score = {s.index: sc for s, sc in sel}
        d.extra["long"] = {"read": len(sel), "of": len(doc), "by": "bm25+decider" if rr else "bm25",
                           "sections": [[s.start, s.end, s.heading, round(float(score[s.index]), 6)] for s in win.sections]}
        if self.retrieve_query:                      # what the sections were searched by, when not the question itself
            d.extra["long"]["query"] = self.retrieve_query
        if self.long == "full":                      # longer than max_len_long: retrieved within it
            d.extra["long"].update(mode="full", fallback="retrieve", tokens=n, max_len=self.max_len_long)
        if isinstance(d.value, Quote):
            got = win.to_doc(d.value.start, d.value.end)
            if got is None:
                d.escalate = d.escalate or "the span crosses two sections of the long text that are not neighbours"
            else:                                    # over neighbouring sections: the document's own text between them
                d.value = dataclasses.replace(d.value, start=got[0], end=got[1],
                                              value=_respan(d.value.value, win.text[d.value.start:d.value.end],
                                                            text[got[0]:got[1]]))
        ev = []
        for e in d.evidence:
            if isinstance(e, Quote):
                got = win.to_doc(e.start, e.end)
                if got is not None:
                    ev.append(dataclasses.replace(e, start=got[0], end=got[1],
                                                  value=_respan(e.value, win.text[e.start:e.end], text[got[0]:got[1]])))
            else:
                ev.append(e)
        d.evidence = ev
        return d, a

    def _relevance(self, texts):
        """The decider's own relevance of passages to this question: p(yes) of "Does this passage help answer: …?"."""
        task = f"Does this passage help answer the question: {self.spec.task}"
        ds = self.model.decide(list(texts), task, ["yes", "no"], kind="noul")
        return [float(d.probs.get("yes", 0.0)) for d in ds]

    # the model's methods for this decision
    def decide(self, text):
        """An input (a text, a state, or Facts by name) → Decision; a list of inputs → a list."""
        one = isinstance(text, Facts) or _single(text)
        xs = [text] if one else list(text)
        ts = [self._input_text(x) for x in xs]
        self.asked += len(xs)
        if self.long is not None and any(self._too_long(t) for t in ts):
            out = [self._one(t, ctx=self._ctx(t, vals=x if isinstance(x, Facts) else None, raw=x)) for t, x in zip(ts, xs)]
        else:
            out = [self._finish(self.model._decision(self.spec, z), a,
                                ctx=self._ctx(t, vals=x if isinstance(x, Facts) else None, raw=x))
                   for (z, a), t, x in zip(self._raw(ts), ts, xs)]
        return out[0] if one else out

    def score(self, text):
        """The probabilities a decision answers with (see decide) → {option: probability}; a list → a list."""
        d = self.decide(text)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def _kw(self):
        sp = self.spec
        kw = dict(task=sp.task, options=sp.options, descriptions=sp.descriptions, multi=sp.multi, other=sp.other or False,
                  kind=sp.kind)
        for x, v in (("unknown", sp.unknown), ("k", sp.k), ("bins", sp.edges), ("unit", sp.unit), ("evidence", sp.evidence)):
            if v not in (None, False, 0):         # the same question key as the part's own spec
                kw[x] = v
        if sp.kind == "number":
            kw.update(coverage=sp.coverage, integer=sp.integer)
        return kw

    # adapt / fit / teach are fitted on the logits the part decides on (self._read: every option order averaged with
    # option_order="average", a long input's retrieved window), not on the model's single-order logits of the whole text
    def adapt(self, texts):
        """Label-bias correction from unlabelled inputs of the domain (see DecideModel.adapt)."""
        ts = [self._input_text(t) for t in texts]
        if self.spec.kind == "span" or not ts:
            return self.model.adapt(ts, **self._kw())             # raises the model's error
        return self.model.adapt(ts, logits=[z for z, _ in self._read(ts)], **self._kw())

    def fit(self, examples, lam=1.0, folds=4):
        """Few-shot "S" from [(input, correct)] (see DecideModel.fit)."""
        ex = [(self._input_text(t), self.spec.label(y)) for t, y in examples if y is not Unknown]
        return self.model.fit(ex, lam=lam, folds=folds, logits=[z for z, _ in self._read([t for t, _ in ex])],
                              **self._kw())

    def teach(self, text, correct):
        """One correction, absorbed at once (see DecideModel.teach). → ms."""
        t = self._input_text(text)
        return self.model.teach(t, self.spec.label(correct), logits=self._read([t])[0][0], **self._kw())

    def reset(self):
        self.model.reset(**self._kw())

    def calls(self):
        """What the part cost since it was made, as a combination's calls(): {"asked" (decisions made by the part on its
        own; inside a combination they count there), "calls" ({"0:<name>": the model's calls}), "calls_per_question"
        (1 per decision: one model)}. A scorer's token count is its own `usage`."""
        return {"asked": self.asked, "calls": {f"0:{self.__name__}": self.asked},
                "calls_per_question": 1.0 if self.asked else 0.0}

    # --- an adapter for this question (the Adapter slot; LoRA: experimental, solvi.experimental.lora)
    adapt_lora = gone_in_1_0("adapt_lora()", "solvi.experimental.lora.adapt_lora(part, examples, ...) — training an adapter is "
                             "experimental and lives in solvi.experimental.lora (solvi.experimental.lora later)", "DecisionPart")
    remove_lora = gone_in_1_0("remove_lora()", "solvi.experimental.lora.remove_lora(part)", "DecisionPart")

    def save_lora(self, path):
        """Write this question's adapter (a .safetensors file with its config, the question and the checkpoint it was
        trained on) → path. save_calibration also writes it, next to the calibration file."""
        ad = self.lora
        if ad is None:
            raise ValueError(f"{self.__name__} has no LoRA adapter (adapt_lora or load_lora first)")
        return ad.save(path)

    def load_lora(self, path, strict=True):
        """Load an adapter written by save_lora (or tools/adapt_lora_gpu.py) for this question: afterwards the part
        answers exactly as right after training. Refuses (ValueError) an adapter for another question or checkpoint
        unless strict=False; needs the torch backend and peft (`solvi[lora]`). Clears the question's adaptation and
        thresholds like solvi.experimental.lora.adapt_lora (load the calibration after it). → self."""
        self._load_adapter("lora", path, strict)
        return self

    def _load_adapter(self, kind, path, strict=True, expect=None):
        """Read an adapter file of `kind` into this question's adapter slot (expect: the hash a calibration file names).
        The module that reads that kind is looked up by name (ADAPTERS), so solvi.core.deciders imports no adapter module."""
        import importlib
        if kind not in ADAPTERS:
            raise ValueError(f"an adapter of an unknown kind {kind!r} (known: {', '.join(sorted(ADAPTERS))})")
        return importlib.import_module(ADAPTERS[kind]).load(self, path, strict, expect=expect)

    def _answer_key(self, y):
        """An answer or a label as calibration compares them: "not stated" as itself, a span by its text (a Quote or
        the text), a ranking as a tuple, anything else as this question's label."""
        if y is Unknown:
            return Unknown
        if self.kind == "span":
            return y.value if isinstance(y, Quote) else y
        if self.kind == "rank":
            return tuple(y)
        return self.spec.label(y)

    def _labelled(self, examples, signal):
        """Decide labelled examples [(input, correct)] → (the signal per example, correct 0/1 per example, "act" |
        "confidence", [Decision]). "Not stated" (solvi.Unknown) is a label like any other."""
        ex = [(self._input_text(t), y) for t, y in examples]
        if not ex:
            raise ValueError("calibration needs labelled examples")
        if signal not in ("auto", "act", "confidence"):
            raise ValueError('signal must be "auto", "act" or "confidence"')
        gold = [self._answer_key(y) for _, y in ex]
        conf, act, ok, ds = [], [], [], []
        for (z, a), y in zip(self._read([t for t, _ in ex]), gold):    # a long input: the window a decision reads
            d = self.model._decision(self.spec, z)
            ds.append(d)
            ok.append(float(self._answer_key(d.value) == y))
            conf.append(d.conf)
            act.append(None if a is None else self.model.act_probability(self.spec, d, a))
        self._source = one_source(ds)                 # mixed log-probabilities and written numbers: refused
        has_act = all(a is not None for a in act)
        if signal == "act" and not has_act:
            raise ValueError("the model gives no act signal: use signal='confidence'")
        if signal == "act" and self.use_act is False:
            raise ValueError("this part was made with use_act=False: a threshold on the act signal would never escalate "
                             "and the recorded guarantee would not hold — use signal='confidence', or make the part with "
                             "use_act=True")
        use_act = signal == "act" or (signal == "auto" and has_act and self.use_act is not False)
        if not use_act and has_act and self.use_act is not False:
            # calibrating on the confidence of a model with an act head: the checkpoint's act threshold goes on
            # escalating (the part's own is cleared), so an example it escalates is never answered alone — signal −1,
            # below every confidence. The calibration's numbers are then the part's.
            gate = self.model.act_threshold
            conf = [c if a >= gate else -1.0 for c, a in zip(conf, act)]
        return (act if use_act else conf), ok, ("act" if use_act else "confidence"), ds

    def _sourced(self, g):
        """A guarantee record with the source of the calibration's probabilities when the model names one (an LLM:
        "logprobs" or "stated"): a decision whose probabilities come from the other source escalates."""
        src = getattr(self, "_source", None)
        return g if src is None else {**g, "probabilities": src}

    def _set_threshold(self, sig, thr, guarantee, groups=None):
        """Set the calibrated threshold on its signal and clear the other signal's (an earlier calibration's threshold on
        the other signal would still escalate, so the new calibration's numbers would not describe the part); what was
        cleared is recorded in the guarantee as "cleared"."""
        other = "escalate_below" if sig == "act" else "act_threshold"
        old = getattr(self, other)
        setattr(self, other, None)
        if sig == "act":
            self.act_threshold = thr
        else:
            self.escalate_below = thr
        if old is not None and guarantee is not None:   # no guarantee (a calibration file with none): nothing to note
            guarantee = {**guarantee, "cleared": {other: old}}
        self.guarantee = guarantee
        self.groups = groups
        extra = [n for n in (groups["by"].names if groups else []) if n not in self.facts]
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts + extra])

    @_deprecate.removed_kwargs(error="max_error")
    def calibrate_for(self, examples, *, max_error=0.05, signal="auto", method="empirical", delta=0.10):
        """Choose the escalation threshold for a target error rate among the answers given alone, on labelled examples
        [(input, correct)]. method="empirical": the lowest threshold at which the calibration decisions it lets through
        are wrong at most `error` of the time — no guarantee on new inputs (on another data set the error can be several
        times the target); method="ltt" (learn-then-test): the error among the answered is ≤ `error` with probability
        ≥ 1 − delta for inputs like the examples — a strong promise, so it often lets nothing through (it tests at most 64
        thresholds, quantiles of the calibration signals: calibration.ltt_grid). signal: "act"
        (the model's act probability → act_threshold), "confidence" (the calibrated confidence → escalate_below) or
        "auto" (act when the model has an act head). On "confidence" with a model that has an act head (and a part not
        made with use_act=False) the checkpoint's act threshold keeps escalating: the examples it escalates count as
        escalated here, so the numbers returned are what the part does. No threshold reaches the target → everything
        escalates (inf).
        Changes the part's fingerprint. → {"signal", "threshold", "answered" (the share answered alone on the
        examples), "error" (among them), "n", "max_error", "method", "guarantee"}. For a guarantee on the share of all
        questions answered wrongly, see act_guard. Every option after the examples is keyword-only (error= and the
        result's keys "coverage" / "target_error", the 0.7 names, were removed in 0.9)."""
        from ..calibration import accuracy_at, check_rate, ltt_threshold
        error = max_error
        examples = list(examples)
        if method not in ("empirical", "ltt"):
            raise ValueError('method must be "empirical" or "ltt"')
        check_rate("error", error, zero=method == "empirical")     # ltt cannot certify 0 (math domain error before)
        if method == "ltt":
            check_rate("delta", delta)
        sig, ok, name, _ = self._labelled(examples, signal)
        if method == "ltt":
            thr = ltt_threshold(sig, [1 - o for o in ok], error, delta)
            g = {"method": "ltt", "error": error, "delta": delta, "n": len(ok), "signal": name,
                 "promise": f"error among the answers given alone ≤ {error:g} with probability ≥ {1 - delta:g}, "
                            "for inputs like the calibration examples"}
        else:
            thr = _threshold(sig, ok, error)
            g = {"method": "empirical", "error": error, "n": len(ok), "signal": name,
                 "promise": "none: the error was measured on the calibration examples only"}
        thr = max(thr, 0.0) if name == "confidence" else thr      # −1 marks an example the act gate escalates
        self._set_threshold(name, thr, self._sourced(g))
        acc, cov = accuracy_at(sig, ok, thr)
        return {"signal": name, "threshold": thr, "answered": cov, "error": (1 - acc) if cov else 0.0,
                "n": len(ok), "max_error": error, "method": method, "guarantee": g["promise"]}

    @_deprecate.removed_kwargs(risk="max_risk")
    def act_guard(self, examples, *, max_risk=0.10, signal="auto", groups=None, min_group=100, delta=0.10):
        """Answer alone only as far as a guarantee allows (conformal risk control), from labelled examples of your own
        stream [(input, correct)] — a few hundred is typical: the escalation threshold is set so that, for inputs like
        the examples, P(answered alone AND wrong) ≤ risk — a share of all questions (answered or escalated), not of
        the answered ones. `correct`: an option (a value), Unknown for "not stated", for a span question the passage's
        text (or a Quote: its text is compared), for a ranking the order. It holds for your stream, not under a shift of domain: recalibrate when the inputs change.
        Too few or too hard examples → everything escalates (threshold inf). Feasibility: when the model is wrong on
        a share μ > risk of the examples, any rule must escalate at least (μ − risk) / (1 − risk) of the inputs
        ("must_escalate_at_least"; arXiv 2606.29054) — a better signal can only get closer to that bound. Changes the
        part's fingerprint; the trace of every decision records the promise. The error among the answers given alone is
        not bounded (calibrate_for(method="ltt") bounds it): with few answered it can be far above `risk`. A signal that
        does not separate right from wrong answers (solvi.core.calibration.separation: AUROC not above chance at the 5%
        level) keeps the promise only by escalating, and is warned about (UserWarning, "warnings"). → {"signal",
        "threshold", "answered" (share answered alone on the examples), "error" (among them), "risk" (answered and
        wrong, on the examples), "n", "guarantee", "promise" (in words, with that error), "base_error",
        "must_escalate_at_least", "warnings" when there are any}.

        groups: a threshold per group — a fact name ("domain"), a hierarchy of fact names (["domain", "task"]) or a
        function of facts returning a group or a path (see GroupBy); the examples then give those facts (Facts(...) or
        a state with those keys). The promise over the whole stream allows a hard group to be answered wrongly far more
        often than `risk`; per group it holds inside each: every group with at least `min_group` examples gets its own
        threshold, a smaller one is pooled with the rest of its parent (whose threshold is calibrated on exactly those
        examples), the rest of the stream takes what is left. delta=0.10: with probability ≥ 90% over the examples,
        P(answered alone and wrong | group) ≤ risk in every group at once (a binomial bound per group at delta divided
        by the number of groups — Bonferroni; after HG-CRC, arXiv 2607.24562); delta=None: conformal risk control per
        group (each group on average). Every decision records its group and the group whose threshold applied; an
        input that does not give its group escalates. The group facts join the part's inputs: register the part in a
        catalog after act_guard. Adds "groups" ({path: {"threshold", "n", "answered", "error", "risk", "pooled"}}) to
        the result; "threshold" is then the rest of the stream's — inf when every example is in a group with its own
        threshold, whatever "answered" says: read the thresholds per group.

        The decider protocol: a combination (solvi.core.deciders.combine) takes the same act_guard(examples, *, max_risk, signal,
        groups, min_group, delta) and returns the same keys. Every option after the examples is keyword-only; risk= is
        the 0.7 name of max_risk= (removed in 0.9)."""
        from ..calibration import check_rate, crc_threshold
        risk = max_risk
        check_rate("max_risk", risk)                      # risk=10 (a percent) or 1.5 would be recorded as a promise
        if groups is not None and delta is not None:
            check_rate("delta", delta)
        examples = list(examples)
        sig, ok, name, _ = self._labelled(examples, signal)
        s, o = np.asarray(sig, float), np.asarray(ok, float)
        base = float(1 - o.mean())
        if groups is None:
            thr = crc_threshold(sig, [1 - x for x in ok], risk)
            thr = max(thr, 0.0) if name == "confidence" else thr  # −1 marks an example the act gate escalates
            g = {"method": "crc", "risk": risk, "n": len(ok), "signal": name,
                 "promise": f"P(answered alone and wrong) ≤ {risk:g} for inputs like the calibration examples"}
            self._set_threshold(name, thr, self._sourced(g))
            auto = s >= thr
            out = {}
        else:
            by = GroupBy(groups)
            paths = [by.path(x if isinstance(x, Facts) else None, x) for x, _ in examples]
            if any(p is None for p in paths):
                i = next(i for i, p in enumerate(paths) if p is None)
                raise ValueError(f"example {i}: its group ({by.names}) is not given; give the examples as Facts(...) or "
                                 "states with those keys")
            nodes, info, auto = _group_guard(s, 1 - o, paths, risk, min_group, delta)
            g = {"method": "crc-groups" if delta is None else "group-bound", "risk": risk, "n": len(ok), "signal": name,
                 "groups": by.label(), "min_group": min_group, "delta": delta,
                 "promise": _group_promise(risk, delta, len(nodes))}
            self._set_threshold(name, nodes[()]["threshold"], self._sourced(g), {"by": by, "nodes": nodes, "signal": name})
            thr = nodes[()]["threshold"]
            out = {"groups": info}
        err = float(1 - o[auto].mean()) if auto.any() else 0.0
        out = {"signal": name, "threshold": thr, "answered": float(auto.mean()), "error": err,
               "risk": float(((1 - o) * auto).mean()), "n": len(ok), "guarantee": g["promise"],
               "promise": guard_promise(risk, err, bool(auto.any())),
               "base_error": base, "must_escalate_at_least": max(0.0, (base - risk) / (1 - risk)), **out}
        warn = no_separation(sig, ok, name, err, base) if auto.any() else None
        if warn:
            out["warnings"] = [warn]
            warnings.warn(warn, UserWarning, stacklevel=2)
        return out

    def conformal(self, examples, coverage=0.90):
        """Conformal answer sets from labelled examples [(input, correct)]: afterwards every decision carries
        `extra["candidates"]` — the answers that cannot be ruled out, which contain the right one with probability
        ≥ coverage for inputs like the examples (score and number questions: one contiguous interval) — and an
        escalation's message lists them for the person who takes over. It does not change what is answered alone
        (see act_guard). Choice, yes/no, score and number questions. → {"coverage", "quantile", "n", "mean_size"}."""
        from ..calibration import check_rate, conformal_quantile, set_scores
        check_rate("coverage", coverage)
        if self.spec.multi or self.kind in ("rank", "span"):
            raise ValueError(f"conformal sets need a single answer from a closed list; not for {self.kind!r} questions")
        examples = list(examples)                    # an iterator (zip, a generator) is read twice below
        _, _, _, ds = self._labelled(examples, "confidence")
        ordinal = self.kind in ("score", "number")
        scores, sizes = [], []
        for d, (_, y) in zip(ds, examples):
            keys = list(d.probs)
            g = Unknown if y is Unknown else self.spec.label(y)
            if g not in keys:
                raise ValueError(f"{y!r}: this question cannot answer it (its answers: {keys})")
            scores.append(float(set_scores([d.probs[k] for k in keys], ordinal, Unknown in keys)[keys.index(g)]))
        q = conformal_quantile(scores, 1 - coverage)
        self.conformal_set = {"coverage": coverage, "quantile": q, "n": len(scores), "ordinal": ordinal}
        for d in ds:
            sizes.append(len(self.candidates(d)))
        return {"coverage": coverage, "quantile": q, "n": len(scores), "mean_size": float(np.mean(sizes))}

    # --- the calibration file's state of this part (solvi.core.calibfile reads and writes only the file format)
    _calibration_kind = "DecisionPart"

    def _calibration_base(self):
        """What a calibration is fitted to: the fingerprint without the thresholds (the checkpoint, the question, this
        question's adaptation, and how the part computes its signal — option_order="average" with its permutations,
        long= with top_k and rerank, the adapter)."""
        a = self.adaptation
        signal = {k: v for k, v in (("option_order", None if self.option_order != "average" else
                                     (self.option_order, self.permutations)),
                                    ("long", self.long_key()),
                                    ("lora", None if self.lora is None else self.lora.hash))
                  if v is not None}
        if signal:                                  # a part with the default signal keeps the fingerprint it had
            return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None,
                          signal)
        return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None)

    def _calibration_models(self):
        """{model id: weights fingerprint} of the model behind the part (for the message when they differ)."""
        return {str(self.model_id): self.model.weights_fingerprint()}

    def _calibration_thresholds(self):
        return {"escalate_below": self.escalate_below, "act_threshold": self.act_threshold}

    def _calibration_adapter(self):
        """The adapter the calibration was made with (written beside the file), or None."""
        return self.lora

    def _apply_calibration(self, rec, grp, path):
        """Set the thresholds a calibration file holds (solvi.core.calibfile.load has checked it)."""
        if grp is not None:
            grp["signal"] = rec["groups"].get("signal")
        self._set_threshold("act", rec.get("act_threshold"), rec.get("guarantee"), grp)   # sets groups, inputs
        self.escalate_below, self.act_threshold = rec.get("escalate_below"), rec.get("act_threshold")   # both, as saved
        self.guarantee = rec.get("guarantee")
        self.conformal_set = rec.get("conformal")

    def save_calibration(self, path):
        """Write this decision's calibration — the escalation thresholds (per group too), the guarantee record, the
        conformal set — with the question and the fingerprint of the model and adaptation it was fitted on, to a JSON
        file (solvi.core.calibfile; `solvi calibrate` writes the same). → path."""
        from ..calibfile import save
        return save(self, path)

    def load_calibration(self, path, groups=None, strict=True):
        """Apply a calibration file written by save_calibration / `solvi calibrate`: afterwards the part escalates, and
        records its guarantee, exactly as right after calibrating (the same fingerprint). Refuses (ValueError) a file made
        for another question, another checkpoint or another adaptation of this question — strict=False loads it anyway.
        Thresholds per group by a function: pass it again as groups=. Call it before registering the part in a catalog
        when the calibration has groups (the group facts join the part's inputs). While `solvi calibrate` loads a
        catalog, calibration files are not applied (the part is calibrated afresh). → self."""
        from ..calibfile import load
        return load(self, path, groups, strict)

    memory = gone_in_1_0("memory()", "solvi.core.knowledge.memory.attach(part, ...) — the memory of corrections moves into the "
                         "knowledge memory", "DecisionPart")

    @_deprecate.removed_kwargs(checkpoints="requires")
    def question(self, cat, name=None, text=None, min_confidence=None, requires=None, require_evidence=False):
        """Make this decision the answer of a question: registers it as the question's rule (`cat.rule(name)(self)`) and
        returns the Question — choice, multi, ordinal (score) or yes_no (noul), with the option descriptions. System.teach on
        that question teaches this decision."""
        name = name or self.__name__
        cat.rule(name)(self)
        sp = self.spec
        order = self._shown_labels if self.option_order == "canonical" else sp.options
        opts = {o: sp.descriptions.get(o, "") for o in order} if sp.descriptions else list(order)
        if sp.kind == "noul":
            at = Answer.yes_no()
            at.descriptions = dict(sp.descriptions)
        elif sp.kind == "span":
            at = Answer.span(source=self.facts[0], type=sp.vtype)
        elif sp.kind == "rank":
            at = Answer.rank(opts, sp.k)
        elif sp.kind == "number":
            at = Answer.estimate(sp.edges, coverage=sp.coverage, unit=sp.unit, integer=sp.integer)
        else:
            at = {"multi": Answer.multi, "score": Answer.ordinal}.get(sp.kind, Answer.choice)(opts)
        if sp.unknown:
            at = Answer.maybe(at)
        return Question(name, text or sp.task, at, requires=list(requires or []), min_confidence=min_confidence,
                        require_evidence=require_evidence)


from ..runtime import plan_batches   # noqa: E402,F401 — re-exported (defined there since 1.0)

__all__ = ["DecisionPart", "plan_batches"]
