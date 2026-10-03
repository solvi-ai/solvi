"""Several models, one decision: a cascade, a vote and a route over decision parts (solvi.core.deciders). The models propose;
deterministic code over their proposals decides; every proposal is in the trace.

    from solvi.core.deciders.combine import Cascade, Route, Vote
    small = base.decision("team", "Which team?", "email", TEAMS)
    large = big.decision("team", "Which team?", "email", TEAMS)

    team = Cascade([small, large])            # ask the small model; the large one only when the small one escalates
    team = Vote([small, large], rule="all")    # answer when they agree and each is sure enough; else escalate
    team = Route({lambda email: len(email) > 2000: large}, default=small)     # code picks the model per input

    cat.fn(team)                               # like any decision part: a fact, or `team.question(cat)` for an answer
    team.act_guard(examples, max_risk=0.10)        # P(answered alone and wrong) ≤ 10%, for the combination as a whole

A combination is a catalog part like a DecisionPart: its value is one of the options, its provenance `decided`, its
fingerprint covers every model's, and the trace records every proposal: `extra["stages"]` and `extra["answered_by"]`
(cascade), `extra["votes"]` (vote), `extra["route"]` and `extra["routed"]` (route), and `extra["calls"]`, the models
called for this decision. The parts must answer the same question (kind and options; the task and the facts they read
may differ). Combinations nest: `Cascade([small, Vote([mid, large])])`.

Thresholds. Uncalibrated, each part escalates by its own thresholds (act_threshold, escalate_below, min_margin). After
`act_guard`, one threshold t applies to every part's signal (its act probability when its model gives one, else its
calibrated confidence) — a one-dimensional family (scale="raw", the default); scale="rank" puts t on each part's rank
among its own calibration signals instead, for models whose signals live on different scales (an LLM's confidence
near 1 and an act probability spread over [0, 1]) — it can help or hurt depending on the data, so compare both. A
cascade's loss is not monotone in t (a higher t can pass a question from a wrong small model to a right large one, or
back), so conformal risk control runs on the loss monotonized from above — the maximum over thresholds ≥ t — which
keeps the guarantee. A cascade saves cost where the small model is often sure; a vote lowers the error among the
automatic answers at the price of answering less."""
from __future__ import annotations

import dataclasses
import inspect
import math
import warnings
from collections.abc import Mapping

import numpy as np

from ... import _deprecate
from ..catalog import Decision, Quote, Unknown, gone_in_1_0
from . import DecisionPart, Facts, GroupBy, _group_info, _group_promise, _single, group_record, guard_promise, no_separation, one_source
from ..provenance import ESCALATED, code_fingerprint, digest
from ..runtime import RECORD_KEYS          # what a replay compares with the recomputed (defined there; re-exported)

# guarantee["signal"]: a name, as a part's ("act", "confidence"): one threshold shared by every part, on each part's
# own signal ("shared") or on its rank among that part's calibration signals ("shared-rank")
_SIGNAL = {"rank": "shared-rank", "raw": "shared"}
_SIGNAL_0_7 = {"shared-rank": "shared threshold on each model's rank among the calibration examples",
               "shared": "shared threshold on each model's signal"}
MAX_RANKS = 1024        # calibration signals kept per model for scale="rank" (more examples: this many evenly spaced ones)
SCALES = ("rank", "raw")
STAGE_FLOOR = 0.05      # act_guard on a cascade warns when a stage answers alone on less than this share


def _table(sigs):
    """A model's calibration signals → the sorted values its rank is taken among (finite ones; at most MAX_RANKS, taken
    evenly by order so the rank stays a monotone map of the signal)."""
    s = np.sort(np.asarray([x for x in sigs if np.isfinite(x)], float))
    if len(s) > MAX_RANKS:
        s = s[np.round(np.linspace(0, len(s) - 1, MAX_RANKS)).astype(int)]
    return s


def _rank(tbl, s):
    """A signal's rank among a model's calibration signals: the share of them ≤ s (0 below all, 1 at or above the
    largest; a model no calibration example reached ranks 0 — it never answers alone)."""
    if not len(tbl):
        return 0.0 if not np.isnan(s) else s
    if np.isnan(s):
        return s
    return int(np.searchsorted(tbl, s, "right")) / len(tbl)


def _raw_t(tbl, t):
    """The raw signal threshold equivalent to a rank threshold t for one model: rank(s) ≥ t ⟺ s ≥ this."""
    if t is None or tbl is None or t == -math.inf:
        return t
    n = len(tbl)
    if not n:
        return -math.inf if t <= 0 else math.inf
    if not t <= 1:                                   # also NaN
        return math.inf
    k = max(0, min(n, math.ceil(t * n)))
    while k > 0 and (k - 1) / n >= t:
        k -= 1
    while k <= n and k / n < t:
        k += 1
    if k > n:
        return math.inf
    return -math.inf if k == 0 else float(tbl[k - 1])


@dataclasses.dataclass
class _Src:
    vals: dict | None = None        # facts by name (a catalog run, Facts)
    raw: object = None              # one input every part reads


def _src(x):
    return _Src(vals=dict(x)) if isinstance(x, Facts) else _Src(raw=x)


def _copy(d):
    return dataclasses.replace(d, probs=dict(d.probs), extra=dict(d.extra), evidence=list(d.evidence))


def _jv(v):
    """A decision's value as JSON data for the trace."""
    from enum import Enum
    if isinstance(v, Quote):
        return {"quote": [v.value, v.start, v.end, v.source]}
    if v is Unknown:
        return {"not_stated": True}
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, (list, tuple)):
        return [_jv(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _key(v):
    from ..runtime import vhash
    return vhash(_jv(v))


def _shown(jv):
    if isinstance(jv, dict) and "quote" in jv:
        return repr(jv["quote"][0])
    if isinstance(jv, dict) and jv.get("not_stated"):
        return "not stated"
    return repr(tuple(jv) if isinstance(jv, list) else jv)


def _who(e):
    return f"{e['part']} ({e['model']})" if e.get("model") else e["part"]


def _probs(p):
    return {str(k): round(float(v), 6) for k, v in p.items()}


def _question(sp):
    """What must be equal for parts to answer the same question."""
    return {"kind": sp.kind, "options": [str(o) for o in sp.options], "as_bool": sp.as_bool, "unknown": sp.unknown,
            "k": sp.k, "bins": sp.edges}


# ------------------------------------------------------------------------------------------------ members
class _LeafState:
    def __init__(self, leaf, d0, act, hard, own, src, ctx=None):
        self.leaf, self.d0, self.act, self.hard, self.own, self.src, self.ctx = leaf, d0, act, hard, own, src, ctx
        self.signal, self.sig = leaf.part._signal(hard)
        self.key = _key(hard.value)

    def calls(self):
        return 1

    def walk(self):
        yield self

    def force(self):
        return self

    def sigs(self):
        return [self.sig]


class _Leaf:
    """A DecisionPart as a member of a combination."""

    kind = "part"

    def __init__(self, part):
        self.part = part
        self.name = part.__name__
        self.cost = 1.0
        self.cost_given = False                   # costs= named it: act_guard and usage report the expected cost
        self.calls = 0

    @property
    def facts(self):
        return list(self.part.facts)

    @property
    def spec(self):
        return self.part.spec

    @property
    def model_id(self):
        return self.part.model_id

    def leaves(self):
        return [self]

    def same_question(self):
        return _question(self.part.spec)

    def fingerprint(self):
        return self.part.fingerprint()

    def text(self, src):
        p = self.part
        return p.text_of(src.vals) if src.vals is not None else p.model.text(src.raw)

    def state(self, src, pre=None):
        p = self.part
        text = self.text(src)
        ctx = dict(p._ctx(text, src.vals, src.raw), combined=True)    # a memory of corrections only checks here
        if pre is not None:
            d0, a = p.model._decision(p.spec, pre[0]), pre[1]
        else:
            d0, a = p._initial(text)          # as the part alone reads it (a long text: its retrieved window)
        hard = p._finish(_copy(d0), a, threshold=-math.inf, ctx=ctx)     # every safeguard but the threshold
        own = p._finish(_copy(d0), a, ctx=ctx)                            # the part's own thresholds
        if src.vals is not None:
            hard, own = p._bind(hard, src.vals), p._bind(own, src.vals)
        return _LeafState(self, d0, a, hard, own, src, ctx)

    def vec(self, st, ts, rk=None):
        """At each threshold of ts (NaN: the part's own thresholds) → (answers alone [G], value key [G], signal [G],
        {key: value}, cost [G], calls [G]). rk: {id(leaf): sorted calibration signals} — the threshold is on the rank
        of the signal among them (scale="rank"); empty or None: on the signal itself."""
        own = np.isnan(ts)
        tbl = rk.get(id(self)) if rk else None
        sig = st.sig if tbl is None else _rank(tbl, st.sig)
        with np.errstate(invalid="ignore"):
            auto = np.where(own, st.own.escalate is None, (st.hard.escalate is None) & (sig >= ts))
        G = len(ts)
        return (auto, np.full(G, st.key, dtype=object), np.full(G, sig), {st.key: st.hard.value},
                np.full(G, self.cost), np.ones(G))

    def final(self, st, t, rk=None):
        """The part's decision at threshold t (None: its own thresholds; on the rank scale with rk, as in vec)."""
        if t is None:
            return _copy(st.own)
        p = self.part
        tbl = rk.get(id(self)) if rk else None
        if tbl is not None:
            t = _raw_t(tbl, t)
        d = p._finish(_copy(st.d0), st.act, threshold=t, ctx=st.ctx)
        return p._bind(d, st.src.vals) if st.src.vals is not None else d

    def entry(self, d, st):
        name, s = self.part._signal(d)
        e = {"part": self.name, "model": self.model_id, "value": _jv(d.value),
             "confidence": round(float(d.conf), 6), "signal": name, "score": round(s, 6), "escalate": d.escalate,
             "probs": _probs(d.probs)}
        if "memory" in d.extra:                       # a memory of corrections consulted by this part (solvi.core.knowledge.memory)
            e["memory"] = d.extra["memory"]
        return e

    def check(self, e):
        opts = self.part.options
        if opts is None or e.get("value") is None:
            return []
        vs = e["value"] if isinstance(e["value"], list) else [e["value"]]
        known = {_key(o) for o in opts}
        return [] if all(_key(v) in known for v in vs) else [f"{self.name} proposed {e['value']!r}, not one of its options"]


class _State:
    """A combination's members' states, computed when first needed (a cascade asks the next model only if it must)."""

    def __init__(self, members, src, pre=None, pick=None):
        self.members, self.src, self.pre, self.pick = members, src, pre or {}, pick
        self.done = {}

    def get(self, i):
        if i not in self.done:
            m = self.members[i]
            self.done[i] = m.state(self.src, self.pre.get(i)) if isinstance(m, _Leaf) else m.state(self.src)
        return self.done[i]

    def calls(self):
        return sum(s.calls() for s in self.done.values())

    def walk(self):
        for s in self.done.values():
            yield from s.walk()

    def force(self):
        for i in (range(len(self.members)) if self.pick is None else [self.pick]):
            self.get(i).force()
        return self

    def sigs(self):
        return [x for s in self.done.values() for x in s.sigs()]


def _wrap(m):
    if isinstance(m, DecisionPart):
        return _Leaf(m)
    if isinstance(m, Combination):
        return m
    raise TypeError(f"a combination is made of decision parts (model.decision(...)) or other combinations, not {m!r}")


# ------------------------------------------------------------------------------------------------ combinations
class Combination:
    """What Cascade, Vote and Route share: a catalog function over the union of the parts' facts that returns a
    Decision; the model recorded in the trace (fingerprint over every part's).

    The decider protocol: a combination has every public method of a DecisionPart, with the same signature and result
    keys. decide / score / act_guard / calibrate_for / conformal / save_calibration / load_calibration act on the
    combination as a whole (one threshold shared by every part); fit / adapt / teach / reset go
    to every part (a list per part, in leaves order, where the part returns one value); calls() counts the models
    called. What belongs to one part — save_lora, load_lora (an adapter is one checkpoint's, for one
    question), budget, sections_k, long_key, long_input (each part reads long texts by its own long=), in_pass (a
    shared forward pass is for parts of one model) — raises NotImplementedError naming the part to call it on."""

    kind_name = "combination"

    def __init__(self, members, name=None, costs=None):
        self.members = [_wrap(m) for m in members]
        if not self.members:
            raise ValueError(f"a {self.kind_name} needs at least one decision part")
        q0 = self.members[0].same_question()
        for m in self.members[1:]:
            q = m.same_question()
            if q != q0:
                diff = {k: (q0[k], q[k]) for k in q0 if q0[k] != q[k]}
                raise ValueError(f"the parts of a {self.kind_name} must answer the same question; {m.name} differs from "
                                 f"{self.members[0].name} in {diff}")
        if costs is not None:
            if len(costs) != len(self.members):
                raise ValueError(f"costs: one per part ({len(self.members)}), not {len(costs)}")
            nested = [m.name for m in self.members if not isinstance(m, _Leaf)]
            if nested:                            # its cost depends on which of its parts it asks: give theirs to it
                raise ValueError(f"costs: {nested} is itself a combination — give the costs of its parts to it "
                                 "(Vote([...], costs=[...])), and none here (costs=None)")
            for m, c in zip(self.members, costs):
                m.cost, m.cost_given = float(c), True
        self.costs = None if costs is None else [float(c) for c in costs]
        self.name = name or self.members[0].name
        self.threshold = None                   # the shared threshold (act_guard); None: each part's own
        self.scale = None                       # what the shared threshold is on: "rank" or "raw" (act_guard)
        self.ranks = None                       # scale="rank": each leaf's sorted calibration signals (leaves order)
        self.groups = None                      # thresholds per group (act_guard(groups=...)): {"by", "nodes"}
        self.guarantee = None
        self.conformal_set = None
        self.asked = 0
        self._setup()

    def _setup(self):
        facts = list(dict.fromkeys(f for m in self.members for f in m.facts))
        self.facts = list(dict.fromkeys(facts + self._extra_facts()))   # two predicates may read the same fact
        self.__name__ = self.__qualname__ = self.name
        self.__doc__ = self.spec.task
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts])
        self.__solvi_model__ = self
        self.__solvi_provenance__ = "decided"
        self.__solvi_options__ = None if self.spec.values is None else list(self._first.options)
        self.__solvi_decision__ = self                # System.teach teaches every part

    def _extra_facts(self):
        return [] if self.groups is None else list(self.groups["by"].names)

    # --- the question and identity
    @_deprecate.removed_kwargs(checkpoints="requires")
    def question(self, cat, name=None, text=None, min_confidence=None, requires=None, require_evidence=False):
        """Make this combination a question's answer (as DecisionPart.question) → the Question."""
        return DecisionPart.question(self, cat, name, text, min_confidence, requires, require_evidence)

    def same_question(self):
        """What the parts must agree on (kind, options, "not stated", rank k, number bins)."""
        return self.members[0].same_question()

    @property
    def spec(self):
        return self.members[0].spec

    @property
    def _first(self):
        """The first DecisionPart at the bottom of the combination: its option order is how the options are shown."""
        m = self.members[0]
        return m.part if isinstance(m, _Leaf) else m._first

    @property
    def option_order(self):
        return self._first.option_order

    @property
    def _shown_labels(self):
        return self._first._shown_labels

    @property
    def options(self):
        return self._first.options

    @property
    def kind(self):
        return self.spec.kind

    @property
    def labels(self):
        """The options as the models read them (the first part's; every part answers the same question)."""
        return self._first.labels

    @property
    def multi(self):
        return self.spec.multi

    @property
    def task(self):
        return self.spec.task

    @property
    def adaptation(self):
        """Every part's adaptation (adapt / fit / teach), in leaves order."""
        return [lf.part.adaptation for lf in self.leaves()]

    @property
    def lora(self):
        """Every part's LoRA adapter (or None), in leaves order."""
        return [lf.part.lora for lf in self.leaves()]

    @property
    def available(self):
        return all(getattr(lf.part, "available", True) for lf in self.leaves())

    @property
    def deterministic(self):
        return all(getattr(lf.part.model, "deterministic", True) for lf in self.leaves())

    @property
    def parts(self):
        """The members: DecisionParts and nested combinations."""
        return [m.part if isinstance(m, _Leaf) else m for m in self.members]

    def leaves(self):
        return [lf for m in self.members for lf in m.leaves()]

    @property
    def model_id(self):
        return f"{self.kind_name}({self._sep().join(m.model_id for m in self.members)})"

    def _sep(self):
        return ", "

    def _describe(self):
        return {}

    def fingerprint(self):
        """A hash of the combination: its kind and rule, every member's fingerprint, and its calibration (threshold,
        guarantee, conformal set)."""
        g = self.guarantee
        if g is not None and g.get("signal") in _SIGNAL_0_7:      # hashed as 0.7 wrote it: the fingerprints, and the
            g = {**g, "signal": _SIGNAL_0_7[g["signal"]] + (", per group" if "groups" in g else "")}  # stored decisions
        th = {k: v for k, v in (("threshold", self.threshold), ("guarantee", g),          # that carry them, stay valid
                                ("conformal", self.conformal_set)) if v is not None}
        if self.scale == "rank":                      # the raw scale keeps the fingerprint it always had
            th["scale"] = ("rank", [np.asarray(r, float) for r in self.ranks or []])
        if self.groups is not None:
            th["groups"] = (self.groups["by"].describe(),
                            sorted((list(k), v["threshold"]) for k, v in self.groups["nodes"].items()))
        return digest(type(self).__name__, self._describe(), [m.fingerprint() for m in self.members], th)

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {[m.name for m in self.members]}, model={self.model_id!r})"

    # --- deciding
    def text_of(self, facts):
        """The facts this combination reads, from a dict of facts (System.teach passes them back to `teach`)."""
        return Facts({f: facts[f] for f in self.facts if f in facts})

    def __call__(self, *args, **kw):
        vals = dict(zip(self.facts, args))
        vals.update(kw)
        return self._decide(_Src(vals=vals))

    @_deprecate.removed_kwargs(x="text")
    def decide(self, text):
        """An input (a text, a state, or Facts by name) → Decision; a list of inputs → a list."""
        one = isinstance(text, Facts) or _single(text)
        out = [self._decide(_src(v)) for v in ([text] if one else list(text))]
        return out[0] if one else out

    def score(self, text):
        """The probabilities the combination answers with (see decide) → {option: probability}; a list → a list."""
        d = self.decide(text)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def _decide(self, src):
        st = self.state(src)
        d = self.final(st, None)
        self.asked += 1
        for ls in st.walk():                          # usage: the models this decision called (not calibration's)
            ls.leaf.calls += 1
        return d

    def _group(self, src):
        """With thresholds per group: (the input's group path, the node whose threshold applies, its info), or None
        when the input does not give its group."""
        from ..calibration import node_of
        path = self.groups["by"].path(src.vals, src.raw)
        if path is None:
            return None
        node = node_of(path, self.groups["nodes"])
        return path, node, self.groups["nodes"][node]

    def _own(self, src):
        """The combination's own threshold for this input: the shared one, its group's, inf when its group is unknown."""
        if self.groups is None or src is None:
            return self.threshold
        g = self._group(src)
        return math.inf if g is None else g[2]["threshold"]

    def _tables(self):
        """{id(leaf): sorted calibration signals} when the shared threshold is on the rank scale, else {} (raw)."""
        if self.scale != "rank" or self.ranks is None:
            return {}
        return {id(lf): r for lf, r in zip(self.leaves(), self.ranks)}

    def _t(self, t, src=None, rk=None):
        """The threshold and its scale that apply: an outer one as given, else this combination's own."""
        if t is None:
            return self._own(src), self._tables()
        return t, (self._tables() if rk is None else rk)

    def _vt(self, ts, src=None, rk=None):
        own = self._own(src)
        if own is not None and np.isnan(ts).all():
            rk = None                                 # this combination's own threshold: its own scale
        return (ts if own is None else np.where(np.isnan(ts), own, ts)), (self._tables() if rk is None else rk)

    def _wrapup(self, d, t, src=None):
        """The combination's own threshold record, guarantee and conformal candidates (a nested combination under an
        outer threshold records none of its own)."""
        if t is None and self.threshold is not None:
            g = None if self.groups is None else self._group(src)
            d.extra["threshold"] = self._own(src)
            if self.guarantee is not None:
                d.extra["guarantee"] = dict(self.guarantee) if g is None else group_record(self.guarantee, *g)
            if self.groups is not None and g is None and d.escalate is not None:
                d.escalate += (f"; group unknown: the thresholds are per group ({self.groups['by'].label()}) and this "
                               f"input does not give {self.groups['by'].names}")
        if self.conformal_set is not None and d.probs:
            cands = self.candidates(d)
            d.extra["candidates"] = cands
            if d.escalate:
                d.escalate += f"; candidates at {self.conformal_set['coverage']:.0%}: {cands!r}"
        return d

    def candidates(self, d):
        """The conformal answer set of a decision (after conformal(...)), most probable first."""
        return DecisionPart.candidates(self, d)

    def entry(self, d, st):
        e = {"part": self.name, "kind": self.kind_name, "value": _jv(d.value), "confidence": round(float(d.conf), 6),
             "escalate": d.escalate}
        e.update({k: d.extra[k] for k in RECORD_KEYS + ("rule", "calls") if k in d.extra})
        return e

    # --- calibration
    def _examples(self, examples):
        ex = list(examples)
        if not ex:
            raise ValueError("calibration needs labelled examples")
        return [_src(x) for x, _ in ex], [y for _, y in ex]

    def _one_source(self, states):
        """Every part's probabilities on the calibration examples come from one source (solvi.core.deciders.one_source: an LLM
        that answered some from log-probabilities and some from the numbers it wrote is refused)."""
        for lf in self.leaves():
            one_source([ls.d0 for st in states for ls in st.walk() if ls.leaf is lf],
                       f"the calibration examples of part {lf.name!r}")

    def _right(self, v, y):
        sp = self.spec
        if isinstance(v, Quote):
            v = v.value
        if isinstance(y, Quote):
            y = y.value
        if sp.kind in ("rank", "span"):
            return v == y
        if v is None:                                     # the part escalated with no value: not a right answer
            return False
        return (Unknown if v is Unknown else sp.label(v)) == (Unknown if y is Unknown else sp.label(y))

    @_deprecate.removed_kwargs(risk="max_risk")
    def act_guard(self, examples, *, max_risk=0.10, signal="auto", groups=None, min_group=100, delta=0.10, scale="raw"):
        """Answer alone only as far as a guarantee allows, for the combination as a whole: on labelled examples of your
        stream [(input, correct)] (an input is what every part reads, or Facts(...) by name) every part is asked, and
        one threshold t shared by every part is chosen by conformal risk control so that P(answered alone AND wrong)
        ≤ risk for inputs like the examples — a share of all questions. A cascade's loss is not monotone in t, so it is
        monotonized from above (the maximum over the thresholds ≥ t) before the choice, which keeps the guarantee.
        Replaces the parts' own thresholds inside this combination (the parts themselves are not changed); changes the
        combination's fingerprint and clears its conformal sets (call conformal afterwards). Too few or too hard
        examples → everything escalates (threshold inf). → {"threshold", "answered", "error" (among the answered),
        "risk" (answered and wrong, on the examples), "n", "guarantee", "calls_per_question" (models called per question), "cost"
        (with costs=), "scale", and for a cascade "answered_by" (the share each stage answered) and "warnings" when a
        stage answers alone on less than 5% of the examples (the cascade is then no better than one model) or a part's
        signal does not separate right from wrong answers (also a UserWarning), "promise" (in words: of all inputs, not
        of the answered ones — "error" is not bounded)}.

        scale: what the shared threshold is on. "raw" (default): the parts' signals themselves (the act probability
        when the model gives one, else the calibrated confidence). When the scales differ — an act probability spread
        over [0, 1], an LLM's confidence near 1 — one raw threshold effectively fits one model and the combination
        behaves like that model alone, which is often the stronger one. "rank" (opt-in): each part's signal is replaced
        by its rank among that part's own signals on the calibration examples (the share of them ≤ it), so every part
        can take part. It helps where a stage never answers on the raw scale and can lower the share answered alone
        elsewhere; the guarantee holds either way. Compare both on held-out calibration data. The rank uses the calibration inputs, not their labels (the guarantee then holds up to a term of order
        1/n). The sorted calibration signals of each part (at most MAX_RANKS = 1024, evenly spaced by order when there
        are more examples) are kept in the combination and in its calibration file.

        groups, min_group, delta: one shared threshold per group, as DecisionPart.act_guard(groups=...) — on the same
        monotonized loss, so the promise holds within every group; the group facts join the combination's inputs.
        Adds "groups" to the result.

        The decider protocol: the signature and the keys of DecisionPart.act_guard — "signal" ("shared", or
        "shared-rank" with scale="rank"; also guarantee["signal"]), "threshold", "answered", "error", "risk", "n",
        "guarantee", "promise", "base_error", "must_escalate_at_least", "warnings", "groups" — plus "calls_per_question"
        (models called per question; "calls" in 0.7, removed in 0.9), "cost", "scale", "answered_by". signal: "auto" only (each part
        brings its own; `scale` says how they share one threshold). Every option after the examples is keyword-only
        (risk=, the 0.7 name of max_risk=, was removed in 0.9)."""
        from ..calibration import certify_groups, check_rate
        risk = max_risk
        check_rate("max_risk", risk)
        if signal != "auto":
            raise ValueError(f"signal={signal!r}: a combination's signal is each part's own (signal=\"auto\"); "
                             "scale=\"raw\" or \"rank\" says how they share one threshold")
        if groups is not None and delta is not None:
            check_rate("delta", delta)
        srcs, gold, states, ranks, rk, grid, loss, auto_all, cost, calls, who = self._sweep(examples, scale)
        n, G = len(states), len(grid)
        mono = np.maximum.accumulate(loss[:, ::-1], axis=1)[:, ::-1]
        promise = f"P(answered alone and wrong) ≤ {risk:g} for inputs like the calibration examples"
        extra = {}
        if groups is None:
            r = (mono.sum(0) + 1) / (n + 1)
            good = np.where(r <= risk + 1e-12)[0]
            gi = np.full(n, int(good[0]) if len(good) else G - 1)
            t = float(grid[gi[0]]) if len(good) else math.inf
            self.groups = None
            self.guarantee = {"method": "crc", "risk": risk, "n": n, "signal": _SIGNAL[scale], "promise": promise}
        else:
            by = GroupBy(groups)
            paths = [by.path(s.vals, s.raw) for s in srcs]
            if any(p is None for p in paths):
                i = next(i for i, p in enumerate(paths) if p is None)
                raise ValueError(f"example {i}: its group ({by.names}) is not given; give the examples as Facts(...)")
            got, owner = certify_groups(lambda ix: (grid, mono[ix].sum(0)), paths, risk, min_group, delta)
            nodes = {k: {"threshold": v["threshold"], "n": v["n"]} for k, v in got.items()}
            # a node that certifies nothing answers nothing: its index is past the grid (every example escalates)
            gi = np.array([got[o]["index"] if got[o]["index"] is not None else G - 1 for o in owner])
            t = nodes[()]["threshold"]
            self.groups = {"by": by, "nodes": nodes}
            self.guarantee = {"method": "crc-groups" if delta is None else "group-bound", "risk": risk, "n": n,
                              "signal": _SIGNAL[scale], "groups": by.label(),
                              "min_group": min_group, "delta": delta, "promise": _group_promise(risk, delta, len(nodes))}
            a_ = auto_all[np.arange(n), gi] & np.array([got[o]["index"] is not None for o in owner])
            extra["groups"] = _group_info(nodes, owner, paths, a_, loss[np.arange(n), gi] > 0)
        self.threshold = t
        self.scale, self.ranks = scale, ranks
        self.conformal_set = None
        self._setup()
        rows = np.arange(n)
        a = auto_all[rows, gi]
        lo = loss[rows, gi]
        if groups is not None:
            live = np.array([got[o]["index"] is not None for o in owner])
            a, lo = a & live, lo * live
        elif not np.isfinite(t):
            a, lo = np.zeros(n, bool), np.zeros(n)
        err = float(lo[a].sum() / a.sum()) if a.any() else 0.0
        base = float(np.mean([not self._right(self.final(st, -math.inf, rk).value, y)       # every question answered
                              for st, y in zip(states, gold)])) if n else 0.0
        out = {"signal": _SIGNAL[scale], "threshold": t, "answered": float(a.mean()), "error": err,
               "risk": float(lo.mean()), "n": n, "guarantee": self.guarantee["promise"],
               "promise": guard_promise(risk, err, bool(a.any())), "base_error": base,
               "must_escalate_at_least": max(0.0, (base - risk) / (1 - risk)),
               "calls_per_question": float(calls[rows, gi].mean()), "scale": scale}
        flat = [] if not a.any() else [
            no_separation([ls.sig for st in states for ls in st.walk() if ls.leaf is lf],
                          [self._right(ls.hard.value, y) for st, y in zip(states, gold) for ls in st.walk()
                           if ls.leaf is lf], "part's", None, None, f"part {lf.name!r}: ") for lf in self.leaves()]
        flat = [w for w in flat if w]
        if any(lf.cost_given for lf in self.leaves()):
            out["cost"] = float(cost[rows, gi].mean())
        if who[0] is not None:
            w = np.array([x[g] for x, g in zip(who, gi)])
            w = np.where(a, w, -1)
            out["answered_by"] = [float((w == j).mean()) for j in range(len(self.members))]
            if a.any():
                warn = [f"stage {j + 1} ({m.name}, {m.model_id}) answers alone on {b:.1%} of the calibration questions "
                        f"(< {STAGE_FLOOR:.0%}): the cascade is then no better than a single model — compare it with "
                        "each model alone (act_guard on each part) on the same examples"
                        + ("; if the models' signals differ in scale (an LLM's confidence near 1), try "
                           "act_guard(..., scale=\"rank\")" if scale == "raw" else "")
                        for j, (m, b) in enumerate(zip(self.members, out["answered_by"])) if b < STAGE_FLOOR]
                if warn:
                    out["warnings"] = warn
        if flat:
            out["warnings"] = out.get("warnings", []) + flat
            for w in flat:
                warnings.warn(w, UserWarning, stacklevel=2)
        out.update(extra)
        return out

    def _sweep(self, examples, scale, grid_of=None):
        """Ask every part on labelled examples and evaluate the combination at every shared threshold of a grid →
        (inputs, labels, states, ranks, rank tables, grid [G] ending in inf, loss [n, G] (answered alone and wrong),
        answered alone [n, G], cost [n, G], calls [n, G], the answering stage per example or None). grid_of: the grid
        from the signals (default: every distinct one)."""
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}, not {scale!r}")
        srcs, gold = self._examples(examples)
        states = [self.state(s).force() for s in srcs]
        self._one_source(states)
        if scale == "rank":
            per = {id(lf): [] for lf in self.leaves()}
            for st in states:
                for ls in st.walk():
                    per[id(ls.leaf)].append(ls.sig)
            ranks = [_table(per[id(lf)]) for lf in self.leaves()]
            rk = {id(lf): r for lf, r in zip(self.leaves(), ranks)}
            sig = np.array([_rank(rk[id(ls.leaf)], ls.sig) for st in states for ls in st.walk()], float)
        else:
            ranks, rk = None, {}
            sig = np.array([x for st in states for x in st.sigs()], float)
        pts = np.unique(sig[np.isfinite(sig)]) if grid_of is None else np.asarray(grid_of(sig), float)
        grid = np.concatenate([pts, [np.inf]])
        n, G = len(states), len(grid)
        loss, auto_all, cost, calls, who = np.zeros((n, G)), np.zeros((n, G), bool), np.zeros((n, G)), np.zeros((n, G)), []
        answering = getattr(self, "_answering", None)
        for i, st in enumerate(states):
            auto, keys, _, vals, c, k = self.vec(st, grid, rk)
            right = {kk: self._right(v, gold[i]) for kk, v in vals.items()}
            ok = np.array([right[kk] for kk in keys])
            loss[i], auto_all[i], cost[i], calls[i] = auto & ~ok, auto, c, k
            who.append(None if answering is None else answering(st, grid, rk))
        return srcs, gold, states, ranks, rk, grid, loss, auto_all, cost, calls, who

    @_deprecate.removed_kwargs(error="max_error")
    def calibrate_for(self, examples, *, max_error=0.05, signal="auto", method="empirical", delta=0.10, scale="raw"):
        """Choose the shared threshold for a target error rate among the answers the combination gives alone, on
        labelled examples [(input, correct)] — as DecisionPart.calibrate_for, with one threshold t for every part's
        signal (on `scale`, as in act_guard). method="empirical": the lowest threshold at which the calibration
        decisions the combination answers alone are wrong at most `max_error` of the time — no guarantee on new inputs;
        method="ltt" (learn-then-test): the error among the answered is ≤ `max_error` with probability ≥ 1 − delta for
        inputs like the examples (a binomial test at every threshold of a grid of at most 64 quantiles of the signals,
        Bonferroni over the grid, which holds although a cascade's error is not monotone in t). No threshold reaches the
        target → everything escalates (inf). Replaces the parts' own thresholds inside the combination, changes its
        fingerprint and clears its conformal set.

        The decider protocol: the signature and the keys of DecisionPart.calibrate_for — "signal" ("shared" or
        "shared-rank"), "threshold", "answered", "error", "n", "max_error", "method", "guarantee" — plus
        "calls_per_question" and "scale". signal: "auto" only (each part brings its own). Every option after the
        examples is keyword-only (error=, the 0.7 name of max_error=, was removed in 0.9)."""
        from ..calibration import _binom_cdf, check_rate, ltt_grid
        error = max_error
        if method not in ("empirical", "ltt"):
            raise ValueError('method must be "empirical" or "ltt"')
        check_rate("max_error", error, zero=method == "empirical")
        if method == "ltt":
            check_rate("delta", delta)
        if signal != "auto":
            raise ValueError(f"signal={signal!r}: a combination's signal is each part's own (signal=\"auto\"); "
                             "scale=\"raw\" or \"rank\" says how they share one threshold")
        _, _, states, ranks, _, grid, loss, auto_all, _, calls, _ = self._sweep(
            examples, scale, ltt_grid if method == "ltt" else None)
        n, G = len(states), len(grid)
        answered, wrong = auto_all[:, :G - 1].sum(0), loss[:, :G - 1].sum(0)
        j = G - 1                                     # the inf threshold: nothing answered alone
        if method == "ltt":
            for i in range(G - 1):
                if answered[i] and _binom_cdf(float(wrong[i]), int(answered[i]), error) <= delta / max(1, G - 1):
                    j = i
                    break
            g = {"method": "ltt", "error": error, "delta": delta, "n": n, "signal": _SIGNAL[scale],
                 "promise": f"error among the answers given alone ≤ {error:g} with probability ≥ {1 - delta:g}, "
                            "for inputs like the calibration examples"}
        else:
            for i in range(G - 2, -1, -1):           # from the highest threshold down, while the error holds
                if not answered[i]:
                    continue
                if wrong[i] / answered[i] <= error + 1e-12:
                    j = i
                else:
                    break
            g = {"method": "empirical", "error": error, "n": n, "signal": _SIGNAL[scale],
                 "promise": "none: the error was measured on the calibration examples only"}
        t = float(grid[j])
        self.threshold, self.guarantee, self.groups = t, g, None
        self.scale, self.ranks = scale, ranks
        self.conformal_set = None
        self._setup()
        a, lo = auto_all[:, j], loss[:, j]
        return {"signal": _SIGNAL[scale], "threshold": t, "answered": float(a.mean()) if n else 0.0,
                "error": float(lo[a].sum() / a.sum()) if a.any() else 0.0, "n": n, "max_error": error,
                "method": method, "guarantee": g["promise"],
                "calls_per_question": float(calls[:, j].mean()) if n else 0.0, "scale": scale}

    # --- the calibration file's state of the combination (solvi.core.calibfile reads and writes only the file format)
    @property
    def _calibration_kind(self):
        return type(self).__name__

    def _calibration_base(self):
        """What a calibration is fitted to: every member's fingerprint and the combination's rule, without its thresholds."""
        return digest(type(self).__name__, self._describe(), [m.fingerprint() for m in self.members], {})

    def _calibration_models(self):
        """{model id: weights fingerprint} of the models behind the combination (for the message when they differ)."""
        return {str(lf.part.model_id): lf.part.model.weights_fingerprint() for lf in self.leaves()}

    def _calibration_thresholds(self):
        out = {"threshold": self.threshold}
        if self.scale is not None:
            out["scale"] = self.scale
        if self.scale == "rank":                    # each member's sorted calibration signals (at most multi.MAX_RANKS)
            out["ranks"] = [[float(x) for x in r] for r in self.ranks]
        return out

    def _calibration_adapter(self):
        return None

    def _apply_calibration(self, rec, grp, path):
        """Set the threshold a calibration file holds (solvi.core.calibfile.load has checked it)."""
        self.threshold, self.guarantee, self.groups = rec.get("threshold"), rec.get("guarantee"), grp
        scale = rec.get("scale", "raw")             # a file from before 0.7 has no scale: raw, as it was made
        if scale == "rank":
            ranks = rec.get("ranks")
            if not isinstance(ranks, list) or len(ranks) != len(self.leaves()):
                raise ValueError(f"{path}: a rank-scale calibration needs the calibration signals of each of the "
                                 f"{len(self.leaves())} models (\"ranks\")")
            self.scale, self.ranks = "rank", [np.asarray(r, float) for r in ranks]
        elif scale == "raw":
            self.scale, self.ranks = "raw", None
        else:
            raise ValueError(f"{path}: unknown scale {scale!r} (rank or raw)")
        self.conformal_set = rec.get("conformal")
        self._setup()

    def save_calibration(self, path):
        """Write the combination's calibration (the shared threshold, per group too, the guarantee, the conformal set) with
        the question and every member's fingerprint to a JSON file (solvi.core.calibfile). → path."""
        from ..calibfile import save
        return save(self, path)

    def load_calibration(self, path, groups=None, strict=True):
        """Apply a file written by save_calibration / `solvi calibrate`; refuses one made for other members or another
        question (strict=False loads it anyway). → self."""
        from ..calibfile import load
        return load(self, path, groups, strict)

    def conformal(self, examples, coverage=0.90):
        """Conformal answer sets for the combination's decisions (the probabilities it answers with: the answering
        stage's for a cascade, the mean of the parts' for a vote), from labelled examples at its current threshold —
        so call it after act_guard. Every decision then carries extra["candidates"]; an escalation lists them.
        Choice, yes/no, score and number questions. → {"coverage", "quantile", "n", "mean_size"}."""
        from ..calibration import check_rate, conformal_quantile, set_scores
        check_rate("coverage", coverage)
        if self.spec.multi or self.kind in ("rank", "span"):
            raise ValueError(f"conformal sets need a single answer from a closed list; not for {self.kind!r} questions")
        srcs, gold = self._examples(examples)
        saved, self.conformal_set = self.conformal_set, None
        try:
            ds = [self.final(self.state(s), None) for s in srcs]
        finally:
            self.conformal_set = saved
        ordinal = self.kind in ("score", "number")
        scores = []
        for d, y in zip(ds, gold):
            keys = list(d.probs)
            g = Unknown if y is Unknown else self.spec.label(y)
            if g not in keys:
                raise ValueError(f"{y!r}: this question cannot answer it (its answers: {keys})")
            scores.append(float(set_scores([d.probs[k] for k in keys], ordinal, Unknown in keys)[keys.index(g)]))
        q = conformal_quantile(scores, 1 - coverage)
        self.conformal_set = {"coverage": coverage, "quantile": q, "n": len(scores), "ordinal": ordinal}
        return {"coverage": coverage, "quantile": q, "n": len(scores),
                "mean_size": float(np.mean([len(self.candidates(d)) for d in ds]))}

    def calls(self):
        """What the combination cost since it was made: {"asked" (decisions), "calls" (models called, per part, in
        leaves order), "calls_per_question" (models called per decision), "cost" (with costs=)}. `usage` is a scorer's
        token count; usage() here was the 0.7 name of calls() (removed in 0.9)."""
        lv = self.leaves()
        calls = [lf.calls for lf in lv]
        out = {"asked": self.asked, "calls": {f"{i}:{lf.name}": c for i, (lf, c) in enumerate(zip(lv, calls))},
               "calls_per_question": sum(calls) / self.asked if self.asked else 0.0}
        if any(lf.cost_given for lf in lv):
            out["cost"] = sum(lf.cost * c for lf, c in zip(lv, calls)) / self.asked if self.asked else 0.0
        return out

    usage = _deprecate.removed_attr("usage()", "calls()", "Combination")

    # --- learning: every part learns
    def _each(self, x):
        for lf in self.leaves():
            yield lf.part, (lf.text(_Src(vals=dict(x))) if isinstance(x, Facts) else x)

    @_deprecate.removed_kwargs(x="text")
    def teach(self, text, correct):
        """One correction, absorbed by every part at once (an input, or Facts by name) → total ms."""
        return sum(p.teach(t, correct) for p, t in self._each(text))

    def fit(self, examples, lam=1.0, folds=4):
        """Few-shot "S" for every part (see DecisionPart.fit) → [Adaptation]."""
        ex = list(examples)
        return [lf.part.fit([(lf.text(_Src(vals=dict(x))) if isinstance(x, Facts) else x, y) for x, y in ex], lam, folds)
                for lf in self.leaves()]

    @_deprecate.removed_kwargs(inputs="texts")
    def adapt(self, texts):
        """Label-bias correction for every part (see DecisionPart.adapt) → [per part]."""
        xs = list(texts)
        return [lf.part.adapt([lf.text(_Src(vals=dict(x))) if isinstance(x, Facts) else x for x in xs])
                for lf in self.leaves()]

    def reset(self):
        """Every member part's reset() (their adaptations and calibrations)."""
        for lf in self.leaves():
            lf.part.reset()

    memory = gone_in_1_0("memory()", "solvi.core.knowledge.memory.attach(combination, ...) (a memory for every part) — the memory of "
                         "corrections moves into the knowledge memory", "Combination")

    remove_lora = gone_in_1_0("remove_lora()", "solvi.experimental.lora.remove_lora(combination) (every part's, in leaves order)",
                              "Combination")
    adapt_lora = gone_in_1_0("adapt_lora()", "solvi.experimental.lora.adapt_lora(part, examples, ...) on one of its parts, then "
                             "calibrate the combination again", "Combination")

    # --- what belongs to one part, not to a combination: each raises, saying where it is
    def _one_part(self, what, why):
        raise NotImplementedError(f"{type(self).__name__}.{what}: {why} — call it on the part "
                                  f"({self.__name__}.parts[i].{what}), then calibrate the combination again if it "
                                  "changed the part")

    def save_lora(self, path):
        """Not for a combination (raises NotImplementedError): an adapter file holds one part's adapter."""
        self._one_part("save_lora()", "an adapter file holds one part's adapter")

    def load_lora(self, path, strict=True):
        """Not for a combination (raises NotImplementedError): an adapter file holds one part's adapter."""
        self._one_part("load_lora()", "an adapter file holds one part's adapter")

    def budget(self):
        """Not for a combination (raises NotImplementedError): each part reads an input by its own model's length."""
        self._one_part("budget()", "each part reads an input by its own model's length and long= setting")

    def sections_k(self):
        """Not for a combination (raises NotImplementedError): each part retrieves by its own long= setting."""
        self._one_part("sections_k()", "each part retrieves the sections of a long text by its own long= setting")

    def long_key(self):
        """Not for a combination (raises NotImplementedError): each part reads long texts by its own long= setting
        (every part's is in the combination's fingerprint)."""
        self._one_part("long_key()", "each part reads long texts by its own long= setting")

    def long_input(self, text):
        """Not for a combination (raises NotImplementedError): each part reads a long text by its own long= setting."""
        self._one_part("long_input()", "each part reads a long text by its own long= setting")

    def in_pass(self, siblings, args, names=None):
        """Not for a combination (raises NotImplementedError): the runtime's shared forward pass is for parts of one
        model; the strategist never puts a combination in one (plan_batches)."""
        self._one_part("in_pass()", "a shared forward pass is for parts of one model, and the strategist never puts a "
                                    "combination in one")

    # --- replay without re-running the models
    def check_record(self, r):
        """A recorded decision of this combination, without re-running any model: does the recorded answer follow from
        the recorded proposals by this combination's rule? → [reason] (empty: it does)."""
        e = dict(r.extra or {})
        from ..runtime import MISSING
        e["escalate"] = r.error
        e["value"] = None if (r.error is not None or r.value is MISSING) else _jv(r.value)
        return self.check(e, top=True)

    def check(self, e, top=False):
        raise NotImplementedError


def _escalated(e, top):
    """Was this (recorded) decision escalated by the combination? A top-level record's error may also be another
    safeguard (min_confidence, a type) that rejected an answer the combination gave."""
    return e.get("escalate") is not None and (not top or str(e["escalate"]).startswith(ESCALATED))


def _same(jv_a, jv_b):
    from ..runtime import vhash
    if isinstance(jv_b, dict) and "quote" in jv_b and not isinstance(jv_a, dict):
        return jv_a == jv_b["quote"][0]              # a span's record value is its text
    return vhash(jv_a) == vhash(jv_b)


class Cascade(Combination):
    """Ask the parts in order; answer with the first whose decision does not escalate; escalate when all do (with the
    last part's answer as "would have answered"). The next model is asked only when the one before escalates, so a
    cheap model that is often sure saves the large model's calls — `calls()` and `extra["calls"]` count them;
    `costs=[45, 137]` (ms, per part) makes act_guard and usage report the expected cost; a member that is itself a
    combination takes its parts' costs itself (Vote([...], costs=[...])), so costs= with one raises."""

    kind_name = "cascade"

    def _sep(self):
        return " → "

    def state(self, src):
        return _State(self.members, src)

    def vec(self, st, ts, rk=None):
        ts, rk = self._vt(ts, st.src, rk)
        G = len(ts)
        auto, keys, sig = np.zeros(G, bool), np.empty(G, dtype=object), np.zeros(G)
        cost, calls, vals = np.zeros(G), np.zeros(G), {}
        for i, m in enumerate(self.members):
            if auto.all():
                break
            a, k, s, v, c, n = m.vec(st.get(i), ts, rk)
            vals.update(v)
            asked = ~auto
            cost += np.where(asked, c, 0.0)
            calls += np.where(asked, n, 0.0)
            put = asked & a if i < len(self.members) - 1 else asked       # not answered by anyone: the last proposal
            keys[put], sig[put] = k[put], s[put]
            auto |= a
        return auto, keys, sig, vals, cost, calls

    def _answering(self, st, ts, rk=None):
        """The stage that answers at each threshold (−1: every stage escalates)."""
        ts, rk = self._vt(ts, st.src, rk)
        who = np.full(len(ts), -1)
        for i, m in enumerate(self.members):
            a = m.vec(st.get(i), ts, rk)[0]
            who = np.where((who < 0) & a, i, who)
        return who

    def final(self, st, t, rk=None):
        te, rk = self._t(t, st.src, rk)
        stages, ds, by = [], [], None
        for i, m in enumerate(self.members):
            s = st.get(i)
            d = m.final(s, te, rk)
            stages.append(m.entry(d, s))
            ds.append(d)
            if d.escalate is None:
                by = i
                break
        use = ds[-1]
        out = Decision(use.value, dict(use.probs), confidence=use.conf, evidence=list(use.evidence))
        out.extra = {"stages": stages, "answered_by": by, "calls": st.calls()}
        if by is None:
            out.escalate = (f"{ESCALATED}: every model of the cascade escalated ("
                            + "; ".join(f"{_who(e)}: {e['escalate']}" for e in stages) + ")")
        return self._wrapup(out, t, st.src)

    def check(self, e, top=False):
        stages, by = e.get("stages"), e.get("answered_by")
        if not stages:
            return ["a cascade's record lists no stages"]
        bad = []
        for m, s in zip(self.members, stages):
            bad += m.check(s)
        if len(stages) > len(self.members):
            return bad + [f"{len(stages)} stages recorded, the cascade has {len(self.members)}"]
        if by is None:
            if len(stages) != len(self.members) or any(s.get("escalate") is None for s in stages):
                bad.append("recorded as escalated, but not every stage escalated")
            if e.get("escalate") is None:
                bad.append("every stage escalated, but the cascade answered")
            return bad
        if by != len(stages) - 1 or any(s.get("escalate") is not None for s in stages[:by]) \
                or stages[by].get("escalate") is not None:
            bad.append(f"stage {by + 1} is recorded as answering, but the recorded stages do not say so")
        elif _escalated(e, top):
            bad.append(f"stage {by + 1} answered, but the cascade escalated")
        elif e.get("value") is not None and not _same(e["value"], stages[by]["value"]):
            bad.append(f"answer {e['value']!r} ≠ stage {by + 1}'s proposal {stages[by]['value']!r}")
        return bad


def _rule(K, A, rule):
    """Votes K [M, G] (value keys) and whether each part answers alone A [M, G] → (agreement [G], the agreed — or would-be
    — key [G], agreeing parts [M, G], answer [G]). "all": every part proposes the same value; "majority": more than half
    of the parts do. Either way every agreeing part must answer alone (its signal ≥ the threshold, no other safeguard)."""
    M, G = K.shape
    if rule == "all":
        agree = (K == K[0]).all(0)
        vk = np.where(agree, K[0], K[-1])
    else:
        cnt = np.stack([(K == K[j]).sum(0) for j in range(M)])
        maj = cnt * 2 > M
        agree = maj.any(0)
        j = np.argmax(maj, 0)
        vk = np.where(agree, K[j, np.arange(G)], K[-1])
    mask = K == vk[None, :]
    return agree, vk, mask, agree & (A | ~mask).all(0)


class Vote(Combination):
    """Ask every part; answer when the rule holds — "all": every part proposes the same value, "majority": more than
    half do — and every agreeing part answers alone (its signal ≥ the threshold); otherwise escalate, listing the
    proposals. Parts of one model that can share a forward pass are asked in one pass. The probabilities are the mean
    of the parts'; the confidence the lowest among the agreeing parts'. Models of different families tend to disagree
    more usefully than a student and its teacher, which often make the same mistakes."""

    kind_name = "vote"

    def __init__(self, members, rule="all", name=None, costs=None):
        if rule not in ("all", "majority"):
            raise ValueError('rule must be "all" or "majority"')
        self.rule = rule
        super().__init__(members, name, costs)

    @property
    def model_id(self):
        return f"vote:{self.rule}({', '.join(m.model_id for m in self.members)})"

    def _describe(self):
        return {"rule": self.rule}

    def state(self, src):
        pre, groups = {}, {}
        for i, m in enumerate(self.members):          # parts of one model: their questions in one forward pass
            if isinstance(m, _Leaf) and m.part.model.batchable and m.part.option_order != "average" \
                    and not m.part.spec.pointer and not (m.part.long is not None and m.part._too_long(m.text(src))):
                groups.setdefault((id(m.part.model), m.text(src)), []).append(i)
        for (_, text), ix in groups.items():
            if len(ix) > 1:
                model = self.members[ix[0]].part.model
                for i, (z, a, _) in zip(ix, model._raw_pass([self.members[i].part.spec for i in ix], text)):
                    pre[i] = (z, a)
        return _State(self.members, src, pre).force()

    def vec(self, st, ts, rk=None):
        ts, rk = self._vt(ts, st.src, rk)
        outs = [m.vec(st.get(i), ts, rk) for i, m in enumerate(self.members)]
        A = np.stack([o[0] for o in outs])
        K = np.stack([o[1] for o in outs])
        S = np.stack([o[2] for o in outs])
        _, vk, mask, auto = _rule(K, A, self.rule)
        sig = np.where(mask, S, np.inf).min(0)
        vals = {}
        for o in outs:
            vals.update(o[3])
        return auto, vk, sig, vals, sum(o[4] for o in outs), sum(o[5] for o in outs)

    def final(self, st, t, rk=None):
        te, rk = self._t(t, st.src, rk)
        ds = [m.final(st.get(i), te, rk) for i, m in enumerate(self.members)]
        votes = [m.entry(d, st.get(i)) for i, (m, d) in enumerate(zip(self.members, ds))]
        K = np.array([[_key(d.value)] for d in ds], dtype=object)
        A = np.array([[d.escalate is None] for d in ds])
        agree, _, mask, auto = _rule(K, A, self.rule)
        agreeing = [d for d, m in zip(ds, mask[:, 0]) if m]
        use = agreeing[0] if agree[0] else ds[-1]
        keys = list(dict.fromkeys(k for d in ds for k in d.probs))
        probs = {k: float(np.mean([d.probs[k] for d in ds if k in d.probs])) for k in keys}
        conf = min(d.conf for d in agreeing) if agree[0] else use.conf
        out = Decision(use.value, probs, confidence=conf, evidence=list(use.evidence))
        out.extra = {"votes": votes, "rule": self.rule, "calls": st.calls()}
        if not auto[0]:
            if not agree[0]:
                out.escalate = (f"{ESCALATED}: the models disagree ({self.rule}): "
                                + ", ".join(f"{_who(e)} {_shown(e['value'])}" for e in votes))
            else:
                out.escalate = (f"{ESCALATED}: the models agree on {use.value!r} ({self.rule}), but "
                                + "; ".join(f"{_who(e)}: {e['escalate']}" for e, m in zip(votes, mask[:, 0])
                                            if m and e["escalate"] is not None))
        return self._wrapup(out, t, st.src)

    def check(self, e, top=False):
        votes = e.get("votes")
        if not votes or len(votes) != len(self.members):
            return [f"a vote's record lists {len(votes or [])} votes, the vote has {len(self.members)} parts"]
        bad = []
        for m, v in zip(self.members, votes):
            bad += m.check(v)
        from ..runtime import vhash
        K = np.array([[vhash(v["value"])] for v in votes], dtype=object)
        A = np.array([[v.get("escalate") is None] for v in votes])
        _, vk, _, auto = _rule(K, A, e.get("rule", self.rule))
        if auto[0] and _escalated(e, top):
            bad.append("the recorded votes agree and answer alone, but the vote escalated")
        elif not auto[0] and e.get("escalate") is None:
            bad.append("the recorded votes do not let the vote answer, but it answered")
        elif auto[0] and e.get("value") is not None:
            v = next(v["value"] for v in votes if vhash(v["value"]) == vk[0])
            if not _same(e["value"], v):
                bad.append(f"answer {e['value']!r} ≠ the agreed proposal {v!r}")
        return bad


class Route(Combination):
    """Pick one part per input by code: `routes` maps a predicate (a function of facts by name — its parameters are
    facts the route reads — returning true to take that part) or a fact name (its value is true) to a part; the first
    that holds picks, else `default`. Only the picked part's model is called; `extra["route"]` records which part and
    why, `extra["routed"]` its proposal. An input that is not Facts(...) (decide(text), the examples of act_guard): a
    state with the route's facts as keys gives them; otherwise a predicate with one parameter is called with the input
    itself, and a route keyed by a fact name — or a predicate of several facts — raises: a text is not the value of a
    fact. A fact that Facts(...) does not give raises too (it is not read as false)."""

    kind_name = "route"

    def __init__(self, routes, default, name=None, costs=None):
        if not isinstance(routes, dict) or not routes:
            raise ValueError("routes: {predicate or fact name: part, ...}")
        self.keys = list(routes)
        for k in self.keys:
            if not (callable(k) or isinstance(k, str)):
                raise TypeError(f"a route's key is a predicate or a fact name, not {k!r}")
        self._params = {i: (list(inspect.signature(k).parameters) if callable(k) else [k]) for i, k in enumerate(self.keys)}
        super().__init__(list(routes.values()) + [default], name, costs)

    def _extra_facts(self):
        return [f for ps in self._params.values() for f in ps] + super()._extra_facts()

    def _by(self, i):
        if i >= len(self.keys):
            return "default"
        k = self.keys[i]
        return k if isinstance(k, str) else getattr(k, "__name__", "predicate")

    @property
    def model_id(self):
        return "route(" + " | ".join(f"{self._by(i)}: {m.model_id}" for i, m in enumerate(self.members)) + ")"

    def _describe(self):
        return {"routes": [k if isinstance(k, str) else code_fingerprint(k) for k in self.keys]}

    def pick(self, src):
        """The index of the member this input goes to."""
        for i, k in enumerate(self.keys):
            ps = self._params[i]
            given = src.vals if src.vals is not None else src.raw if isinstance(src.raw, Mapping) else None
            if given is not None and all(p in given for p in ps):
                args = {p: (given[p].value if isinstance(given[p], Quote) else given[p]) for p in ps}
            elif src.vals is not None:              # a missing fact is not "false": the route cannot be decided
                raise ValueError(f"route {self._by(i)} reads {[p for p in ps if p not in given]}, which the input does "
                                 f"not give (it gives {sorted(given)})")
            elif callable(k) and len(ps) == 1:      # a predicate of the input itself
                args = {ps[0]: src.raw}
            else:                                   # a raw input is not the value of a fact
                raise ValueError(f"route {self._by(i)} reads the fact{'s' if len(ps) > 1 else ''} {ps}: give the input "
                                 "as Facts(...) or a state with those keys")
            if (bool(args[k]) if isinstance(k, str) else bool(k(**args))):
                return i
        return len(self.keys)

    def state(self, src):
        return _State(self.members, src, pick=self.pick(src))

    def vec(self, st, ts, rk=None):
        return self.members[st.pick].vec(st.get(st.pick), *self._vt(ts, st.src, rk))

    def final(self, st, t, rk=None):
        i, m = st.pick, self.members[st.pick]
        s = st.get(i)
        d = m.final(s, *self._t(t, st.src, rk))
        out = Decision(d.value, dict(d.probs), confidence=d.conf, evidence=list(d.evidence), escalate=d.escalate)
        out.extra = {"route": {"to": i, "part": m.name, "by": self._by(i)}, "routed": m.entry(d, s), "calls": st.calls()}
        return self._wrapup(out, t, st.src)

    def check(self, e, top=False):
        rt, ent = e.get("route"), e.get("routed")
        if not isinstance(rt, dict) or not isinstance(ent, dict) or not 0 <= int(rt.get("to", -1)) < len(self.members):
            return ["a route's record does not say which part it picked"]
        bad = self.members[int(rt["to"])].check(ent)
        routed_esc = ent.get("escalate") is not None
        if routed_esc and e.get("escalate") is None:
            bad.append("the routed part escalated, but the route answered")
        elif not routed_esc and not top and e.get("escalate") is not None:
            bad.append("the routed part answered, but the route escalated")
        elif not routed_esc and e.get("value") is not None and not _same(e["value"], ent["value"]):
            bad.append(f"answer {e['value']!r} ≠ the routed part's proposal {ent['value']!r}")
        return bad


__all__ = ["Cascade", "Combination", "MAX_RANKS", "RECORD_KEYS", "Route", "Vote"]
