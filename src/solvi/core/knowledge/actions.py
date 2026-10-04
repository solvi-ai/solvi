"""The action model: what an environment accepts, refuses and changes — learned conservatively from outcomes, over an
explicit vocabulary of conditions.

    from solvi.core.knowledge import ConservativeActionModel, Vocabulary
    vocab = Vocabulary({"status": lambda state, args: state["orders"][args["order_id"]]["status"],
                        "own_payment": lambda state, args: args["payment_id"] in state["user"]["payments"]})
    am = ConservativeActionModel(vocab)
    am.observe(state, "cancel_order", {"order_id": "A1"}, accepted=False)             # the environment refused it
    am.observe(state2, "cancel_order", {"order_id": "A2"}, accepted=True, effect={"status": "cancelled"})
    p = am.predict(state3, "cancel_order", {"order_id": "A3"})
    p.verdict, p.risk, p.support, p.reason, p.hard, p.effects     # "accept" | "refuse" | "unknown", ...

The ActionModel protocol: observe(state, action, args, accepted, effect=None), predict(state, action, args) →
Prediction, fingerprint(). Every prediction carries both a verdict and an estimate (the risk if the action is taken and
the number of cases behind it); a RiskPolicy (solvi.core.knowledge.risk) turns them into take / avoid / ask System 2.

ConservativeActionModel: per action, the values
each condition of the vocabulary took in accepted transitions are "allowed"; a refused transition is explained by the
conditions whose value was never accepted, and the minimal such sets are its refusal signatures. predict says "accept"
when every condition holds a value seen accepted, "refuse" when the violated conditions contain a signature, and
"unknown" otherwise — and for an action never observed, or one with refusals the vocabulary cannot explain (then only
exactly seen condition vectors are answered). "unknown" hands over to System 2 or a person. Effects: predicted when
every accepted transition of the same condition vector (else of the action) had the same effect.

Scope, stated plainly — this is what "stable" covers and what it does not:
- vocabulary-bound: the model can only learn checks the vocabulary can express. On τ-bench retail probes
  (benchmarks/knowledge/taubench_action_model.py), with a vocabulary written by someone who had read the tools' code,
  refusal precision and recall were 1.000 (7,602 of 7,602 answered held-out refusals) and it abstained on 0.26%;
  with pair features left out it abstained on 2.6%, with money comparisons left out on 10.7% and made 1 false refusal
  (a condition vector seen only refused, which a shorter vocabulary cannot tell from an accepted one);
- it learns what the environment checks: a rule the environment does not enforce (confirm with the customer,
  authenticate first, policy-only rules) is never refused by it — those must come from a written policy (agenda gates,
  hard checks);
- necessary conditions transfer; sufficient conditions are not promised to: conditions that held at every success by
  accident (an item carried, a material nearby, a place's name) make rules learned in one world over-conservative in a
  new one — "unknown", not wrong (benchmarks/knowledge/toy_crafting.py, arm km_place);
- learned from an agent's own traces behind a hand guard it sees few refusals and abstains often: sound where it
  answers, little coverage.

Vocabulary: {name: predicate(state, args) → a JSON scalar}. A predicate that cannot be read (a KeyError, an entity not
looked up) gives MISSING, which is never "allowed" until an accepted transition shows it — so the model says "unknown"
there instead of guessing. `hard=` names conditions whose violation is a hard rule (instant harm, a policy): a refusal
resting on one is a hard prediction that no risk policy trades. The vocabulary's fingerprint (each predicate's code) is
in the model's fingerprint and in every action item it commits to a KnowledgeStore."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..provenance import code_fingerprint, digest

MISSING = "<missing>"                    # a predicate that could not be read on this state
VERDICTS = ("accept", "refuse", "unknown")


@dataclass
class Prediction:
    """What an action model (or any knowledge item) says about taking an action: the verdict, the estimated risk if the
    action is taken (probability of refusal / harm × its severity), the number of cases behind the estimate (`support`;
    -1: a rate from a written spec), the reason, whether a hard rule stands behind a refusal (never traded), and the
    predicted effects (None: not predicted)."""
    verdict: str
    risk: float = 0.0
    support: int = 0
    reason: str = ""
    hard: bool = False
    effects: object = None
    action: str | None = None

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict is one of {VERDICTS}, not {self.verdict!r}")

    def to_dict(self):
        return {"verdict": self.verdict, "risk": round(float(self.risk), 6), "support": int(self.support),
                "reason": self.reason, "hard": self.hard, "effects": self.effects, "action": self.action}


@runtime_checkable
class ActionModel(Protocol):
    """observe what the environment did; predict accept / refuse / unknown with risk and support; fingerprint."""

    def observe(self, state, action, args, accepted, effect=None): ...

    def predict(self, state, action, args) -> Prediction: ...

    def fingerprint(self) -> str: ...


class Vocabulary:
    """The conditions an action model may learn from: {name: predicate(state, args) → JSON scalar}. hard: names whose
    violation is a hard rule; only: {action: [names]} — the conditions read for that action (default: all)."""

    def __init__(self, predicates, *, hard=(), only=None):
        if not predicates:
            raise ValueError("a vocabulary has at least one predicate")
        bad = [n for n, f in predicates.items() if not callable(f)]
        if bad:
            raise TypeError(f"predicates are functions (state, args) → value; not callable: {bad}")
        self.predicates = dict(sorted(predicates.items()))
        self.hard = frozenset(hard)
        unknown = sorted(self.hard - set(self.predicates))
        if unknown:
            raise ValueError(f"hard names conditions of the vocabulary; unknown: {unknown}")
        self.only = {a: tuple(sorted(ns)) for a, ns in (only or {}).items()}
        for a, ns in self.only.items():
            missing = sorted(set(ns) - set(self.predicates))
            if missing:
                raise ValueError(f"only[{a!r}] names unknown conditions: {missing}")

    def names(self, action):
        """The condition names read for an action, in order."""
        return self.only.get(action) or tuple(self.predicates)

    def features(self, state, action, args):
        """The condition values for one (state, action, args) → tuple (MISSING where a predicate cannot be read)."""
        out = []
        for n in self.names(action):
            try:
                v = self.predicates[n](state, args)
            except (LookupError, AttributeError, TypeError, ValueError):
                v = MISSING
            if isinstance(v, list):
                v = tuple(v)
            out.append(v)
        return tuple(out)

    def predicate_fingerprints(self):
        """{name: the predicate's code fingerprint}."""
        return {n: code_fingerprint(f) for n, f in self.predicates.items()}

    def fingerprint(self):
        return digest("Vocabulary", self.predicate_fingerprints(), sorted(self.hard),
                      {a: list(ns) for a, ns in sorted(self.only.items())})


class _Learned:
    """One action's learned model (rebuilt from its transitions when they change)."""

    def __init__(self, transitions, names):
        acc = [t for t in transitions if t[1]]
        self.allowed = [set() for _ in names]
        for fv, _, _ in acc:
            for i, v in enumerate(fv):
                self.allowed[i].add(v)
        sigs, self.unexplained = defaultdict(int), 0
        self.violations = []
        for fv, ok, _ in transitions:
            if not ok:
                v = self.violated(fv)
                if v:
                    self.violations.append(v)
                else:
                    self.unexplained += 1
        keep = []
        for s in sorted(set(self.violations), key=_sig_order):
            if not any(k <= s for k in keep):
                keep.append(s)
        for v in self.violations:
            for k in keep:
                if k <= v:
                    sigs[k] += 1
        self.signatures = [(k, sigs[k]) for k in keep]
        self.n_accepted, self.n = len(acc), len(transitions)
        self.lookup = defaultdict(set)
        for fv, ok, _ in transitions:
            self.lookup[fv].add(ok)
        effects, by_fv = [], defaultdict(list)
        for fv, ok, eff in acc:
            effects.append(eff)
            by_fv[fv].append(eff)
        self.effect = effects[0] if effects and all(e == effects[0] for e in effects) else None
        self.effect_varies = bool(effects) and self.effect is None
        self.effect_by_fv = {fv: es[0] for fv, es in by_fv.items() if all(e == es[0] for e in es)}
        self._fv_accepted = set(by_fv)

    def violated(self, fv):
        return frozenset((i, v) for i, v in enumerate(fv) if v not in self.allowed[i])

    def add(self, fv, ok, eff):
        """One more transition, learned in place → False when it changes the allowed values (rebuild instead)."""
        if ok and any(v not in self.allowed[i] for i, v in enumerate(fv)):
            return False
        self.n += 1
        self.lookup[fv].add(ok)
        if ok:
            self.n_accepted += 1
            if self.n_accepted == 1:
                self.effect, self.effect_varies = eff, False
            elif not self.effect_varies and eff != self.effect:
                self.effect, self.effect_varies = None, True
            if fv not in self._fv_accepted:
                self._fv_accepted.add(fv)
                self.effect_by_fv[fv] = eff
            elif fv in self.effect_by_fv and self.effect_by_fv[fv] != eff:
                del self.effect_by_fv[fv]
            return True
        v = self.violated(fv)
        if not v:
            self.unexplained += 1
            return True
        self.violations.append(v)
        sigs, hit = [], False
        for k, c in self.signatures:
            if k <= v:
                c, hit = c + 1, True
            sigs.append((k, c))
        if not hit:                                       # a new minimal signature: supersets of it are not minimal
            sigs = [(k, c) for k, c in sigs if not v <= k] + [(v, sum(1 for w in self.violations if v <= w))]
            sigs.sort(key=lambda kc: _sig_order(kc[0]))
        self.signatures = sigs
        return True


def _sig_order(s):
    return (len(s), sorted(map(repr, s)))


class ConservativeActionModel:
    """See the module docstring. vocabulary: a Vocabulary; severity: the harm of a refused / failed action in your
    unit (a number, {action: number}, default 1.0) — risk = P(refused) × severity; store: a KnowledgeStore that every
    observation is journaled into (and from which `replay` rebuilds the model)."""

    def __init__(self, vocabulary, *, severity=1.0, store=None):
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("ConservativeActionModel(vocabulary=Vocabulary({...})): the vocabulary is an explicit "
                            "argument — the model learns only what it can express")
        self.vocabulary = vocabulary
        self.severity = severity
        self.store = store
        self.transitions = defaultdict(list)          # action → [(features, accepted, effect)]
        self._learned = {}
        self.surprises = 0

    def _sev(self, action):
        s = self.severity.get(action, 1.0) if isinstance(self.severity, dict) else self.severity
        return float(s)

    def _model(self, action):
        m = self._learned.get(action)
        if m is None and self.transitions.get(action):
            m = self._learned[action] = _Learned(self.transitions[action], self.vocabulary.names(action))
        return m

    def observe(self, state, action, args, accepted, effect=None):
        """What the environment did with this call: accepted (with `effect`, any JSON value — the change it made) or
        refused. → the prediction made before this observation (compare it with what happened: a surprise is counted
        in `surprises`)."""
        fv = self.vocabulary.features(state, action, args)
        before = self._predict_fv(action, fv)
        surprise = (before.verdict == "accept" and not accepted) or (before.verdict == "refuse" and accepted)
        self.surprises += surprise
        self._add(action, fv, bool(accepted), effect)
        if self.store is not None:
            self.store._write("observe", action=action, fv=list(fv), accepted=bool(accepted), effect=effect,
                              predicted=before.verdict, surprise=surprise, vocabulary=self.vocabulary.fingerprint())
        return before

    def _add(self, action, fv, accepted, effect):
        fv = tuple(fv)
        self.transitions[action].append((fv, accepted, effect))
        m = self._learned.get(action)
        if m is not None and not m.add(fv, accepted, effect):
            del self._learned[action]                  # the allowed values grew: rebuilt on the next prediction

    def predict(self, state, action, args):
        """→ Prediction(verdict, risk, support, reason, hard, effects) for taking `action` with `args` in `state`."""
        return self._predict_fv(action, self.vocabulary.features(state, action, args))

    def _predict_fv(self, action, fv):
        sev, names = self._sev(action), self.vocabulary.names(action)
        m = self._model(action)
        if m is None:
            return Prediction("unknown", 0.5 * sev, 0, "action never observed", action=action)
        if m.unexplained:
            seen = m.lookup.get(fv)
            if seen is not None and len(seen) == 1:
                ok = True in seen
                n = sum(1 for t in self.transitions[action] if t[0] == fv)
                risk = sev / (n + 2) if ok else sev * (n + 1) / (n + 2)
                return Prediction("accept" if ok else "refuse", risk, n, "condition vector seen exactly",
                                  effects=m.effect_by_fv.get(fv) if ok else None, action=action)
            return Prediction("unknown", 0.5 * sev, 0, f"{m.unexplained} refusal(s) the vocabulary cannot explain; this "
                              "condition vector not seen", action=action)
        v = m.violated(fv)
        if not v:
            eff = m.effect_by_fv.get(fv, m.effect)
            return Prediction("accept", sev / (m.n_accepted + 2), m.n_accepted, "every condition holds a value seen "
                              "accepted", effects=eff, action=action)
        for sig, n in m.signatures:
            if sig <= v:
                why = ", ".join(f"{names[i]}={x!r}" for i, x in sorted(sig, key=lambda p: p[0]))
                hard = any(names[i] in self.vocabulary.hard for i, _ in sig)
                return Prediction("refuse", sev * (n + 1) / (n + 2), n, f"refused before when {why}", hard=hard,
                                  action=action)
        why = ", ".join(f"{names[i]}={x!r}" for i, x in sorted(v, key=lambda p: p[0])[:3])
        return Prediction("unknown", 0.5 * sev, 0, f"no evidence for {why}", action=action)

    def summary(self, action):
        """What the model learned about one action: allowed values per condition, refusal signatures with their counts,
        unexplained refusals, transitions, the effect (when constant)."""
        m = self._model(action)
        if m is None:
            return None
        names = self.vocabulary.names(action)
        return {"conditions": list(names),
                "allowed": {names[i]: sorted(map(repr, a)) for i, a in enumerate(m.allowed)},
                "refusals": [{"when": {names[i]: repr(x) for i, x in sorted(s, key=lambda p: p[0])}, "n": n}
                             for s, n in m.signatures],
                "unexplained": m.unexplained, "transitions": m.n, "accepted": m.n_accepted, "effect": m.effect,
                "effect_varies": m.effect_varies}

    def actions(self):
        return sorted(self.transitions)

    def fingerprint(self):
        """The vocabulary's fingerprint and every transition the model learned from."""
        return digest("ConservativeActionModel", self.vocabulary.fingerprint(), self.severity,
                      {a: [[list(fv), ok, eff] for fv, ok, eff in ts] for a, ts in sorted(self.transitions.items())})

    def commit(self, store=None, *, scope=None):
        """Write each action's learned model into a KnowledgeStore as an "action" item (source "outcome", by
        "environment"; the vocabulary's predicate fingerprints in the body): the newer version refutes the older one.
        → {action: item id}."""
        store = store if store is not None else self.store
        if store is None:
            raise ValueError("commit(store): no KnowledgeStore given and none set on the model")
        out = {}
        sc = {"vocabulary": self.vocabulary.fingerprint(), **(scope or {})}
        preds = self.vocabulary.predicate_fingerprints()
        with store.batch():
            for a in self.actions():
                s = self.summary(a) or {}
                body = {"s": a, "r": "model", "o": {**s, "predicates": {n: preds[n] for n in s["conditions"]}}}
                out[a] = store.add("action", body, sc, source="outcome", by="environment",
                                   evidence=[s["transitions"], s["accepted"]])
        return out

    @classmethod
    def replay(cls, store, vocabulary, *, severity=1.0):
        """The model rebuilt from the observations journaled in `store` with this vocabulary (same fingerprint)."""
        m = cls(vocabulary, severity=severity)
        fp = vocabulary.fingerprint()
        for r in store.journal:
            if r.get("op") == "observe" and r.get("vocabulary") == fp:
                fv = tuple(tuple(x) if isinstance(x, list) else x for x in r["fv"])
                m._add(r["action"], fv, bool(r["accepted"]), r.get("effect"))
        return m


__all__ = ["ActionModel", "ConservativeActionModel", "MISSING", "Prediction", "VERDICTS", "Vocabulary"]
