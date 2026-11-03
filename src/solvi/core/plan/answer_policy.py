"""Offline candidate-answer policy, faithfully ported from the archived prototype.

This is a conditional model, not a guarantee of correctness or independence.
Its historical research acceptance remains separate from functional tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _number(x, name, *, positive=False, probability=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number")
    if x < 0 or (positive and x == 0) or (probability and x > 1):
        raise ValueError(f"{name} is outside its domain")
    return float(x)


def _names(names):
    names = tuple(names)
    if any(not isinstance(k, str) or not k for k in names) or len(set(names)) != len(names):
        raise ValueError("check names must be distinct nonempty strings")
    return names


def _record(record, names=None, *, training=False):
    if not isinstance(record, dict) or not isinstance(record.get("checks"), dict):
        raise ValueError("record must contain checks")
    _finite_answer(record.get("answer"))
    if not isinstance(record.get("correct"), bool):
        raise ValueError("record correct must be boolean")
    if "gold" in record and record["gold"] is not None:
        _finite_answer(record["gold"])
    for k, c in record["checks"].items():
        if not isinstance(k, str) or not isinstance(c, dict) or c.get("status") not in ("agree", "fail", "disagree"):
            raise ValueError("invalid stored check")
        if not isinstance(c.get("correct"), bool):
            raise ValueError("check correct must be boolean")
        if c["status"] != "fail":
            _finite_answer(c.get("value"))
        elif c.get("value") is not None:
            raise ValueError("failed check must have value None")
    if names is not None and not set(names).issubset(record["checks"]):
        raise ValueError("record lacks a configured check")


def _finite_answer(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("answer values must be finite numbers")


OUT = ("agree", "fail", "truth", "other")             # outcome classes when a0 is wrong ("truth"/"other" = disagree)


def close(a, b, rel: float = 1e-6) -> bool:
    return a is not None and b is not None and abs(a - b) <= rel * max(1.0, abs(b))


# ----------------------------------------------------------------------------------------------------- check model
@dataclass
class CheckParams:
    s: float          # P(agree | a0 right)
    phic: float       # P(fail  | a0 right)
    f: float          # P(agree | a0 wrong)        — the check repeats the mistake
    t: float          # P(true value | a0 wrong)
    phiw: float       # P(fail  | a0 wrong)

    @property
    def r(self) -> float:
        return max(1e-9, 1.0 - self.f - self.t - self.phiw)

    @property
    def dis_c(self) -> float:
        return max(1e-9, 1.0 - self.s - self.phic)


class CheckModel:
    """Per-check parameters estimated from answer records with stored check values and gold, pooled across checks:
    each check's Dirichlet prior is the pooled outcome distribution of all checks (weight `pool` pseudo-counts), so a
    check with no wrong-class examples (or a new check) starts at what checks have been worth so far (S102).

    records: {"answer", "gold" or "correct", "checks": {k: {"value", "status", "correct"}}}."""

    def __init__(self, checks=None, pool: float = 4.0, alpha: float = 0.5, collision_prior=(1.0, 4.0)):
        self.checks = None if checks is None else _names(checks)
        self.pool = _number(pool, "pool", positive=True)
        self.alpha = _number(alpha, "alpha", positive=True)
        if len(collision_prior) != 2:
            raise ValueError("collision_prior needs two positive counts")
        self.cprior = tuple(_number(x, "collision prior", positive=True) for x in collision_prior)

    def fit(self, records):
        records = list(records)
        for record in records:
            _record(record, training=True)
        names = self.checks or sorted({k for r in records for k in r["checks"]})
        cnt = {k: {"right": {"agree": 0, "fail": 0, "dis": 0}, "wrong": {o: 0 for o in OUT}} for k in names}
        coll = [0, 0]
        for r in records:
            ok = r["correct"]
            vals = []
            for k in names:
                c = r["checks"].get(k)
                if c is None:
                    continue
                st = c["status"]
                if ok:
                    cnt[k]["right"]["dis" if st == "disagree" else st] += 1
                else:
                    o = st if st != "disagree" else ("truth" if c["correct"] else "other")
                    cnt[k]["wrong"][o] += 1
                    if o == "other":
                        vals.append(c["value"])
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    coll[0] += close(vals[i], vals[j])
                    coll[1] += 1
        self.counts = cnt
        pooled = {}
        for cls, outs in (("right", ("agree", "fail", "dis")), ("wrong", OUT)):
            tot = {o: sum(cnt[k][cls][o] for k in names) for o in outs}
            n = sum(tot.values())
            pooled[cls] = {o: (tot[o] + self.alpha) / (n + self.alpha * len(outs)) for o in outs}
        self.pooled = pooled
        self.params = {k: self._params(cnt[k]) for k in names}
        self.collision = (coll[0] + self.cprior[0]) / (coll[1] + sum(self.cprior))
        self.collision_counts = coll
        return self

    def _params(self, c):
        def post(cls, outs):
            n = sum(c[cls][o] for o in outs)
            return {o: (c[cls][o] + self.pool * self.pooled[cls][o]) / (n + self.pool) for o in outs}
        a = post("right", ("agree", "fail", "dis"))
        b = post("wrong", OUT)
        return CheckParams(s=a["agree"], phic=a["fail"], f=b["agree"], t=b["truth"], phiw=b["fail"])

    def get(self, k) -> CheckParams:
        if k not in self.params:                              # unseen check → pooled behaviour
            self.params[k] = self._params({"right": {"agree": 0, "fail": 0, "dis": 0}, "wrong": {o: 0 for o in OUT}})
        return self.params[k]


# ----------------------------------------------------------------------------------------------------- policy
@dataclass
class Action:
    kind: str                       # "answer" | "answer_check" | "abstain" | "check"
    check: str | None = None        # which check (for answer_check: whose value is answered; for check: what to buy)
    value: float | None = None      # expected utility of the action
    p_answer: float | None = None   # P(the answered candidate is right)
    why: str = ""


@dataclass
class Replay:
    answered: bool
    answer: float | None
    correct: bool
    calls: int
    used: list = field(default_factory=list)
    final: Action | None = None


class AnswerPolicy:
    """VOI policy over candidates. `checks`: names that may be bought (in order of preference on ties); `weights`:
    optional callable subset → {check: w} (e.g. CorrelationAwareCombiner.weights) tempering correlated evidence."""

    def __init__(self, model: CheckModel, checks=None, cost: float = 0.05, allow_replace: bool = True,
                 weights=None, max_checks: int | None = None, costs: dict | None = None, units: dict | None = None,
                 max_units: float | None = None):
        if checks is None:
            checks = tuple(costs) if costs else ("P", "D")
        self.m, self.checks, self.cost = model, _names(checks), cost
        self.allow_replace, self.weights, self.max_checks = allow_replace, weights, max_checks
        self.costs = dict(costs or {})
        self.units = dict(units or {})
        self.max_units = max_units
        self._validate({})

    def _validate(self, state, q0=None, lam=None):
        if q0 is not None:
            _number(q0, "q0", probability=True)
        if lam is not None:
            _number(lam, "lambda")
        _number(self.cost, "cost")
        for mapping in (self.costs, self.units):
            if not isinstance(mapping, dict) or any(k not in self.checks for k in mapping):
                raise ValueError("prices/units must name configured checks")
            for value in mapping.values():
                _number(value, "price/units")
        if self.max_units is not None:
            _number(self.max_units, "max_units")
        if self.max_checks is not None and (isinstance(self.max_checks, bool) or not isinstance(self.max_checks, int) or self.max_checks < 0):
            raise ValueError("max_checks must be a nonnegative integer")
        if not isinstance(state, dict) or any(k not in self.checks for k in state):
            raise ValueError("state must name configured checks")
        for o in state.values():
            if not (isinstance(o, str) and o in ("agree", "fail")) and not (
                    isinstance(o, tuple) and len(o) == 2 and o[0] == "val" and isinstance(o[1], str) and o[1]):
                raise ValueError("invalid check outcome")
        if self.max_checks is not None and len(state) > self.max_checks:
            raise ValueError("state exceeds max_checks")
        if self.max_units is not None and self.spent(state) > self.max_units + 1e-9:
            raise ValueError("state exceeds max_units")
        if not hasattr(self.m, "params"):
            raise ValueError("fit CheckModel before use")
        _number(self.m.collision, "collision", positive=True, probability=True)
        for p in self.m.params.values():
            for x in (p.s, p.phic, p.f, p.t, p.phiw):
                _number(x, "check probability", probability=True)
            if p.s <= 0 or p.s + p.phic > 1 or p.f + p.t + p.phiw > 1:
                raise ValueError("inconsistent check probabilities")

    def price(self, k) -> float:
        """Utility price of buying check k (S138: one price per check, `cost` for the rest)."""
        return float(self.costs.get(k, self.cost))

    def unit(self, k) -> float:
        return float(self.units.get(k, 1.0))

    def spent(self, state) -> float:
        return sum(self.unit(k) for k in state)

    # ---- posterior over hypotheses
    def posterior(self, q0: float, state: dict):
        """state: {check: "agree" | "fail" | ("val", value_id)} → (P(a0), {value_id: P(value)}, P(unseen))."""
        _number(q0, "q0", probability=True)
        self._validate(state, q0)
        q0 = min(max(q0, 1e-9), 1 - 1e-9)
        obs = list(state)
        w = self.weights(obs) if (self.weights and obs) else {k: 1.0 for k in obs}
        if not isinstance(w, dict) or any(k not in obs for k in w):
            raise ValueError("weights must map observed checks")
        for value in w.values():
            _number(value, "weight")
        groups = {}
        for k, o in state.items():
            if isinstance(o, tuple):
                groups.setdefault(o[1], []).append(k)
        c = self.m.collision
        la0 = math.log(q0)
        lw = math.log(1 - q0)
        base_wrong = lw
        for k, o in state.items():
            p = self.m.get(k)
            wk = w.get(k, 1.0)
            if o == "agree":
                la0 += wk * math.log(p.s)
                base_wrong += wk * math.log(max(p.f, 1e-12))
            elif o == "fail":
                la0 += wk * math.log(max(p.phic, 1e-12))
                base_wrong += wk * math.log(max(p.phiw, 1e-12))
            else:
                la0 += wk * math.log(p.dis_c)
        for g, ks in groups.items():
            la0 += (len(ks) - 1) * math.log(c)
        # wrong hypotheses: start from base_wrong, add the value-returning checks
        lother = base_wrong
        for g, ks in groups.items():
            lother += sum(w.get(k, 1.0) * math.log(self.m.get(k).r) for k in ks) + (len(ks) - 1) * math.log(c)
        lv = {}
        for g, ks in groups.items():
            log_weight = base_wrong + sum(w.get(k, 1.0) * math.log(max(self.m.get(k).t, 1e-12)) for k in ks)
            for g2, ks2 in groups.items():
                if g2 != g:
                    log_weight += sum(w.get(k, 1.0) * math.log(self.m.get(k).r) for k in ks2) + (len(ks2) - 1) * math.log(c)
            lv[g] = log_weight
        allv = [la0, lother] + list(lv.values())
        mx = max(allv)
        z = sum(math.exp(x - mx) for x in allv)
        return (math.exp(la0 - mx) / z, {g: math.exp(weight - mx) / z for g, weight in lv.items()}, math.exp(lother - mx) / z)

    def _predictive(self, k, state, post):
        """Outcomes of buying k at `state`: [(prob, outcome)] with outcome "agree" | "fail" | ("val", id)."""
        pa0, pv, pun = post
        p = self.m.get(k)
        c = self.m.collision
        ids = list(pv)
        m = len(ids)
        new_id = f"new{len(state)}:{k}"
        while new_id in pv:
            new_id += ":"
        new = ("val", new_id)
        out = {"agree": pa0 * p.s + (1 - pa0) * p.f, "fail": pa0 * p.phic + (1 - pa0) * p.phiw}
        pnew = pa0 * p.dis_c * max(0.0, 1 - c * m) + pun * (p.t + p.r * max(0.0, 1 - c * m))
        for g in ids:
            pe = pa0 * p.dis_c * c + pv[g] * p.t + sum(pv[h] * p.r * c for h in ids if h != g) + pun * p.r * c
            out[("val", g)] = pe
            pnew += pv[g] * p.r * max(0.0, 1 - c * (m - 1))
        out[new] = pnew
        tot = sum(out.values())
        return [(v / tot, o) for o, v in out.items() if v > 0]

    # ---- decision
    def _answer_options(self, post, state, lam):
        pa0, pv, _ = post
        opts = [Action("answer", None, pa0 - lam * (1 - pa0), pa0)]
        if self.allow_replace:
            for g, pg in pv.items():
                k = next(kk for kk, o in state.items() if o == ("val", g))
                opts.append(Action("answer_check", k, pg - lam * (1 - pg), pg))
        return opts

    def _value(self, q0, state, lam, memo):
        # Preserve outcome type: candidate IDs may themselves be 'agree'/'fail'.
        key = tuple(sorted(state.items()))
        if key in memo:
            return memo[key]
        post = self.posterior(q0, state)
        best = Action("abstain", None, 0.0, None)
        for a in self._answer_options(post, state, lam):
            if a.value > best.value + 1e-12:
                best = a
        can_buy = self.max_checks is None or len(state) < self.max_checks
        spent = self.spent(state) if self.max_units is not None else 0.0
        for k in self.checks:
            if k in state or not can_buy:
                continue
            if self.max_units is not None and spent + self.unit(k) > self.max_units + 1e-9:
                continue                                  # over the per-item cap (budget mode)
            v = -self.price(k)
            for pr, o in self._predictive(k, state, post):
                v += pr * self._value(q0, {**state, k: o}, lam, memo).value
            if v > best.value + 1e-12:
                best = Action("check", k, v, None)
        memo[key] = best
        return best

    def decide(self, q0: float, state: dict, lam: float) -> Action:
        _number(q0, "q0", probability=True)
        _number(lam, "lambda")
        self._validate(state, q0, lam)
        # Public configuration/model/weight callbacks may change between decisions.
        # Keep one tree per decision; recursion still shares the original DP memo.
        a = self._value(q0, dict(state), lam, {})
        a = Action(a.kind, a.check, a.value, a.p_answer)  # the memo keeps its own copy
        pa0, pv, pun = self.posterior(q0, state)
        a.why = (f"P(original)={pa0:.3f}" + "".join(f", P({g})={p:.3f}" for g, p in pv.items()) +
                 f", P(unseen)={pun:.3f}; λ={lam}: {a.kind}" + (f" {a.check}" if a.check else ""))
        return a

    # ---- offline replay on a stored record
    def replay(self, q0: float, record: dict, lam: float, gold=None) -> Replay:
        _number(q0, "q0", probability=True)
        _number(lam, "lambda")
        self._validate({}, q0, lam)
        _record(record, self.checks)
        gold = record.get("gold") if gold is None else gold
        if gold is not None:
            _finite_answer(gold)
        state, values, used = {}, {}, []
        for _ in range(len(self.checks) + 1):
            a = self.decide(q0, state, lam)
            if a.kind == "check":
                c = record["checks"].get(a.check)
                if c is None:
                    raise KeyError(f"record has no stored value for check {a.check!r}")
                used.append(a.check)
                if c["status"] in ("agree", "fail"):
                    state[a.check] = c["status"]
                else:
                    gid = next((g for g, v in values.items() if close(v, c["value"])), None)
                    if gid is None:
                        gid = f"v{len(values)}"
                        values[gid] = c["value"]
                    state[a.check] = ("val", gid)
                continue
            if a.kind == "abstain":
                return Replay(False, None, False, 1 + len(used), used, a)
            ans = record["answer"] if a.kind == "answer" else values[state[a.check][1]]
            ok = record["correct"] if a.kind == "answer" else close(ans, gold)
            return Replay(True, ans, bool(ok), 1 + len(used), used, a)
        raise RuntimeError("policy did not stop")


# ----------------------------------------------------------------------------------------------------- budget mode
class BudgetPacer:
    """Mean spending on checks ≤ `budget` units per item over a stream (S140).

    budget  per-item budget b for CHECK units (the always-paid primary answer is not counted here)
    kappa   the base utility price per unit (the pre-registered κ); the policy's price is (κ + μ_t)·units
    eta     step of the dual price μ (utility per unit per unit of overspend); default κ / b
    slack   extra units the hard allowance may borrow (0 → mean ≤ b exactly; the first items may get less)

    configure(pol, unit_costs) sets pol.costs / pol.units / pol.max_units for the next item; charge(units_spent)
    (or charge(replay) — sums pol.units over replay.used) books it. mean() is the running mean spending."""

    def __init__(self, budget: float, kappa: float, eta: float | None = None, slack: float = 0.0,
                 mu0: float = 0.0):
        for value, name in ((budget, "budget"), (kappa, "kappa"), (slack, "slack"), (mu0, "mu0")):
            _number(value, name)
        if eta is not None:
            _number(eta, "eta")
        self.b, self.kappa, self.slack = float(budget), float(kappa), float(slack)
        self.eta = (kappa / budget if budget > 0 else kappa) if eta is None else float(eta)
        self.mu, self.spent, self.t = float(mu0), 0.0, 0
        self._units = {}
        self.history = []

    def allowance(self) -> float:
        """Units the next item may spend: b·(t+1) + slack − spent so far (never negative)."""
        return max(0.0, self.b * (self.t + 1) + self.slack - self.spent)

    def configure(self, pol: "AnswerPolicy", unit_costs: dict) -> "AnswerPolicy":
        if any(k not in pol.checks for k in unit_costs):
            raise ValueError("units must name configured checks")
        for value in unit_costs.values():
            _number(value, "units")
        self._units = dict(unit_costs)
        pol.units = dict(unit_costs)
        pol.costs = {k: (self.kappa + self.mu) * u for k, u in unit_costs.items()}
        pol.max_units = self.allowance()
        return pol

    def charge(self, spent) -> float:
        if isinstance(spent, Replay):
            spent = sum(self._units.get(k, 1.0) for k in spent.used)
        spent = _number(spent, "spent")
        if spent > self.allowance() + 1e-6:
            raise ValueError("budget pacer overspent: the policy ignored max_units")
        self.spent += spent
        self.t += 1
        self.mu = max(0.0, self.mu + self.eta * (spent - self.b))
        self.history.append((spent, self.mu))
        if self.spent > self.b * self.t + self.slack + 1e-6:
            raise AssertionError("budget pacer overspent: the policy ignored max_units")
        return self.mu

    def mean(self) -> float:
        return self.spent / self.t if self.t else 0.0
