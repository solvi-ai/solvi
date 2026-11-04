"""Provenance and descriptive pooled check models. Source SHA256: 161cb7c69fb07f21f019ff3f962857ce8b1c93028b1c93e51cb23d7578ccc273."""
from __future__ import annotations

import math
from dataclasses import dataclass




__all__ = ["Check", "CheckRegistry", "error_correlation", "NaiveCombiner", "CorrelationAwareCombiner"]


def _number(value, name, low=None, high=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (positive and value <= 0) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} outside domain")
    return float(value)


def _logadd(a, b):
    """Log(exp(a) + exp(b)), including zero counts represented by -inf."""
    hi, lo = max(a, b), min(a, b)
    return hi if lo == -math.inf else hi + math.log1p(math.exp(lo - hi))


def _logcount(n):
    return math.log(n) if n else -math.inf


def _names(values):
    values = list(values)
    if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError("names must be distinct nonempty strings")
    return values


def _label(value):
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError("targets must be bool or integer 0/1")
    return int(value)


def _observation(row, names, outcomes):
    if not isinstance(row, dict) or any(k not in names for k in row):
        raise ValueError("observation must map configured names")
    if any(v is not None and v not in outcomes for v in row.values()):
        raise ValueError("unsupported outcome")


# ----------------------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Check:
    name: str
    family: str                       # model family / data source / "human" / "code"
    method: str                       # how the check computes (program, re-derivation, expression, lookup, …)
    shares_call_with: str | None = None   # name of a source whose model call this check re-reads (S90)
    cost: float = 1.0                 # in calls (or any unit the decision uses)
    note: str = ""

    def __post_init__(self):
        _names([self.name])
        for value in (self.family, self.method):
            if not isinstance(value, str) or not value:
                raise ValueError("family and method must be nonempty strings")
        if self.shares_call_with is not None:
            _names([self.shares_call_with])
        _number(self.cost, "cost", low=0)
        if not isinstance(self.note, str):
            raise ValueError("note must be a string")


class CheckRegistry:
    """Checks with provenance. Prior error correlation between two sources (used to shrink data estimates):
    same call → rho_call; same family and method → rho_method; same family → rho_family; else rho_other."""

    def __init__(self, primary: Check, checks=(), rho_call: float = 0.9, rho_method: float = 0.6,
                 rho_family: float = 0.35, rho_other: float = 0.05):
        if not isinstance(primary, Check) or primary.shares_call_with is not None:
            raise ValueError("primary must be a root Check")
        for value in (rho_call, rho_method, rho_family, rho_other):
            _number(value, "rho", low=-1, high=1)
        self.primary = primary
        self.checks: dict[str, Check] = {}
        self.rho = {"call": rho_call, "method": rho_method, "family": rho_family, "other": rho_other}
        for c in checks:
            self.add(c)

    def add(self, check: Check) -> "CheckRegistry":
        if not isinstance(check, Check):
            raise ValueError("expected Check")
        if check.name in self.checks or check.name == self.primary.name:
            raise ValueError(f"duplicate check name {check.name!r}")
        if check.shares_call_with and check.shares_call_with not in self.checks and \
                check.shares_call_with != self.primary.name:
            raise ValueError(f"{check.name!r} shares a call with unknown source {check.shares_call_with!r}")
        self.checks[check.name] = check
        return self

    def source(self, name) -> Check:
        return self.primary if name == self.primary.name else self.checks[name]

    def independent(self) -> list[str]:
        """Checks that do not re-read the primary's call (directly or through another check)."""
        out = []
        for n in self.checks:
            if self._root(n) != self.primary.name:
                out.append(n)
        return out

    def _root(self, name):
        seen = set()
        while True:
            s = self.source(name)
            if not s.shares_call_with or name in seen:
                return name
            seen.add(name)
            name = s.shares_call_with

    def prior_rho(self, a: str, b: str) -> float:
        self.source(a), self.source(b)
        if a == b:
            return 1.0
        sa, sb = self.source(a), self.source(b)
        if self._root(a) == self._root(b):
            return self.rho["call"]
        if sa.family == sb.family:
            return self.rho["method"] if sa.method == sb.method else self.rho["family"]
        return self.rho["other"]

    def warnings(self, names=None) -> list[str]:
        names = list(self.checks) if names is None else _names(names)
        w = []
        for n in names:
            if self._root(n) == self.primary.name:
                w.append(f"{n!r} shares the call with {self.primary.name!r}: its agreement is not independent "
                         f"evidence (S90)")
        ind = [n for n in names if self._root(n) != self.primary.name]
        fams = {self.source(n).family for n in ind}
        if ind and fams == {self.primary.family}:
            w.append(f"all independent checks ({', '.join(ind)}) are family {self.primary.family!r}, like the answer: "
                     f"expect correlated errors, e.g. the same misreading of the task (S104)")
        for i, a in enumerate(ind):
            for b in ind[i + 1:]:
                if self._root(a) == self._root(b):
                    w.append(f"{a!r} and {b!r} share one call")
        return w

    def n_effective(self, names, rho=None) -> float:
        """Effective number of independent checks 1ᵀR⁻¹1 under the prior (or given) correlation."""
        import numpy as np
        names = _names(names)
        for name in names:
            self.source(name)
        if not names:
            return 0.0
        R = _matrix(names, self.prior_rho if rho is None else rho)
        return float(np.ones(len(names)) @ np.linalg.solve(R, np.ones(len(names))))


def _matrix(names, rho):
    import numpy as np
    names = _names(names)
    if not callable(rho):
        if not isinstance(rho, dict):
            raise ValueError("rho must be a pair mapping or callable")
        for pair, value in rho.items():
            if not isinstance(pair, tuple) or len(pair) != 2 or any(k not in names for k in pair):
                # Full configured matrices may include sources outside the observed subset.
                if not isinstance(pair, tuple) or len(pair) != 2 or any(not isinstance(k, str) or not k for k in pair):
                    raise ValueError("rho keys must be source pairs")
            _number(value, "rho", low=-1, high=1)
            if pair[0] == pair[1] and value != 1:
                raise ValueError("rho diagonal must equal one")
            if (pair[1], pair[0]) in rho and rho[(pair[1], pair[0])] != value:
                raise ValueError("rho must be symmetric")
    if callable(rho):
        R = np.array([[rho(a, b) for b in names] for a in names], float)
    else:
        R = np.array([[1.0 if a == b else rho.get((a, b), rho.get((b, a), 0.0)) for b in names] for a in names], float)
    if R.shape != (len(names), len(names)) or not np.isfinite(R).all() or (np.abs(R) > 1).any():
        raise ValueError("invalid correlation matrix")
    if not np.allclose(R, R.T, rtol=0, atol=1e-12) or not np.allclose(np.diag(R), 1, rtol=0, atol=1e-12):
        raise ValueError("rho must be symmetric with unit diagonal")
    return _make_pd(R)


def _make_pd(R, floor: float = 0.05):
    import numpy as np
    R = (R + R.T) / 2
    np.fill_diagonal(R, 1.0)
    ev = np.linalg.eigvalsh(R).min() if len(R) else 1.0
    if ev < floor:                                        # blend with the identity until positive definite
        t = (floor - ev) / (1 - ev + 1e-12)
        R = (1 - t) * R + t * np.eye(len(R))
    return R


# ----------------------------------------------------------------------------------------------------- correlation
def error_correlation(rows, sources, prior=None, strength: float = 5.0):
    """Pairwise error correlation of sources from data.

    rows: iterable of {source: True (its value is right) | False (wrong) | None (failed / not observed)}.
    For each pair, phi = corr(err_a, err_b) over rows where both are observed, shrunk towards prior(a, b) with weight
    strength / (strength + n_inf), n_inf = rows where at least one of the two errs (only those carry information when
    errors are rare). Returns {"rho": {(a, b): ρ}, "raw": {(a, b): {...counts, phi, P(err_b | err_a)}}}."""
    import numpy as np
    rows, sources = list(rows), _names(sources)
    _number(strength, "strength", low=0)
    for row in rows:
        if not isinstance(row, dict) or any(k not in sources for k in row):
            raise ValueError("rows must map source names")
        for value in row.values():
            if value is not None:
                _label(value)
    prior = prior or (lambda a, b: 0.0)
    rho, raw = {}, {}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            both = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
            n = len(both)
            ea = np.array([0.0 if x else 1.0 for x, _ in both])
            eb = np.array([0.0 if y else 1.0 for _, y in both])
            n_inf = int(((ea + eb) > 0).sum())
            if n and ea.std() > 0 and eb.std() > 0:
                phi = float(np.corrcoef(ea, eb)[0, 1])
            else:
                phi = float("nan")
            p0 = _number(prior(a, b), "prior rho", low=-1, high=1)
            r_ = p0 if math.isnan(phi) else (n_inf * phi + strength * p0) / (n_inf + strength)
            rho[(a, b)] = r_
            raw[(a, b)] = {"n": n, "err_a": int(ea.sum()), "err_b": int(eb.sum()), "both_err": int((ea * eb).sum()),
                           "n_inf": n_inf, "phi": phi, "prior": p0, "rho": r_,
                           "p_err_b_given_err_a": float((ea * eb).sum() / ea.sum()) if ea.sum() else float("nan"),
                           "p_err_b": float(eb.mean()) if n else float("nan")}
    return {"rho": rho, "raw": raw}


# ----------------------------------------------------------------------------------------------------- combiners
def _logit(q):
    q = min(max(q, 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def _sig(z):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


class NaiveCombiner:
    """P(o_k | class) per source: Dirichlet posterior with the pooled (over sources) class distribution as prior of
    total weight `pool` pseudo-counts, itself smoothed by `alpha` per outcome. log LR = Σ observed sources."""

    def __init__(self, sources, outcomes=("agree", "disagree", "fail"), pool: float = 2.0, alpha: float = 0.5):
        self.sources, self.outcomes = _names(sources), tuple(_names(outcomes))
        if not self.sources or not self.outcomes:
            raise ValueError("sources and outcomes must be nonempty")
        self.pool = _number(pool, "pool", positive=True)
        self.alpha = _number(alpha, "alpha", positive=True)

    def fit(self, obs, y):
        if getattr(self, "_readonly", False):
            raise ValueError("a frozen check model cannot be refitted")
        # Public configuration may have changed since construction.
        self.sources = _names(self.sources)
        self.outcomes = tuple(_names(self.outcomes))
        if not self.sources or not self.outcomes:
            raise ValueError("sources and outcomes must be nonempty")
        pool = _number(self.pool, "pool", positive=True)
        alpha = _number(self.alpha, "alpha", positive=True)
        logpool, logalpha = math.log(pool), math.log(alpha)
        obs, y = list(obs), [_label(v) for v in y]
        if len(obs) != len(y):
            raise ValueError("obs and targets must have equal lengths")
        for row in obs:
            _observation(row, self.sources, self.outcomes)
        self.counts = {s: {c: {o: 0 for o in self.outcomes} for c in (0, 1)} for s in self.sources}
        for o, t in zip(obs, y):
            for s in self.sources:
                if o.get(s) in self.outcomes:
                    self.counts[s][t][o[s]] += 1
        self.lik, self._log_lik = {}, {}
        for c in (0, 1):
            tot = {o: sum(self.counts[s][c][o] for s in self.sources) for o in self.outcomes}
            n = sum(tot.values())
            logden = _logadd(_logcount(n), logalpha + math.log(len(self.outcomes)))
            logpooled = {o: _logadd(_logcount(tot[o]), logalpha) - logden for o in self.outcomes}
            for s in self.sources:
                ns = sum(self.counts[s][c].values())
                logden_s = _logadd(_logcount(ns), logpool)
                logs = {o: _logadd(_logcount(self.counts[s][c][o]), logpool + logpooled[o]) - logden_s
                        for o in self.outcomes}
                self._log_lik.setdefault(s, {})[c] = logs
                self.lik.setdefault(s, {})[c] = {o: math.exp(value) for o, value in logs.items()}
        self.prior_rate = (sum(y) + 0.5) / (len(y) + 1.0)
        return self

    def llr(self, s, outcome) -> float:
        if not hasattr(self, "lik"):
            raise ValueError("fit before prediction")
        if s not in self.sources or outcome not in self.outcomes:
            raise ValueError("unknown source or outcome")
        return self._log_lik[s][1][outcome] - self._log_lik[s][0][outcome]

    def observed(self, o):
        _observation(o, self.sources, self.outcomes)
        return [s for s in self.sources if o.get(s) in self.outcomes]

    def log_lr(self, o) -> float:
        return sum(self.llr(s, o[s]) for s in self.observed(o))

    def posterior(self, prior, o) -> float:
        _number(prior, "prior", low=0, high=1)
        return _sig(_logit(prior) + self.log_lr(o))

    def freeze(self):
        """Detached fitted prediction snapshot; refitting is disabled.

        Correlation callbacks are resolved to a fixed matrix now. Public configuration
        of the returned snapshot must not be mutated; fingerprint describes its
        prediction state. This does not certify calibration or independence.
        """
        from copy import deepcopy

        if not hasattr(self, "_log_lik"):
            raise ValueError("fit before freezing a check model")
        result = deepcopy(self)
        result.sources = tuple(_names(self.sources))
        result.outcomes = tuple(_names(self.outcomes))
        for source in result.sources:
            for outcome in result.outcomes:
                if not math.isfinite(self.llr(source, outcome)):
                    raise ValueError("fitted log likelihood ratios must be finite")
        if isinstance(self, CorrelationAwareCombiner):
            def matrix(rho):
                values = ({(a, b): float(rho(a, b)) for a in result.sources for b in result.sources}
                          if callable(rho) else dict(rho))
                if callable(rho):
                    _matrix(result.sources, lambda a, b: values[a, b])
                    # Match the native callback path's symmetry tolerance and pre-projection averaging.
                    values = {(a, b): 1.0 if a == b else (values[a, b] + values[b, a]) / 2
                              for a in result.sources for b in result.sources}
                else:
                    _matrix(result.sources, values)
                return values
            result.rho = matrix(self.rho)
            result.rho_by_domain = {domain: matrix(rho) for domain, rho in self.rho_by_domain.items()}
        result._readonly = True
        return result

    def fingerprint(self):
        """SHA256 of fitted prediction state, with callbacks snapshotted."""
        import hashlib
        import json

        model = self.freeze()
        payload = {"version": 1, "type": type(model).__name__, "sources": model.sources,
                   "outcomes": model.outcomes, "log_likelihoods": model._log_lik,
                   "counts": model.counts, "pool": model.pool, "alpha": model.alpha,
                   "prior_rate": model.prior_rate}
        if isinstance(model, CorrelationAwareCombiner):
            def rows(matrix):
                return sorted([a, b, value] for (a, b), value in matrix.items())
            payload.update(rho=rows(model.rho),
                           rho_by_domain={domain: rows(rho) for domain, rho in model.rho_by_domain.items()},
                           domain_counts=model.domain_counts, domain_shrinkage=model.domain_shrinkage)
        raw = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


class CorrelationAwareCombiner(NaiveCombiner):
    """Weighted composite likelihood with weights from the error-correlation matrix of the OBSERVED checks.

    rho: {(a, b): ρ} (e.g. error_correlation(...)["rho"]) or a callable (a, b) → ρ (e.g. Registry.prior_rho)."""

    def __init__(self, sources, rho, outcomes=("agree", "disagree", "fail"), pool: float = 2.0, alpha: float = 0.5,
                 *, rho_by_domain=None, domain_counts=None, domain_shrinkage=0.0):
        super().__init__(sources, outcomes, pool, alpha)
        self.rho = rho
        if rho_by_domain is not None and not isinstance(rho_by_domain, dict):
            raise ValueError("rho_by_domain must map nonempty domain names to correlation mappings")
        self.rho_by_domain = {}
        for domain, matrix in (rho_by_domain or {}).items():
            if not isinstance(domain, str) or not domain or not isinstance(matrix, dict):
                raise ValueError("rho_by_domain must map nonempty domain names to correlation mappings")
            _matrix(self.sources, matrix)
            self.rho_by_domain[domain] = dict(matrix)
        self.domain_shrinkage = _number(domain_shrinkage, "domain_shrinkage", low=0)
        if domain_counts is not None and not isinstance(domain_counts, dict):
            raise ValueError("domain_counts must map configured domains to nonnegative integer counts")
        self.domain_counts = dict(domain_counts or {})
        if any(k not in self.rho_by_domain or type(n) is not int or n < 0
               for k, n in self.domain_counts.items()):
            raise ValueError("domain_counts must map configured domains to nonnegative integer counts")
        if self.domain_shrinkage and set(self.domain_counts) != set(self.rho_by_domain):
            raise ValueError("positive domain_shrinkage requires a count for every configured domain")

    def _domain_rho(self, domain):
        if domain is None:
            if self.rho_by_domain:
                import warnings
                warnings.warn("no correlation domain selected: using global rho, not domain-conditioned evidence",
                              UserWarning, stacklevel=3)
            return self.rho
        if not isinstance(domain, str) or domain not in self.rho_by_domain:
            raise ValueError("unknown correlation domain; pass domain=None explicitly to use global rho")
        matrix = self.rho_by_domain[domain]
        if not self.domain_shrinkage:
            return matrix
        n = self.domain_counts[domain]
        if n < 2:
            raise ValueError("domain correlation needs at least two supplied observations; use global domain=None")
        # Stable convex weight, including extremely strong but finite shrinkage.
        log_ratio = math.log(n) - math.log(self.domain_shrinkage)
        ratio = math.exp(-abs(log_ratio))
        weight = 1.0 / (1.0 + ratio) if log_ratio >= 0 else ratio / (1.0 + ratio)

        def shrunk(a, b):
            if a == b:
                return 1.0
            local = matrix.get((a, b), matrix.get((b, a), 0.0))
            global_rho = self.rho(a, b) if callable(self.rho) else self.rho.get((a, b), self.rho.get((b, a), 0.0))
            return weight * local + (1.0 - weight) * global_rho

        return shrunk

    def weights(self, subset, *, domain=None) -> dict:
        import numpy as np
        key = sorted(_names(subset))
        if any(k not in self.sources for k in key):
            raise ValueError("unknown source")
        rho = self._domain_rho(domain)
        if not key:
            return {}
        R = _matrix(key, rho)
        w = np.clip(np.linalg.solve(R, np.ones(len(key))), 0.0, 1.0)
        return dict(zip(key, w.tolist()))

    def n_effective(self, subset, *, domain=None) -> float:
        import numpy as np
        subset = sorted(_names(subset))
        if any(k not in self.sources for k in subset):
            raise ValueError("unknown source")
        rho = self._domain_rho(domain)
        if not subset:
            return 0.0
        R = _matrix(subset, rho)
        return float(np.ones(len(subset)) @ np.linalg.solve(R, np.ones(len(subset))))

    def log_lr(self, o, *, domain=None) -> float:
        obs = self.observed(o)
        w = self.weights(obs, domain=domain)
        return sum(w[s] * self.llr(s, o[s]) for s in obs)

    def posterior(self, prior, o, *, domain=None) -> float:
        _number(prior, "prior", low=0, high=1)
        return _sig(_logit(prior) + self.log_lr(o, domain=domain))
