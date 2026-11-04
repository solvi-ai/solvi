"""Finite conditional-table check acquisition; no calibration or FAR guarantee."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from .answer_policy import Action, _number, _names

__all__ = ["CheckPolicy"]


class CheckPolicy:
    """Bellman search over at most six checks with fixed utility prices.

    The search is exact for this finite surrogate: point scores mix fixed class
    likelihood tables to predict outcomes; native scores determine stop utility.
    With correlation tempering or a risk quantile this is an approximate VOI
    surrogate, not a coherent joint Bayesian model. Pooled priors, correlations,
    outcome tables and prices are fixed; no coverage or false-accept guarantee.
    Budgets include checks already present in the supplied outcomes.
    """

    def __init__(self, fitted_combiner, *, costs, max_checks=None, max_cost=None,
                 risk_quantile=None, n=1000, seed=0):
        from ...stat import NaiveCombiner, CorrelationAwareCombiner

        if type(fitted_combiner) not in (NaiveCombiner, CorrelationAwareCombiner):
            raise TypeError("CheckPolicy requires a fitted native check combiner")
        self.model = fitted_combiner.freeze()
        if set(self.model.outcomes) != {"agree", "disagree", "fail"}:
            raise ValueError("CheckPolicy requires exactly agree/disagree/fail model outcomes")
        if len(self.sources) > 6:
            raise ValueError("CheckPolicy finite search supports at most six sources")
        if not isinstance(costs, dict) or set(costs) != set(self.sources):
            raise ValueError("costs must contain exactly all model sources")
        self.costs = {s: _number(costs[s], "check cost") for s in self.sources}
        if max_checks is not None and (type(max_checks) is not int or max_checks < 0):
            raise ValueError("max_checks must be a nonnegative integer or None")
        self.max_checks = max_checks
        self.max_cost = None if max_cost is None else _number(max_cost, "max_cost")
        self.risk_quantile = None if risk_quantile is None else _number(
            risk_quantile, "risk_quantile", probability=True)
        if self.risk_quantile is not None and not hasattr(self.model, "_log_concentrations"):
            raise ValueError("refit a legacy check model before requesting uncertainty")
        if type(n) is not int or n <= 0 or type(seed) is not int or seed < 0:
            raise ValueError("n must be positive integer and seed nonnegative integer")
        self.n, self.seed = n, seed

    @property
    def sources(self):
        return tuple(self.model.sources)

    def freeze(self):
        return type(self)(self.model, costs=self.costs, max_checks=self.max_checks,
                          max_cost=self.max_cost, risk_quantile=self.risk_quantile,
                          n=self.n, seed=self.seed)

    def fingerprint(self):
        snapshot = self.freeze()
        payload = {"version": 1, "model": snapshot.model.fingerprint(), "costs": snapshot.costs,
                   "max_checks": snapshot.max_checks, "max_cost": snapshot.max_cost,
                   "risk_quantile": snapshot.risk_quantile, "n": snapshot.n, "seed": snapshot.seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode()).hexdigest()

    def decide(self, prior, outcomes, loss, *, available=None):
        """Choose stop or a next check; acquisition prices are charged once."""
        import numpy as np

        prior = _number(prior, "prior", probability=True)
        loss = _number(loss, "loss")
        # Revalidate public configuration on each decision; no stale search cache.
        policy = self.freeze()
        if not isinstance(outcomes, dict) or any(
                s not in policy.sources or o not in policy.model.outcomes for s, o in outcomes.items()):
            raise ValueError("outcomes must map model sources to declared nonmissing outcomes")
        names = policy.sources
        if available is None:
            available = names
        else:
            if isinstance(available, (str, bytes)):
                raise ValueError("available must be a subset of model sources")
            try:
                available = _names(available)
            except TypeError as exc:
                raise ValueError("available must be a subset of model sources") from exc
            if not set(available).issubset(names):
                raise ValueError("available must be a subset of model sources")
        initial = tuple(outcomes.get(s) for s in names)
        if policy.max_checks is not None and len(outcomes) > policy.max_checks:
            raise ValueError("observed checks exceed max_checks")
        if policy.max_cost is not None and sum(policy.costs[s] for s in outcomes) > policy.max_cost:
            raise ValueError("observed checks exceed max_cost")

        @lru_cache(None)
        def solve(state):
            observed = {s: o for s, o in zip(names, state) if o is not None}
            point = policy.model.posterior(prior, observed) if observed else prior
            score = point
            if policy.risk_quantile is not None and observed:
                score = float(np.quantile(policy.model.posterior_samples(
                    prior, observed, n=policy.n, seed=policy.seed), policy.risk_quantile))
            answer = score - loss * (1 - score)
            best = Action("answer" if answer > 0 else "abstain", value=max(0.0, answer),
                          p_answer=score, why="conditional model stop utility; fixed tables and prices")
            spent = sum(policy.costs[s] for s in observed)
            for i, source in enumerate(names):
                if state[i] is not None or source not in available:
                    continue
                if policy.max_checks is not None and len(observed) >= policy.max_checks:
                    continue
                price = policy.costs[source]
                if policy.max_cost is not None and spent + price > policy.max_cost:
                    continue
                value = -price
                for outcome in policy.model.outcomes:
                    probability = (point * policy.model.lik[source][1][outcome]
                                   + (1 - point) * policy.model.lik[source][0][outcome])
                    future = state[:i] + (outcome,) + state[i + 1:]
                    value += probability * solve(future).value
                if value > best.value:
                    best = Action("check", check=source, value=value, p_answer=score,
                                  why="finite conditional-table VOI surrogate; fixed correlations and prices")
            return best

        return solve(initial)
