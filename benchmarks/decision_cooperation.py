"""Frozen finite cooperative-search benchmark helper (no evaluation at import).

Three authored families, selected by seed % 3: resources, routes and configs.
These are finite synthetic cases, not an out-of-distribution or field claim.
Fixed parameters: 9 + seed % 5 alternatives, two constraints, one pass through
at most two ready solvers. Cooperative and RR order constraints then objective;
independent portfolio orders objective first (a deliberately strong portfolio).
Each arm stops on its first verified feasible incumbent. No parameter fitting.

Tariff: one fixture setup unit plus one input unit per assignment; production
FiniteProblem charges full semantic materialization, all predicate/cost lookups,
residual construction, accepted/rejected evidence checking and certificate work.
The benchmark adds one dispatch unit per called solver and one control unit per
exhaustive alternative. All states in an arm share ONE WorkBudget, including
independent portfolio states. Fixture generation and the independent scorer are
paid in every arm and never supply their reference answer to a solver.
Final exhaustive certification is paid by every arm; exhaustion leaves a verified
feasible incumbent with unknown optimality. A completed exhaustive baseline does
not repeat its own enumeration but still pays an independent candidate check.
Work units are declared operations, not equal wall time. Scorer arithmetic is
independent of production finite_reference and never supplies an answer to arms.
"""
from __future__ import annotations

from random import Random

from solvi.experimental.experience._common import BudgetStop, WorkBudget, natural, portable
from solvi.experimental.experience.cooperation import (
    ConstraintSolver, CooperativeSearch, Evidence, FiniteProblem, ObjectiveSolver,
)


ARMS = ("cooperative", "portfolio", "round_robin", "exhaustive")
FAMILIES = ("resources", "routes", "configs")


def _fixture(seed, budget):
    rng = Random(seed)
    family = seed % 3
    assignments = {}
    for i in range(9 + seed % 5):
        budget.charge("benchmark.input", 5)
        assignments[f"a{i:02d}"] = {
            "capacity": 1 if i == 0 else rng.randrange(4, 14),
            "latency": rng.randrange(1, 12),
            "reliability": rng.randrange(5, 11),
            "price": rng.randrange(1, 20),
        }
    # A known feasible anchor prevents accidentally measuring only infeasibility.
    assignments["a01"] = {"capacity": 12, "latency": 2, "reliability": 10, "price": 18}
    return family, assignments


def _host(family):
    if family == 0:
        return ({"capacity": lambda a: a["capacity"] >= 6,
                 "latency": lambda a: a["latency"] <= 8},
                lambda a: 3 * a["price"] + a["latency"])
    if family == 1:
        return ({"capacity": lambda a: a["capacity"] >= 4,
                 "reliability": lambda a: a["reliability"] >= 8},
                lambda a: 2 * a["latency"] + a["price"])
    return ({"capacity": lambda a: a["capacity"] >= 7,
             "reliability": lambda a: a["reliability"] >= 7},
            lambda a: a["price"] + 2 * (10 - a["reliability"]) + a["latency"])


def _score_raw(family, assignments, selected, budget):
    """Independent scalar arithmetic; no production callbacks or reference oracle."""
    eligible = {}
    for key, attrs in assignments.items():
        budget.charge("benchmark.independent_score", 8)
        capacity, latency = attrs["capacity"], attrs["latency"]
        reliability, price = attrs["reliability"], attrs["price"]
        if family == 0:
            allowed, cost = capacity >= 6 and latency <= 8, price * 3 + latency
        elif family == 1:
            allowed, cost = capacity >= 4 and reliability >= 8, latency * 2 + price
        else:
            allowed = capacity >= 7 and reliability >= 7
            cost = price + (10 - reliability) * 2 + latency
        if allowed:
            eligible[key] = cost
    budget.charge("benchmark.score_optimum", 1 + len(eligible))
    optimum = min((cost, key) for key, cost in eligible.items()) if eligible else None
    violation = int(selected is not None and selected not in eligible)
    regret = eligible[selected] - optimum[0] if selected in eligible and optimum else None
    return optimum, regret, violation


def run_trial(seed, arm, limit):
    """Return portable metrics for one fixed finite trial; never launch jobs."""
    natural(seed, "seed")
    natural(limit, "limit")
    if arm not in ARMS:
        raise ValueError("unknown cooperation benchmark arm")
    family, raw = seed % 3, {}
    budget = WorkBudget(limit)
    selected = None
    selected_cost = None
    status = "unknown"
    reason = "budget"
    searches = []
    exchange_effect = 0
    exhaustive_done = False
    try:
        budget.charge("benchmark.setup")
        family, raw = _fixture(seed, budget)
        constraints, objective = _host(family)
        problem = FiniteProblem(raw, constraints, objective, revision=f"finite-{family}-{seed}", budget=budget)
        if arm == "exhaustive":
            best = None
            for key in problem.ids:
                budget.charge("benchmark.exhaustive_control")
                if problem.feasible(key, budget):
                    pair = (problem.cost(key, budget), key)
                    best = pair if best is None else min(best, pair)
            exhaustive_done = True
            search = CooperativeSearch(problem, budget)
            searches.append(search)
            if best is not None:
                cost, key = best
                search.submit(Evidence("exhaustive-best", "exhaustive", problem.revision,
                                       "incumbent", (key,), cost))
                selected, selected_cost = key, cost
            status, reason = ("optimal" if best else "infeasible"), "complete"
        else:
            constraint_search = CooperativeSearch(problem, budget)
            objective_search = constraint_search if arm == "cooperative" else CooperativeSearch(problem, budget)
            searches.append(constraint_search)
            if objective_search is not constraint_search:
                searches.append(objective_search)
            constraint = ConstraintSolver(problem, "constraint")
            objective_solver = ObjectiveSolver(problem, "objective")
            sequence = [(constraint_search, constraint), (objective_search, objective_solver)]
            if arm == "portfolio":
                sequence.reverse()
            for search, solver in sequence:
                budget.charge("benchmark.dispatch")
                residual = search.residual()
                if solver is objective_solver:
                    exchange_effect = len(problem.ids) - len(residual.candidates)
                claim = solver(residual, budget)
                if claim is None:
                    continue
                search.submit(claim)
                if claim.kind == "incumbent":
                    selected, selected_cost = claim.targets[0], claim.value
                    break
            if selected is not None:
                status, reason = "feasible", "partial"
                certificate = objective_search.certify()
                if certificate["status"] == "optimal":
                    optimum_cost, _ = certificate["solution"]
                    status = "optimal" if optimum_cost == selected_cost else "feasible"
                    reason = "complete" if status == "optimal" else "partial"
                elif certificate["status"] == "unknown":
                    reason = "budget"
            else:
                reason = "partial"
    except BudgetStop:
        reason = "budget"
    # submit can commit independently checked evidence before its paid residual
    # return exhausts the budget. Recover only actual accepted journal entries.
    accepted = [event["accept"] for search in searches for event in search._events if "accept" in event]
    incumbents = [(e["value"], e["targets"][0]) for e in accepted if e["kind"] == "incumbent"]
    if incumbents:
        selected_cost, selected = min(incumbents)
        if status == "unknown":
            status = "feasible"
    try:
        budget.charge("benchmark.accepted_scan", 1 + len(accepted))
        optimum, regret, violations = _score_raw(family, raw, selected, budget)
    except BudgetStop:
        optimum = regret = violations = None
        status, reason = "unknown", "budget before independent scoring"
    return portable({
        "seed": seed, "arm": arm, "family": FAMILIES[family], "limit": limit,
        "assignments": raw, "alternatives": len(raw), "selected": selected, "selected_cost": selected_cost,
        "optimum": list(optimum) if optimum else None, "regret": regret,
        "solved": status != "unknown" and selected is not None and violations == 0,
        "hardviolations": violations, "status": status, "reason": reason,
        "work": budget.spent, "budget": budget.snapshot(),
        "accepted_evidence": len(accepted), "exchange_effect": exchange_effect,
        "exhaustive_completed": exhaustive_done,
    })
