"""Paid disagreement-per-price heuristic over a bounded host observation pool.

Only candidate predictions are inspected. No oracle or labels are accepted.
A bounded committee need not contain the true program; this heuristic promises
neither globally optimal information nor a correct prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ._common import BudgetStop, canonical, identifier, natural
from .programs import Grammar, Program


@dataclass(frozen=True, init=False)
class ObservationOption:
    name: str
    _inputs: str
    cost: int

    def __init__(self, name, inputs, cost):
        identifier(name, "observation name")
        natural(cost, "observation cost", positive=True)
        if type(inputs) is not dict or any(type(k) is not str or type(v) not in (int, bool) for k, v in inputs.items()):
            raise ValueError("strict scalar input dictionary required")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_inputs", canonical(inputs))
        object.__setattr__(self, "cost", cost)

    @property
    def inputs(self):
        return json.loads(self._inputs)


@dataclass(frozen=True)
class ObservationChoice:
    name: str | None
    reason: str
    pairs_separated: int


class _DebitFailure(Exception):
    def __init__(self, error):
        self.error = error


class _Paid:
    def __init__(self, budget):
        self.budget = budget

    def charge(self, kind, units=1):
        before = self.budget.spent
        try:
            self.budget.charge(kind, units)
        except Exception as error:  # noqa: BLE001 - preserve external debit failures exactly
            if isinstance(error, BudgetStop) and self.budget.spent == before:
                raise
            raise _DebitFailure(error) from error


def select_observation(programs, options, grammar, budget):
    """Choose the first maximal typed separated-pairs / declared read cost.

    Read prices are compared but never debited: this function does not read the
    environment. It charges planning, AST visits, outputs and pair comparisons.
    Invalid AST execution has one separate marker, never an invented label.
    At most 32 programs and 256 distinct named options are accepted; all reads
    and subsequent planning use the caller's existing shared budget.
    """
    paid = _Paid(budget)
    try:
        paid.charge("observation_plan")
        if type(grammar) is not Grammar:
            raise ValueError("Grammar required")
        committee = []
        for program in programs:
            paid.charge("observation_plan")
            if len(committee) >= 32 or type(program) is not Program:
                raise ValueError("at most 32 Program instances required")
            committee.append(program)
        pool, names = [], set()
        for option in options:
            paid.charge("observation_plan")
            if len(pool) >= 256 or type(option) is not ObservationOption or option.name in names:
                raise ValueError("at most 256 uniquely named observation options required")
            inputs = option.inputs
            paid.charge("observation_plan", len(inputs) + 1)
            if set(inputs) != set(grammar.inputs) or any(
                type(v) is int and abs(v) > grammar.max_abs for v in inputs.values()
            ):
                raise ValueError("observation inputs outside grammar schema or bounds")
            names.add(option.name)
            pool.append((option, inputs))
        scored = []
        for option, inputs in pool:
            outputs = []
            for program in committee:
                paid.charge("observation_plan")
                try:
                    value = program.evaluate(inputs, paid, grammar)
                    output = ("valid", canonical(value))
                except ValueError:
                    output = ("invalid",)
                paid.charge("observation_output")
                outputs.append(output)
            pairs = 0
            for i, left in enumerate(outputs):
                for right in outputs[i + 1 :]:
                    paid.charge("observation_compare")
                    pairs += left != right
            scored.append((option, pairs))
        # Price is checked against the final shared balance after all planning.
        paid.charge("observation_plan", len(scored))
        disagreement = any(pairs for _, pairs in scored)
        if not disagreement:
            return ObservationChoice(None, "no_disagreement", 0)
        # Select the best affordable option without further uncharged searches.
        affordable = [(option, pairs) for option, pairs in scored if pairs and option.cost <= budget.remaining]
        best, best_pairs = None, 0
        for option, pairs in affordable:
            if best is None or pairs * best.cost > best_pairs * option.cost:
                best, best_pairs = option, pairs
        if best is None:
            return ObservationChoice(None, "budget", 0)
        return ObservationChoice(best.name, "selected", best_pairs)
    except _DebitFailure as failure:
        raise failure.error from None
    except BudgetStop:
        return ObservationChoice(None, "budget", 0)


__all__ = ["ObservationOption", "ObservationChoice", "select_observation"]
