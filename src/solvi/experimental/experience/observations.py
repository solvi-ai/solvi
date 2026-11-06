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


class _ObservationLimit(BudgetStop):
    """The selector's paid work window is exhausted, before a debit."""


class _BoundedPaid(_Paid):
    def __init__(self, budget, max_work, reserve):
        super().__init__(budget)
        self.start = budget.spent
        self.max_work = max_work
        self.reserve = reserve

    @property
    def spent(self):
        return self.budget.spent

    @property
    def remaining(self):
        return max(0, min(self.max_work - (self.spent - self.start), self.budget.remaining - self.reserve))

    def charge(self, kind, units=1):
        natural(units, "work units")
        if units > self.remaining:
            raise _ObservationLimit("bounded observation work exhausted")
        try:
            super().charge(kind, units)
        except BudgetStop as error:
            raise _ObservationLimit(str(error)) from error


def select_observation_bounded(models, options, budget, *, predict, max_work=1024, reserve=0, result_type=None):
    """Choose a paid typed disagreement/price row, retaining completed work.

    ``predict(model, inputs, paid_budget)`` must be pure, charge its own work
    before execution, and have no access to labels or an oracle. It may wrap
    Program.evaluate or an external backend. The optional strict bool/int
    result contract is a host promise, not authentication of the callback.
    Without a contract, ValueError becomes one shared invalid marker.

    All selector and prediction debits share the parent's ledger and the
    max_work window, while preserving reserve. A cap returns selected_partial
    when a fully scored, still affordable row exists. Prices are never paid
    here. Metadata and a final affordable-row pass are paid before processing;
    input copies are made only for visited options. Schema and input bounds
    belong to the caller's backend wrapper. At most 32 models / 256 options
    are copied, including when the supplied iterables are unbounded.
    """
    natural(max_work, "max_work")
    natural(reserve, "reserve")
    if result_type is not None and result_type is not bool and result_type is not int:
        raise ValueError("result_type must be bool, int or None")
    if not callable(predict):
        raise ValueError("pure paid predict callback required")
    # Sized oversized inputs can be rejected without traversing or copying.
    for values, limit in ((models, 32), (options, 256)):
        if hasattr(values, "__len__") and len(values) > limit:
            raise ValueError(f"at most {limit} entries required")
    paid = _BoundedPaid(budget, max_work, reserve)
    scored = []
    stopped = False
    try:
        paid.charge("observation_plan")
        committee = []
        for model in models:
            paid.charge("observation_plan")
            if len(committee) >= 32:
                raise ValueError("at most 32 models required")
            committee.append(model)
        pool, names = [], set()
        for option in options:
            paid.charge("observation_plan")
            if len(pool) >= 256 or type(option) is not ObservationOption or option.name in names:
                raise ValueError("at most 256 uniquely named observation options required")
            # Immutable ObservationOption already validates strict metadata.
            names.add(option.name)
            pool.append(option)
        if len(committee) < 2 or not pool:
            return ObservationChoice(None, "no_disagreement", 0)
        # Prepay the bounded final pass so a cap cannot prevent affordable
        # fallback selection among rows completed before that cap.
        paid.charge("observation_compare", len(pool))
        paid.charge("observation_cache", len(pool))
        suffix_cost = [None] * (len(pool) + 1)
        for index in range(len(pool) - 1, -1, -1):
            tail = suffix_cost[index + 1]
            suffix_cost[index] = pool[index].cost if tail is None else min(pool[index].cost, tail)
        size = len(committee)
        max_pairs = size * size // 4 if result_type is bool else size * (size - 1) // 2
        best, best_pairs = None, 0
        for index, option in enumerate(pool):
            paid.charge("observation_plan")
            if best is not None and best.cost <= budget.remaining and (
                best_pairs * suffix_cost[index] >= max_pairs * best.cost
            ):
                break
            paid.charge("observation_cache")
            inputs = option.inputs
            histogram, pairs = {}, 0
            for model_index, model in enumerate(committee):
                paid.charge("observation_plan")
                paid.charge("observation_output")
                try:
                    value = predict(model, inputs, paid)
                except ValueError:
                    if result_type is not None:
                        raise
                    output = (None,)
                else:
                    if type(value) not in (bool, int):
                        if result_type is not None:
                            raise ValueError("prediction violates declared result_type")
                        output = (None,)
                    elif result_type is not None and type(value) is not result_type:
                        raise ValueError("prediction violates declared result_type")
                    else:
                        output = (type(value), value)
                paid.charge("observation_compare")
                equal = histogram.get(output, 0)
                pairs += model_index - equal
                histogram[output] = equal + 1
            paid.charge("observation_compare")
            scored.append((option, pairs))
            if pairs and (best is None or pairs * best.cost > best_pairs * option.cost):
                best, best_pairs = option, pairs
    except _DebitFailure as failure:
        raise failure.error from None
    except _ObservationLimit:
        stopped = True
    # This pass was prepaid before any prediction. Unfinished rows were never
    # appended, so neither partial output vectors nor partial pair counts win.
    best, best_pairs = None, 0
    disagreement = False
    for option, pairs in scored:
        disagreement = disagreement or bool(pairs)
        if pairs and option.cost <= budget.remaining and (
            best is None or pairs * best.cost > best_pairs * option.cost
        ):
            best, best_pairs = option, pairs
    if best is not None:
        return ObservationChoice(best.name, "selected_partial" if stopped else "selected", best_pairs)
    return ObservationChoice(None, "budget" if stopped or disagreement else "no_disagreement", 0)


__all__ = ["ObservationOption", "ObservationChoice", "select_observation", "select_observation_bounded"]
