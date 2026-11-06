"""Bounded minimax choice of a distinguishing observation on finite alternatives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf

from ...core.costs import BudgetStop
from ._common import WorkBudget, canonical, identifier, natural
from .worlds import WorldSet


@dataclass(frozen=True)
class Probe:
    name: str
    read: Callable
    cost: int = 8

    def __post_init__(self):
        identifier(self.name, "probe name")
        natural(self.cost, "probe cost", positive=True)
        if not callable(self.read):
            raise TypeError("probe read must be a pure callable")


@dataclass(frozen=True)
class ProbeChoice:
    name: str | None
    status: str
    worst_case_cost: int | None = None
    reason: str = ""


def select_probe(
    worlds: WorldSet, question: Callable, probes: Sequence[Probe], budget: WorkBudget, *, horizon: int = 2
) -> ProbeChoice:
    """Find the cheapest worst-case resolving tree within a bounded horizon.

    This plans only over the supplied finite alternatives and pure read models.
    All label/probe evaluations and branch expansions are paid. Predictions are
    not observations. If no tree is found, a deterministic fallback splits the
    remaining worlds; this is incomplete planning, not a guarantee of optimality.
    """
    natural(horizon, "lookahead horizon", positive=True)
    if horizon > 16:
        raise ValueError("lookahead horizon exceeds the finite planning cap")
    names = [p.name for p in probes]
    if len(names) != len(set(names)):
        raise ValueError("probe names must be distinct")
    if not len(worlds):
        return ProbeChoice(None, "contradiction", reason="no modeled state")
    states = worlds.worlds
    labels, outputs = [], []
    post_debit_failure = False

    def charge(kind, units=1):
        nonlocal post_debit_failure
        before = budget.spent
        try:
            budget.charge(kind, units)
        except BudgetStop:
            # A durable sink can raise BudgetStop too. After debit this is a
            # failed ledger write, not ordinary exhaustion: stop the caller.
            post_debit_failure = budget.spent != before
            raise

    try:
        for state in states:
            charge("probe_decision")
            labels.append(canonical(question(state)))
        if len(set(labels)) == 1:
            return ProbeChoice(None, "resolved", 0, "answer already invariant")
        for probe in probes:
            row = []
            for state in states:
                charge("probe_model")
                row.append(canonical(probe.read(state)))
            outputs.append(row)

        partitions = {}

        def partition(indices, p):
            # A reuse still costs a lookup; only avoid rebuilding paid groups.
            charge("probe_partition_lookup")
            key = (indices, p)
            if key in partitions:
                return partitions[key]
            groups = {}
            for i in indices:
                charge("probe_partition")
                groups.setdefault(outputs[p][i], []).append(i)
            result = tuple(tuple(g) for _, g in sorted(groups.items()))
            partitions[key] = result
            return result

        memo = {}

        def solve(indices, available, depth):
            charge("probe_branch")
            key = (indices, available, depth)
            if key in memo:
                return memo[key]
            if len({labels[i] for i in indices}) == 1:
                memo[key] = (0, None)
                return (0, None)
            if not depth:
                memo[key] = (inf, None)
                return (inf, None)
            best = (inf, None)
            for p in available:
                charge("probe_candidate")
                # Preserve input order; an equal lower bound only loses if
                # this candidate also cannot improve the name tie break.
                loses_tie = best[1] is not None and probes[p].name >= probes[best[1]].name
                if probes[p].cost > best[0] or (probes[p].cost == best[0] and loses_tie):
                    continue
                groups = partition(indices, p)
                if len(groups) <= 1:
                    continue
                rest = tuple(i for i in available if i != p)
                cost = probes[p].cost
                for group in groups:
                    child = solve(group, rest, depth - 1)[0]
                    if child == inf:
                        cost = inf
                        break
                    # Costs are nonnegative. Once this lower bound loses,
                    # remaining children cannot make this candidate win.
                    cost = max(cost, probes[p].cost + child)
                    if cost > best[0] or (cost == best[0] and loses_tie):
                        break
                if cost < best[0] or (
                    cost == best[0] and cost != inf and (best[1] is None or probes[p].name < probes[best[1]].name)
                ):
                    best = (cost, p)
            memo[key] = best
            return best

        indices = tuple(range(len(states)))
        cost, selected = solve(indices, tuple(range(len(probes))), horizon)
        if selected is not None:
            if probes[selected].cost > budget.remaining:
                return ProbeChoice(None, "budget", reason="cannot afford chosen external observation")
            return ProbeChoice(probes[selected].name, "selected", int(cost), "bounded minimax resolving tree")
        # Do not falsely claim that depth-limited failure proves no useful
        # sequence exists (XOR and other synergies need multiple observations).
        fallback = []
        for p, probe in enumerate(probes):
            groups = partition(indices, p)
            if len(groups) <= 1:
                continue
            # A world split alone is insufficient: an independent noise sensor
            # must not be bought merely because it creates many partitions.
            # Check conditional relevance given the other available sensors.
            # This also recognizes XOR synergy without claiming completeness.
            other = {}
            for i in indices:
                charge("probe_relevance", max(1, len(probes)))
                key = tuple(outputs[j][i] for j in range(len(probes)) if j != p)
                other.setdefault(key, []).append(i)
            useful = False
            for group in other.values():
                before = len({labels[i] for i in group})
                after = partition(tuple(group), p)
                if any(len({labels[i] for i in child}) < before for child in after):
                    useful = True
                    break
            if useful:
                fallback.append((max(map(len, groups)), probe.cost, probe.name))
        if fallback:
            # All fallback planning has now been paid. Recheck affordability
            # against this final balance, not an earlier loop iteration.
            affordable = [f for f in fallback if f[1] <= budget.remaining]
            if not affordable:
                return ProbeChoice(None, "budget", reason="planning left no affordable distinguishing observation")
            return ProbeChoice(
                min(affordable)[2], "selected", reason="incomplete lookahead; relevant splitting fallback"
            )
        return ProbeChoice(None, "unavailable", reason="no affordable probe distinguishes modeled states")
    except BudgetStop:
        if post_debit_failure:
            raise
        return ProbeChoice(None, "budget", reason="planning exhausted work budget; no observation bought")


__all__ = ["Probe", "ProbeChoice", "select_probe"]
