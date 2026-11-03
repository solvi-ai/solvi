"""Exact finite-support checks, not Monte Carlo estimates or a general FDR proof."""
from copy import deepcopy
from itertools import product

import pytest

from solvi.stat import ELOND, e_bh


def test_ebh_all_null_disjoint_support_has_exact_fdr_q():
    # Disjoint events: with probability q/m only null i has e=m/q;
    # otherwise every e is zero. Each null e has expectation exactly one.
    m, q = 4, .1
    fdr = 0
    for index in range(m):
        values = [0] * m
        values[index] = m / q
        rejected = e_bh(values, q)
        assert rejected == [index]
        fdr += q / m
    assert e_bh([0] * m, q) == []
    assert fdr == pytest.approx(q)


@pytest.mark.parametrize("schedule", ["uniform", "basel"])
def test_elond_exact_all_paths_for_each_three_test_null_configuration(schedule):
    q, horizon = .1, 3
    for nulls in product([False, True], repeat=horizon):
        initial = ELOND(q, schedule=schedule, **({"horizon": horizon} if schedule == "uniform" else {}))
        paths = [(initial, 1., 0)]
        for is_null in nulls:
            next_paths = []
            for rule, mass, false_count in paths:
                allocation = rule.reserve()
                # Null: E_t is threshold with probability alpha, else zero;
                # conditional expected e-value is one even after past discoveries.
                outcomes = [(allocation.alpha, allocation.threshold), (1 - allocation.alpha, 0)] if is_null else [(1, float("inf"))]
                for probability, evidence in outcomes:
                    branch = deepcopy(rule)
                    rejected = branch.finish(allocation.index, evidence)
                    next_paths.append((branch, mass * probability, false_count + int(is_null and rejected)))
            paths = next_paths
            assert sum(mass for _, mass, _ in paths) == pytest.approx(1)
            fdr = sum(mass * false_count / max(1, rule.discoveries) for rule, mass, false_count in paths)
            assert fdr <= q + 1e-14
