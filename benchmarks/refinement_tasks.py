"""Finite evaluator-side tasks for the refinement benchmark.

The host owns references and privileged formulas. Learning arms receive a
grammar, permitted inputs and paid labels through the benchmark boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

from solvi.experimental.experience.programs import Grammar, Program


Point = dict[str, int | bool]


@dataclass(frozen=True)
class Problem:
    task_id: str
    family: str
    grammar: Grammar
    points: tuple[Point, ...]
    reference: Callable[[Point], int | bool]
    formula: Program
    description: str


def _problem(task_id, family, axes, operators, reference, formula, description):
    names = tuple(axes)
    grammar = Grammar(
        inputs=names,
        constants=(0, 1),
        operators=operators,
        max_depth=4,
        max_nodes=7,
        max_candidates=1500,
        max_abs=1000,
    )
    points = tuple(dict(zip(names, values, strict=True)) for values in product(*axes.values()))

    def target(point):
        if set(point) != set(axes):
            raise ValueError("reference input schema mismatch")
        for name, axis in axes.items():
            value = point[name]
            if type(value) is not type(axis[0]) or value not in axis:
                raise ValueError("reference input outside its typed domain")
        return reference(point)

    return Problem(task_id, family, grammar, points, target, Program(formula), description)


def _input(name):
    return ("input", name)


def tasks() -> tuple[Problem, ...]:
    """Build twelve author-defined mechanisms, each with 32 distinct inputs."""
    four, eight = range(4), range(8)
    flags = (False, True)
    return (
        _problem(
            "net_balance",
            "numeric_tabular",
            {"credit": eight, "debit": four},
            ("sub", "add"),
            lambda p: p["credit"] - p["debit"],
            ("sub", _input("credit"), _input("debit")),
            "Net account balance after a debit, allowing negative balances.",
        ),
        _problem(
            "capped_allocation",
            "numeric_tabular",
            {"requested": eight, "quota": four},
            ("max", "min"),
            lambda p: p["requested"] if p["requested"] <= p["quota"] else p["quota"],
            ("min", _input("requested"), _input("quota")),
            "Allocate at most the requested units and the available quota.",
        ),
        _problem(
            "unfilled_orders",
            "numeric_tabular",
            {"demand": four, "stock": eight},
            ("max", "sub"),
            lambda p: p["demand"] - p["stock"] if p["demand"] > p["stock"] else 0,
            ("max", ("const", 0), ("sub", _input("demand"), _input("stock"))),
            "Count orders that cannot be filled from current stock.",
        ),
        _problem(
            "line_total",
            "numeric_tabular",
            {"quantity": four, "unit_price": eight},
            ("add",),
            lambda p: p["quantity"] * p["unit_price"],
            ("mul", _input("quantity"), _input("unit_price")),
            "Line price is quantity times unit price. The additive learning grammar cannot express this rule.",
        ),
        _problem(
            "below_alert",
            "boolean_policy",
            {"reading": eight, "threshold": four},
            ("lt", "eq"),
            lambda p: p["reading"] < p["threshold"],
            ("lt", _input("reading"), _input("threshold")),
            "Raise an alert strictly below the configured threshold.",
        ),
        _problem(
            "release_gate",
            "boolean_policy",
            {"needed": four, "stock": four, "hold": flags},
            ("lt", "or", "not"),
            lambda p: not p["hold"] and p["stock"] >= p["needed"],
            ("not", ("or", _input("hold"), ("lt", _input("stock"), _input("needed")))),
            "Release an order only when stock suffices and there is no hold.",
        ),
        _problem(
            "priority_access",
            "boolean_policy",
            {"rank": four, "cutoff": four, "override": flags},
            ("lt", "or", "and"),
            lambda p: True if p["override"] else p["rank"] < p["cutoff"],
            ("or", _input("override"), ("lt", _input("rank"), _input("cutoff"))),
            "Admit an override or a rank strictly ahead of the cutoff.",
        ),
        _problem(
            "signed_match",
            "boolean_policy",
            {"sent": four, "received": four, "signed": flags},
            ("eq", "and", "or"),
            lambda p: p["sent"] == p["received"] if p["signed"] else False,
            ("and", _input("signed"), ("eq", _input("sent"), _input("received"))),
            "Accept a reconciliation only when the counts match and it is signed.",
        ),
        _problem(
            "arrival_slot",
            "multistep_dependency",
            {"ready": four, "bay_free": four, "travel": (1, 3)},
            ("max", "add"),
            lambda p: (p["ready"] if p["ready"] > p["bay_free"] else p["bay_free"]) + p["travel"],
            ("add", ("max", _input("ready"), _input("bay_free")), _input("travel")),
            "Depart once both the load and bay are ready, then add travel time.",
        ),
        _problem(
            "available_funds",
            "multistep_dependency",
            {"opening": four, "deposit": four, "reserved": (1, 4)},
            ("add", "sub"),
            lambda p: p["opening"] - p["reserved"] + p["deposit"],
            ("sub", ("add", _input("opening"), _input("deposit")), _input("reserved")),
            "Post the deposit and deduct the amount reserved for a pending payment.",
        ),
        _problem(
            "backup_supply",
            "multistep_dependency",
            {"main": four, "backup": four, "online": flags},
            ("if", "add"),
            lambda p: p["backup"] + (p["main"] if p["online"] else 0),
            ("if", _input("online"), ("add", _input("main"), _input("backup")), _input("backup")),
            "Backup supply remains available when the main supply goes offline.",
        ),
        _problem(
            "pipeline_yield",
            "multistep_dependency",
            {"lanes": four, "rate": four, "outlet": (2, 6)},
            ("mul", "min"),
            lambda p: min(p["outlet"], p["lanes"] * p["rate"]),
            ("min", ("mul", _input("lanes"), _input("rate")), _input("outlet")),
            "Parallel lanes feed a shared outlet with its own capacity.",
        ),
    )
