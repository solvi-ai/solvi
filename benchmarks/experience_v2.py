"""Pre-registration r2 component harness. No development/final launcher yet.

This module runs engineering fixtures only. Final remains closed until the
strategic adapter, process isolation, recovery audits and independent review.
"""

from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import time

from solvi.experimental.experience._common import WorkBudget, BudgetStop, canonical, digest, natural
from solvi.experimental.experience import acquisition, programs
from solvi.experimental.experience.competences import CompetenceRegistry, ExternalExample
from solvi.experimental.experience.worlds import WorldSet

FAMILIES = {
    "inventory": (
        ("stock", "reserved", "damaged"),
        (range(5),) * 3,
        ("sub", "max", "add"),
        "max(0, stock-reserved-damaged)",
    ),
    "eligibility": (
        ("requested", "available", "expired"),
        (range(5), range(5), (False, True)),
        ("lt", "not", "and", "or", "eq"),
        "requested < available AND NOT expired",
    ),
    "schedule": (("a", "b", "c", "d"), (range(4),) * 4, ("add", "max", "sub"), "max(a+b,c)+d"),
}


def ordered(seed, purpose, family, items):
    return sorted(items, key=lambda item: (digest(["tab-v2-r2", seed, purpose, family, item]), canonical(item)))


def points(family):
    return [list(row) for row in itertools.product(*FAMILIES[family][1])]


def roles(seed, family):
    rows = points(family)
    maximum = max(row[0] for row in rows)
    held = lambda row: (row[0] == maximum and row[1] % 2 == 0) or (row[0] == 0 and row[1] % 2 == 1)
    regular = ordered(seed, "rows-R", family, [r for r in rows if not held(r)])
    boundary = ordered(seed, "rows-H", family, [r for r in rows if held(r)])
    return dict(proposal=regular[:8], admission=regular[8:16], transfer_R=regular[16:40], transfer_H=boundary[:8])


class Wire:
    def __init__(self, seed, family, enabled=False):
        self.seed, self.family, self.enabled = seed, family, enabled
        self.names = FAMILIES[family][0]

    def identity(self, kind, original):
        return (
            ("wire-" + kind + ":" + digest(["wire", self.seed, self.family, kind, original]))
            if self.enabled
            else kind + ":" + str(original)
        )

    def encode(self, row):
        pairs = [
            (("w" + str(i)) if self.enabled else name, value) for i, (name, value) in enumerate(zip(self.names, row))
        ]
        return dict(reversed(pairs)) if self.enabled else dict(pairs)

    def decode(self, message):
        names = tuple("w" + str(i) for i in range(len(self.names))) if self.enabled else self.names
        if set(message) != set(names):
            raise ValueError("wire input schema mismatch")
        return [message[name] for name in names]

    def receipt(self, value):
        return {"wt" if self.enabled else "target": value}

    def decode_receipt(self, receipt):
        key = "wt" if self.enabled else "target"
        if set(receipt) != {key}:
            raise ValueError("wire result schema mismatch")
        return receipt[key]


def sql_value(family, row):
    names, _, _, expression = FAMILIES[family]
    with sqlite3.connect(":memory:") as db:
        columns = ",".join("? AS " + name for name in names)
        value = db.execute("SELECT " + expression + " FROM (SELECT " + columns + ")", row).fetchone()[0]
    return bool(value) if family == "eligibility" else value


class NestedBudget:
    """An atomic sublimit over one ledger, never fresh credit or double debit."""

    def __init__(self, parent, limit=100000):
        self.parent, self.limit, self.start = parent, natural(limit), parent.spent

    @property
    def spent(self):
        return self.parent.spent - self.start

    @property
    def remaining(self):
        return max(0, min(self.limit - self.spent, self.parent.remaining))

    @property
    def counts(self):
        return self.parent.counts

    def charge(self, kind, units=1):
        natural(units)
        if units > self.remaining:
            raise BudgetStop("nested work cap")
        self.parent.charge(kind, units)


def load_baseline(root, expected_hashes):
    """Load only caller-pinned exact source bytes; no git/network execution."""
    modules = {}
    for leaf in ("programs", "acquisition"):
        path = Path(root) / "src/solvi/experimental/experience" / (leaf + ".py")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hashes[leaf]:
            raise ValueError("baseline source hash mismatch: " + leaf)
        name = "solvi.experimental.experience._baseline_v2_" + leaf + "_" + expected_hashes[leaf][:12]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[leaf] = module
    return modules


def s1_fixture(seed, profile, prices, wire=False):
    if profile == "threshold":
        axes = (range(3), range(3), range(2))

        def columns(row):
            u, v, n = row
            q = u + v >= 3
            return q, (q, u, v, n, u >= 1, q)
    elif profile == "conditional":
        axes = (range(2), range(2), range(2), range(3))

        def columns(row):
            f, u, n, v = row
            return (v == 2 if f else u == 1), (f, u, v, n, v == 2, u == 1)
    elif profile == "hidden":
        axes = (range(2),) * 4

        def columns(row):
            f, u, n, h = row
            return (u == 1 if f else h == 1), (f, u, n, f + u, u == 1, n == 1)
    else:
        raise ValueError("unknown S1 profile")
    target = "wt" if wire else "answer"
    field = lambda index: ("w0" if wire else "p") + str(index)
    states = []
    for index, row in enumerate(itertools.product(*axes)):
        answer, values = columns(row)
        state = {
            "wi" if wire else "model_id": ("wire-model:" if wire else "model:") + str(index),
            target: answer,
            **{field(i): value for i, value in enumerate(values)},
        }
        # Exercise the actual serialized representation before normalization.
        encoded = dict(reversed(list(state.items()))) if wire else state
        states.append(json.loads(json.dumps(encoded, sort_keys=not wire)))
    question = lambda state: state[target]
    order = ordered(seed, "probe-order", profile, list(range(6)))
    if wire:
        order = list(reversed(order))
    probes = [acquisition.Probe(field(i), lambda state, i=i: state[field(i)], prices[i]) for i in order]
    return WorldSet(states), question, probes


def s1_reference(worlds, question, probes, horizon, remaining):
    """Uncached exhaustive trees; independent direct conditional relevance."""
    states = worlds.worlds
    labels = [canonical(question(s)) for s in states]
    outputs = [[canonical(p.read(s)) for s in states] for p in probes]
    work = 0

    def groups(indices, p):
        nonlocal work
        result = {}
        for i in indices:
            work += 1
            result.setdefault(outputs[p][i], []).append(i)
        return list(result.values())

    def tree(indices, available, depth):
        nonlocal work
        work += 1
        if len({labels[i] for i in indices}) == 1:
            return 0, None
        if not depth:
            return float("inf"), None
        options = []
        for p in available:
            children = groups(indices, p)
            if len(children) <= 1:
                continue
            costs = [tree(child, [j for j in available if j != p], depth - 1)[0] for child in children]
            options.append((probes[p].cost + max(costs), probes[p].name, p))
        finite = [x for x in options if x[0] != float("inf")]
        if not finite:
            return float("inf"), None
        best = min(finite)
        return best[0], best[2]

    indices = list(range(len(states)))
    if not indices:
        return dict(status="contradiction", name=None, cost=None, work=work)
    if len(set(labels)) == 1:
        return dict(status="resolved", name=None, cost=0, work=work)
    cost, p = tree(indices, list(range(len(probes))), horizon)
    if p is not None:
        return dict(
            status="selected" if probes[p].cost <= remaining else "budget",
            name=probes[p].name if probes[p].cost <= remaining else None,
            cost=cost if probes[p].cost <= remaining else None,
            work=work,
        )
    fallback = []
    for p, probe in enumerate(probes):
        partition = groups(indices, p)
        if len(partition) <= 1:
            continue
        others = {}
        for i in indices:
            work += 1
            others.setdefault(tuple(outputs[j][i] for j in range(len(probes)) if j != p), []).append(i)
        useful = any(
            any(len({labels[i] for i in child}) < len({labels[i] for i in group}) for child in groups(group, p))
            for group in others.values()
        )
        if useful:
            fallback.append((max(map(len, partition)), probe.cost, probe.name))
    affordable = [x for x in fallback if x[1] <= remaining]
    status = "selected" if affordable else ("budget" if fallback else "unavailable")
    return dict(status=status, name=min(affordable)[2] if affordable else None, cost=None, work=work)


def compare_s1(
    seed, profile, prices, horizon, old, parents, wire=False, *, approved_phase=None, include_reference=True
):
    require_engineering_seed(seed, approved_phase=approved_phase)
    worlds, question, probes = s1_fixture(seed, profile, prices, wire)
    result = {}
    for name, module in [("old", old), ("new", acquisition)]:
        budget = NestedBudget(parents[name])
        start = parents[name].spent
        choice = module.select_probe(
            worlds, question, [module.Probe(p.name, p.read, p.cost) for p in probes], budget, horizon=horizon
        )
        result[name] = dict(
            status=choice.status,
            name=choice.name,
            cost=choice.worst_case_cost,
            spent=budget.spent,
            start=start,
            end=parents[name].spent,
            remaining=budget.remaining,
        )
        if include_reference:
            result[name]["reference"] = s1_reference(worlds, question, probes, horizon, budget.remaining)
    return result


def compare_s2(seed, family, old, wire=False, limit=200000, *, approved_phase=None):
    require_engineering_seed(seed, approved_phase=approved_phase)
    codec = Wire(seed, family, wire)
    selected = roles(seed, family)["proposal"]
    results = {}
    for version, module in [("old", old), ("new", programs)]:
        parent = WorkBudget(limit)
        budget = NestedBudget(parent)
        names = tuple("w" + str(i) for i in range(len(codec.names))) if wire else codec.names
        grammar = module.Grammar(
            names,
            constants=(0, 1),
            operators=FAMILIES[family][2],
            max_depth=4,
            max_nodes=7,
            max_candidates=10000,
            max_abs=32,
        )
        started = time.monotonic()
        outcome = None
        try:
            rows = []
            for i, row in enumerate(selected):
                budget.charge("journal_event")
                budget.charge("proposal_label", 8)
                expected = codec.decode_receipt(codec.receipt(sql_value(family, codec.decode(codec.encode(row)))))
                budget.charge("journal_event")
                rows.append(
                    ExternalExample(
                        codec.identity("event", i),
                        codec.identity("task", i),
                        "environment",
                        "v1",
                        "scalar",
                        family,
                        codec.encode(row),
                        expected,
                        "proposal",
                    )
                )
            # Same registry validation/registration boundary; only enumerator differs.
            registry = CompetenceRegistry(programs.Grammar.from_dict(grammar.to_dict()))
            budget.charge("competence_propose")
            validated = registry._batch(rows, budget, "proposal")
            registry._remember(validated)
            search = module.synthesize([(e.inputs, e.expected) for e in validated], grammar, budget)
            outcome = dict(
                status="completed",
                reason=search.reason,
                tried=search.tried,
                ast=None if search.candidate is None else search.candidate.to_dict(),
            )
        except BudgetStop:
            outcome = dict(status="budget", reason="nested_budget", tried=None, ast=None)
        except Exception as error:  # noqa: BLE001 - retain failed trials instead of losing their costs
            outcome = dict(status="error", reason=type(error).__name__, message=str(error), tried=None, ast=None)
        results[version] = {
            **outcome,
            "spent": parent.spent,
            "counts": parent.counts,
            "wall_seconds": time.monotonic() - started,
        }
    return results


class TrialLog:
    """Incremental chained JSONL. Resume never reruns terminal trial IDs."""

    TERMINAL = {"completed", "budget", "error", "censored", "interrupted", "rejected", "unknown"}

    def __init__(self, path, config, *, resume=False):
        self.path = Path(path)
        self.config = config
        self.previous = ""
        self.finished = set()
        if self.path.exists() and not resume:
            raise ValueError("fresh output required")
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                checksum = record.pop("hash")
                if (
                    record["previous"] != self.previous
                    or record["config"] != digest(config)
                    or digest(record) != checksum
                ):
                    raise ValueError("invalid trial log chain/config")
                if record["trial"] in self.finished:
                    raise ValueError("duplicate terminal trial")
                self.finished.add(record["trial"])
                self.previous = checksum

    def append(self, trial, status, payload):
        if status not in self.TERMINAL or trial in self.finished:
            raise ValueError("invalid or duplicate terminal")
        record = dict(previous=self.previous, config=digest(self.config), trial=trial, status=status, payload=payload)
        record["hash"] = digest(record)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            import os

            os.fsync(stream.fileno())
        self.previous = record["hash"]
        self.finished.add(trial)


def require_engineering_seed(seed, *, approved_phase=None):
    if 8101 <= seed <= 8112 and approved_phase != "final":
        raise ValueError("final remains closed")
    if 5001 <= seed <= 5012 and approved_phase != "development":
        raise ValueError("comparative development requires independent harness review")
