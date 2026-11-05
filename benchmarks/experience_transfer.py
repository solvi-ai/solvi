"""CPU finite composite transfer harness with source-bound evaluation gates.

Non-smoke evaluation requires the protocol, implementation and split freeze.
OOD adaptation uses a paid teacher prefix and one fresh proposal/admission.
This is a scalar action task; it does not implement the superseded xyh draft.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import threading
import time

from solvi.experimental.experience._common import BudgetStop, WorkBudget, natural
from solvi.experimental.experience.acquisition import Probe, select_probe
from solvi.experimental.experience.competences import CompetenceRegistry, ExternalExample
from solvi.experimental.experience.environments import FiniteEnvironment
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.worlds import WorldSet

try:
    from ._experience_scorer import hard_violation, score
except ImportError:
    from _experience_scorer import hard_violation, score

ARMS = ("SOLVER", "FORMULA_CACHE", "FIXED", "RANDOM", "ACTIVE", "FULL", "NO_CALL")
SEEDS = tuple(range(1101, 1113))
BUDGETS = (16384, 65536, 262144)
PRICES = (1, 8, 32)
SPEC = "xy-int-v2-composite-teacher-prefix"


class CompositeEnvironment(FiniteEnvironment):
    """Benchmark-only service fee/setup; base library semantics stay intact."""

    @staticmethod
    def _truth(case):
        base = FiniteEnvironment._truth(case)
        if case["family"] == "resource":
            inventory = list(range(base))
            if inventory:
                inventory.pop()
            return len(inventory)
        clock = base
        for _ in ("setup",):
            clock += 1
        return clock

    def task(self, task_id):
        task = super().task(task_id)
        answers = sorted({w["answer"] + offset for w in task.worlds for offset in (-1, 0, 1)})
        worlds = tuple(dict(answer=a, noise=n, parity=a % 2) for a in answers for n in (0, 1))
        return replace(task, scope="composite:" + task.scope, worlds=worlds)


def is_composite(program):
    def operators(ast):
        return int(ast[0] not in ("input", "const")) + sum(operators(x) for x in ast[1:] if isinstance(x, tuple))

    return set(program.input_names) == {"x", "y"} and operators(program.ast) >= 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def component_seed(master, family, split, episode, component):
    return int(digest([master, family, split, episode, component])[:16], 16)


def partitions(seed, family, split="dev"):
    """Dev never creates final case lists. OOD is paired with ID, explicitly."""
    if split not in ("dev", "final") or family not in ("resource", "workflow"):
        raise ValueError("invalid split/family")
    pairs = [(x, y) for x in range(32) for y in range(32) if family == "workflow" or x >= y + 3]
    random.Random(component_seed(seed, family, "partition", 0, "shuffle")).shuffle(pairs)
    if family == "workflow":
        # Deterministic before eval: both orientations in every learning packet.
        for start in (0, 16, 352, 368) if split == "final" else (0, 16):
            for offset, less in ((0, True), (1, False)):
                target = start + offset
                source = next(
                    i
                    for i in range(target, len(pairs))
                    if (pairs[i][0] < pairs[i][1] if less else pairs[i][0] > pairs[i][1])
                )
                pairs[target], pairs[source] = pairs[source], pairs[target]
    names = [("proposal", 16), ("admission", 16), ("dev", 64)]
    if split == "final":
        names += [("id", 128), ("encoding", 128)]
    result, offset = {}, 0
    for name, count in names:
        rows = {}
        for i, (x, y) in enumerate(pairs[offset : offset + count]):
            rows[f"{seed}:{family}:{name}:{i}"] = dict(
                family=family, x=x, y=y, mechanism="v1", announced_version=False, threshold=4, noise=i % 2
            )
        result[name] = rows
        offset += count
    if split == "final":
        result["ood"] = {
            f"{seed}:{family}:ood:{i}": dict(
                family=family, x=x, y=y, mechanism="v2", announced_version=seed <= 1106, threshold=4, noise=i % 2
            )
            for i, (x, y) in enumerate(pairs[offset : offset + 32])
        }
        result["ood"].update(
            {
                f"{seed}:{family}:ood:{i + 32}": dict(case, mechanism="v2", announced_version=seed <= 1106)
                for i, case in enumerate(list(result["id"].values())[:96])
            }
        )
    return result


def manifest(parts):
    return dict(
        spec=SPEC,
        paired_splits={"id_first96": "ood_suffix96"},
        schema_renamed=False,
        partitions={
            name: dict(sha256=digest(rows), ids=list(rows), tuples=[[c["x"], c["y"]] for c in rows.values()])
            for name, rows in parts.items()
        },
    )


class Ledger(WorkBudget):
    """Counts use exact eighth-units: compute 8, effect 64, probes p/4p/8p."""

    def __init__(self, limit, observation_price):
        super().__init__(8 * natural(limit, "raw budget"))
        self.observation_price = natural(observation_price, "observation price", positive=True)
        self.raw = {}

    def charge(self, kind, units=1):
        natural(units, "raw work units")
        prices = {
            "observation": 8 * self.observation_price,
            "observe_measure": 8 * self.observation_price,
            "observe_diagnose": 4 * self.observation_price,
            "observe_noise": self.observation_price,
            "action": 64,
        }
        price = prices.get(kind, 8)
        super().charge(kind, units * price)
        self.raw[kind] = self.raw.get(kind, 0) + units

    def probe_cost(self, nominal_cost):
        return natural(nominal_cost, "nominal probe cost", positive=True) * self.observation_price


def external_example(task, receipt, split):
    return ExternalExample(
        receipt.event_id,
        task.task_id,
        receipt.source,
        receipt.mechanism,
        task.schema,
        task.scope,
        task.inputs,
        receipt.value["value"] if receipt.kind == "observe" else receipt.value["answer"],
        split,
    )


def acquire(cases, split, budget):
    env = CompositeEnvironment(cases)
    rows = []
    for task_id in cases:
        task = env.task(task_id)
        budget.charge("observation")
        receipt = env.execute(task_id, "observe", {"probe": "measure", "field": "answer"}, task_id + ":train")
        rows.append(external_example(task, receipt, split))
    return rows


def solve_task(env, task_id, arm, budget, registry, cache, rng, encoding=False):
    task = env.task(task_id)
    app, proposed = None, None
    if arm == "FULL":
        app = registry.predict(
            task.inputs, task_id=task_id, scope=task.scope, mechanism=task.mechanism, schema=task.schema, budget=budget
        )
        if app.status == "applied":
            proposed = app.value
    if arm == "FORMULA_CACHE":
        budget.charge("cache_lookup")
        key = (task.scope, task.mechanism, task.inputs["x"], task.inputs["y"])
        if key in cache:
            proposed = cache[key]
        else:
            x, y = task.inputs["x"], task.inputs["y"]
            budget.charge("formula_primitive", 2)
            proposed = x - y - 1 if task.scope == "composite:resource" else max(x, y) + 1
            if task.mechanism == "v2":
                budget.charge("formula_primitive", 2)
                proposed = x - y - 1 - int(x >= 4) if task.scope == "composite:resource" else x + y + 1
            budget.charge("cache_fill")
            cache[key] = proposed
    if proposed is None:
        budget.charge("world_materialize", len(task.worlds))
        worlds = WorldSet(task.worlds)
        # Encoding control renames probe IDs only, translating at the host boundary.
        aliases = {"measure": "zeta", "noise": "alpha", "diagnose": "mu"} if encoding else {}
        available = {aliases.get(p.name, p.name): p for p in task.probes}
        while True:
            decision = worlds.decide(lambda w: w["answer"], budget)
            if decision.resolved:
                proposed = decision.value
                break
            if decision.status == "budget":
                raise BudgetStop("decision exhausted")
            if not available or decision.status == "contradiction":
                return dict(status="unknown", success=False, applied=False)
            names = sorted(available)
            if arm in ("ACTIVE", "FULL", "NO_CALL"):
                choice = select_probe(
                    worlds,
                    lambda w: w["answer"],
                    [
                        Probe(name, lambda w, f=available[name].field: w[f], budget.probe_cost(available[name].cost))
                        for name in names
                    ],
                    budget,
                    horizon=2,
                )
                if choice.name is None:
                    if choice.status == "budget":
                        raise BudgetStop("probe planning exhausted")
                    return dict(status="unknown", success=False, applied=False)
                selected = choice.name
            else:
                selected = rng.choice(names) if arm == "RANDOM" else names[0]
            probe = available.pop(selected)
            budget.charge("observe_" + probe.name)
            receipt = env.execute(
                task_id, "observe", dict(probe=probe.name, field=probe.field), task_id + ":probe:" + selected
            )
            worlds = worlds.observe(probe.field, receipt.value["value"], receipt.event_id, budget)
    budget.charge("action")
    receipt = env.execute(task_id, "act", dict(answer=proposed), task_id + ":act")
    row = dict(
        status="done" if receipt.value["ok"] else "wrong",
        proposed=proposed,
        receipt=receipt.to_dict(),
        applied=bool(app and app.status == "applied"),
    )
    if row["applied"]:
        row.update(application_id=app.application_id, competence_id=app.competence_id)
        try:
            registry.observe(app.application_id, external_example(task, receipt, "evaluation"), budget)
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            # Outcome survives failed revision; caller checkpoints it before another task.
            row.update(revision_error=repr(exc), revision_pending=True)
    return row


def teacher_task(env, task_id, index, budget, registry, record=None):
    """Paid external label precedes safety checks and prefix action."""
    task = env.task(task_id)
    budget.charge("observation")
    measurement = env.execute(task_id, "observe", dict(probe="measure", field="answer"), task_id + ":teacher")
    record = {} if record is None else record
    record.update(measurement=measurement.to_dict(), proposed=measurement.value["value"], applied=False)
    snapshot = registry.snapshot()
    revoked = []
    for competence_id, competence in snapshot["competences"].items():
        budget.charge("adaptation_lookup")
        if competence["status"] != "active":
            continue
        candidate = snapshot["candidates"][competence["candidate_id"]]
        if candidate["scope"] != task.scope:
            continue
        if candidate["mechanism"] != task.mechanism:
            reason = "announced_mechanism_change"
        else:
            prediction = Program.from_dict(candidate["program"]).evaluate(task.inputs, budget, registry.grammar)
            budget.charge("adaptation_compare")
            if prediction == measurement.value["value"]:
                continue
            reason = "external_prefix_counterexample"
        budget.charge("adaptation_invalidate")
        revoked.extend(registry.invalidate(competence_id, reason, measurement.event_id))
    example = external_example(task, measurement, "proposal" if index < 16 else "admission")
    budget.charge("action")
    proposed = measurement.value["value"]
    receipt = env.execute(task_id, "act", dict(answer=proposed), task_id + ":teacher-act")
    record.update(
        status="done" if receipt.value["ok"] else "wrong",
        proposed=proposed,
        receipt=receipt.to_dict(),
        measurement=measurement.to_dict(),
        applied=False,
        adaptation_example=example.to_dict(),
        revoked=revoked,
    )
    return record


def run_trial(
    seed,
    family,
    arm,
    limit,
    observation_price,
    stratum="dev",
    tasks=None,
    emit=None,
    checkpoint=None,
    checkpoint_path=None,
):
    """Real environment effects, one aggregate budget including all training."""
    if arm not in ARMS or stratum not in ("dev", "id", "encoding", "ood"):
        raise ValueError("invalid arm/stratum")
    started = time.monotonic()
    random.seed(seed)
    parts = partitions(seed, family, "dev" if stratum == "dev" else "final")
    cases = parts[stratum]
    if tasks is not None:
        cases = dict(list(cases.items())[:tasks])
    budget = Ledger(limit, observation_price)
    grammar = Grammar(("x", "y"), operators=("sub", "max", "add"))
    registry = CompetenceRegistry(grammar)
    cache, rows, learning = {}, [], {"status": "disabled"}
    training_failed = False
    if arm in ("FULL", "NO_CALL") and checkpoint is None:
        if checkpoint_path is not None:
            atomic_json(
                Path(checkpoint_path),
                dict(
                    config=[SPEC, seed, family, arm, limit, observation_price, stratum, tasks],
                    stage="initial_learning_inflight",
                ),
            )
        try:
            proposal = acquire(parts["proposal"], "proposal", budget)
            candidate = registry.propose(proposal, budget)
            learning = dict(status="search_failed", reason=registry.last_search_reason)
            if candidate is not None:
                budget.charge("composite_gate", candidate.program.nodes)
            if candidate is not None and not is_composite(candidate.program):
                learning = dict(status="not_composite", candidate=candidate.to_dict())
            elif candidate is not None:
                admission = acquire(parts["admission"], "admission", budget)
                record = registry.admit(candidate, admission, budget, min_examples=16)
                learning = dict(status="admission", candidate=candidate.to_dict(), record=record.__dict__)
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            training_failed = True
            learning = dict(status="budget" if isinstance(exc, BudgetStop) else "error", error=repr(exc))
    training_cost = budget.spent
    env = CompositeEnvironment(cases)
    exhausted = training_failed
    config = [SPEC, seed, family, arm, limit, observation_price, stratum, tasks]
    adaptation_rows, adaptation_done, adaptation_learning = [], False, {"status": "pending"}
    if checkpoint is not None:
        if checkpoint["config"] != config:
            raise ValueError("checkpoint config mismatch")
        if checkpoint.get("stage") != "idle":
            raise ValueError("unresolved partial work; refusing to repeat paid work or effects")
        restored = WorkBudget.restore(checkpoint["budget"])
        budget._counts, budget._spent = restored.counts, restored.spent
        budget.raw = checkpoint["raw"]
        registry = CompetenceRegistry.restore(checkpoint["registry"], grammar=grammar)
        env = CompositeEnvironment.restore(checkpoint["environment"], cases=cases)
        cache = {tuple(k): v for k, v in checkpoint["cache"]}
        rows, learning = checkpoint["rows"], checkpoint["learning"]
        training_cost, exhausted = checkpoint["training_cost"], checkpoint["exhausted"]
        adaptation_rows = checkpoint["adaptation_rows"]
        adaptation_done = checkpoint["adaptation_done"]
        adaptation_learning = checkpoint["adaptation_learning"]

    def save(stage="idle"):
        if checkpoint_path is not None:
            atomic_json(
                Path(checkpoint_path),
                dict(
                    config=config,
                    stage=stage,
                    budget=budget.snapshot(),
                    raw=budget.raw,
                    registry=registry.snapshot(),
                    environment=env.snapshot(),
                    cache=[[list(k), v] for k, v in cache.items()],
                    rows=rows,
                    learning=learning,
                    training_cost=training_cost,
                    exhausted=exhausted,
                    adaptation_rows=adaptation_rows,
                    adaptation_done=adaptation_done,
                    adaptation_learning=adaptation_learning,
                ),
            )

    def adapt_once():
        nonlocal adaptation_done, adaptation_learning, exhausted
        if stratum != "ood" or arm not in ("FULL", "NO_CALL") or adaptation_done or len(rows) < 32:
            return
        save("learning_inflight")
        before = budget.spent
        try:
            if len(adaptation_rows) != 32:
                adaptation_learning = {"status": "incomplete_prefix"}
            else:
                examples = [ExternalExample.from_dict(e) for e in adaptation_rows]
                candidate = registry.propose(examples[:16], budget)
                adaptation_learning = {"status": "search_failed"}
                if candidate is not None:
                    budget.charge("composite_gate", candidate.program.nodes)
                    if is_composite(candidate.program):
                        record = registry.admit(candidate, examples[16:], budget, min_examples=16)
                        adaptation_learning = dict(
                            status="admission", candidate=candidate.to_dict(), record=record.__dict__
                        )
                    else:
                        adaptation_learning = dict(status="not_composite", candidate=candidate.to_dict())
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            adaptation_learning = dict(status="budget" if isinstance(exc, BudgetStop) else "error", error=repr(exc))
            exhausted = True
        adaptation_learning["cost"] = budget.spent - before
        adaptation_done = True
        save()

    for index, (task_id, case) in enumerate(cases.items()):
        if index < len(rows):
            continue
        adapt_once()
        save("task_inflight")
        before = budget.spent
        row = dict(
            task_id=task_id, index=index, phase="prefix" if stratum == "ood" and index < 32 else "suffix", case=case
        )
        try:
            if exhausted:
                row.update(status="unattempted", success=False, applied=False)
            else:
                if stratum == "ood" and index < 32 and arm in ("FULL", "NO_CALL"):
                    teacher_task(env, task_id, index, budget, registry, record=row)
                    adaptation_rows.append(row["adaptation_example"])
                else:
                    row.update(
                        solve_task(
                            env,
                            task_id,
                            arm,
                            budget,
                            registry,
                            cache,
                            random.Random(component_seed(seed, family, stratum, index, "probes")),
                            stratum == "encoding",
                        )
                    )
                row["success"] = score(case, row.get("proposed"), row.get("receipt"))
                if row.get("revision_pending"):
                    exhausted = True
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            row.update(
                status="budget" if isinstance(exc, BudgetStop) else "error",
                error=repr(exc),
                success=False,
                applied=False,
            )
            exhausted = isinstance(exc, BudgetStop)
        row.update(
            cost=budget.spent - before,
            cumulative_cost=budget.spent,
            hard_violation=hard_violation(case, row.get("receipt")),
        )
        rows.append(row)
        save()
        if emit:
            emit(row)
    adapt_once()
    return dict(
        spec=SPEC,
        seed=seed,
        family=family,
        arm=arm,
        limit=limit,
        observation_price=observation_price,
        stratum=stratum,
        denominator=len(cases),
        success=sum(r["success"] for r in rows),
        applications=sum(r["applied"] for r in rows),
        weighted_total=budget.spent,
        accounting_scale=8,
        exact_accounting_total={"numerator": budget.spent, "denominator": 8},
        hard_violations=sum(r["hard_violation"] for r in rows),
        false_applications=sum(bool(r.get("applied")) and not r["success"] for r in rows),
        phases={
            phase: dict(
                denominator=sum(r["phase"] == phase for r in rows),
                success=sum(r["success"] for r in rows if r["phase"] == phase),
            )
            for phase in ("prefix", "suffix")
        },
        raw_vector=budget.raw,
        weighted_vector=budget.counts,
        training_cost=training_cost,
        learning=learning,
        adaptation_learning=adaptation_learning,
        registry=registry.snapshot(),
        manifest=manifest(parts),
        rows=rows,
        wall_seconds=time.monotonic() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_freeze(path):
    """Strict code/protocol/manifest binding; no mere presence flag."""
    frozen = json.loads(path.read_text())
    if frozen.get("spec") != SPEC or frozen.get("status") != "IMPLEMENTATION_FREEZE":
        raise ValueError("wrong freeze spec/status")
    root = Path(__file__).resolve().parents[1]
    required = {
        "benchmarks/experience_transfer.py",
        "benchmarks/_experience_scorer.py",
        "tests/test_experience_transfer.py",
    }
    required.update(str(p.relative_to(root)) for p in (root / "src/solvi/experimental/experience").glob("*.py"))
    if not required <= set(frozen["code_sha256"]):
        raise ValueError("freeze omits implementation source")
    for name, expected_hash in frozen["code_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("source hash mismatch: " + name)
    protocol = Path(frozen["protocol_path"])
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != frozen["protocol_sha256"]:
        raise ValueError("protocol hash mismatch")
    # Only after source/protocol verification create final manifests.
    sealed = {
        f"{seed}:{family}": manifest(partitions(seed, family, "final"))
        for seed in SEEDS
        for family in ("resource", "workflow")
    }
    if digest(sealed) != frozen["manifest_sha256"]:
        raise ValueError("partition manifest hash mismatch")
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--protocol-freeze", type=Path)
    args = parser.parse_args(argv)
    if args.smoke and args.split == "final":
        parser.error("smoke uses dev only; final generation forbidden")
    if not args.smoke:
        if args.protocol_freeze is None:
            parser.error("verified --protocol-freeze required for evaluation")
        verify_freeze(args.protocol_freeze)
        if args.max_seeds is not None or args.tasks is not None:
            parser.error("truncation is allowed only for smoke")
    if (args.tasks is not None and args.tasks < 1) or (args.max_seeds is not None and not 1 <= args.max_seeds <= 12):
        parser.error("positive smoke tasks and 1..12 seeds required")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw.jsonl"
    done, episodes = set(), set()
    if raw.exists():
        if not args.resume:
            parser.error("output exists; use --resume")
        for line in raw.read_text().splitlines():
            value = json.loads(line)
            if value["kind"] == "trial":
                done.add(value["key"])
            elif value["kind"] == "episode":
                episodes.add((value["key"], value["row"]["task_id"]))
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print("heartbeat: experience_transfer running", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    seeds = SEEDS[: args.max_seeds or 1] if args.smoke else SEEDS
    limits, prices = ((65536,), (8,)) if args.smoke else (BUDGETS, PRICES)
    strata = ("dev",) if args.split == "dev" else ("id", "encoding", "ood")
    tasks = args.tasks or 4 if args.smoke else None
    try:
        with raw.open("a") as stream:

            def append(value):
                stream.write(canonical(value) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            for seed in seeds:
                for family in ("resource", "workflow"):
                    shift = (seed - SEEDS[0]) % len(ARMS)
                    for arm in ARMS[shift:] + ARMS[:shift]:
                        for limit in limits:
                            for price in prices:
                                for stratum in strata:
                                    key = digest([SPEC, seed, family, arm, limit, price, stratum, tasks])
                                    if key in done:
                                        continue
                                    chunk = args.output / (key + ".checkpoint.json")
                                    checkpoint = (
                                        json.loads(chunk.read_text()) if args.resume and chunk.exists() else None
                                    )
                                    if checkpoint is not None and checkpoint.get("stage") != "idle":
                                        raise ValueError(
                                            "unresolved checkpoint: interrupted paid task or learning; no automatic replay"
                                        )

                                    def emit(row):
                                        identity = (key, row["task_id"])
                                        if identity not in episodes:
                                            append(dict(kind="episode", key=key, row=row))
                                            episodes.add(identity)

                                    if checkpoint:
                                        for row in checkpoint["rows"]:
                                            emit(row)
                                    result = run_trial(
                                        seed,
                                        family,
                                        arm,
                                        limit,
                                        price,
                                        stratum,
                                        tasks=tasks,
                                        emit=emit,
                                        checkpoint=checkpoint,
                                        checkpoint_path=chunk,
                                    )
                                    append(dict(kind="trial", key=key, result=result))
                                    done.add(key)
                                    atomic_json(args.output / "progress.json", dict(completed=sorted(done), last=key))
                                    print(
                                        f"{family} {arm} {stratum}: {result['success']}/{result['denominator']}",
                                        flush=True,
                                    )
    finally:
        stop.set()


if __name__ == "__main__":
    main()
