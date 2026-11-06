"""Authored finite refinement experiment; host references never enter learner rows.

Run --prepare before evaluation, --split dev, freeze approval, then --split final.
Supply --protocol with the frozen host protocol and, for final, --final-claim
with one stable host path shared across all output directories for this study.
An interrupted claimed trial blocks rather than silently spending a new budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time

from refinement_tasks import tasks
from solvi.experimental.experience._common import BudgetStop, WorkBudget, canonical, digest
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain
from solvi.experimental.experience.observations import ObservationOption
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.refinement import RefinementSession

ARMS = ("SINGLE", "REFINE", "FIXED", "RANDOM", "ACTIVE", "MEASURE", "FORMULA")
SEEDS = {"smoke": [901], "dev": [1901, 1902, 1903], "final": [2901, 2902, 2903, 2904, 2905]}
LIMIT = 100000


def write(path, body):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(body, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_manifest(protocol_path=None):
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "src/solvi/experimental/experience").glob("*.py")),
             Path(__file__).resolve(), Path(__file__).with_name("refinement_tasks.py"), Path(__file__).with_name("refinement_score.py")]
    for name in ("_common", "domain_checks", "programs", "observations", "refinement"):
        module = importlib.import_module("solvi.experimental.experience." + name)
        expected = root / "src/solvi/experimental/experience" / (name + ".py")
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError("foreign actor import origin: " + name)
    files.extend([root / "pyproject.toml", root / "uv.lock"])
    result = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if protocol_path is not None:
        result["host/protocol"] = hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()
    return result


def partition(problem, seed):
    points = list(problem.points)
    random.Random(seed).shuffle(points)
    return dict(initial=points[:2], pool=points[2:10], verification=points[10:18], transfer=points[18:])


class PaidOracle:
    """Only explicitly allowed learning points; receipt cache is per trial."""
    source_id = "bounded-reference"

    def __init__(self, problem, allowed):
        self._reference = problem.reference
        self.source_revision = digest(dict(task=problem.task_id, grammar=problem.grammar.to_dict()))
        self.allowed = {canonical(point) for point in allowed}
        self.receipts = {}

    def read(self, request_id, source_revision, point):
        if source_revision != self.source_revision or canonical(point) not in self.allowed:
            raise ValueError("outside permitted learning boundary")
        if request_id in self.receipts:
            raise ValueError("duplicate external request")
        result = DomainReceipt(request_id, "event:" + request_id, self.source_id, source_revision,
                               digest(point), self._reference(point))
        self.receipts[request_id] = result
        return result

    def reconcile(self, request_id):
        return self.receipts.get(request_id)


def _truth(problem, point):
    try:
        return problem.reference(point)
    except Exception as error:
        raise RuntimeError("host reference failed") from error


def trial(problem, seed, arm):
    split = partition(problem, seed)
    # Only the host retains split['transfer']; the actor sees ten observation
    # inputs and eight verification inputs, and receives no transfer feedback.
    budget = WorkBudget(LIMIT)
    allowed = split["initial"] + split["pool"] + split["verification"]
    oracle = PaidOracle(problem, allowed)
    program = None
    learning = None
    start = time.monotonic()
    if arm in ("FORMULA", "MEASURE"):
        status = "known_formula" if arm == "FORMULA" else "direct_measurement"
        if arm == "FORMULA":
            program = problem.formula
    else:
        output_type = "bool" if problem.family == "boolean_policy" else "int"
        domain = FiniteDomain(problem.task_id, problem.task_id, "typed-v1", problem.grammar,
                              oracle.source_id, oracle.source_revision, split["verification"],
                              budget=budget, result_type=output_type,
                              input_types={n: "bool" if type(allowed[0][n]) is bool else "int"
                                           for n in problem.grammar.inputs})
        options = tuple(ObservationOption(str(i), point, 8)
                        for i, point in enumerate(split["initial"] + split["pool"]))
        mode = {"SINGLE": "none", "REFINE": "none", "FIXED": "fixed",
                "RANDOM": "random", "ACTIVE": "active"}[arm]
        session = RefinementSession(problem.grammar, domain, oracle, options, budget,
                                    session_id=f"{problem.task_id}:{seed}:{arm}", initial=("0", "1"),
                                    mode=mode, seed=seed, max_rounds=1 if arm == "SINGLE" else 6,
                                    committee_size=3, max_observations=3)
        result = session.run()
        if result["status"] == "blocked" or session.state["phase"] != "done":
            raise RuntimeError("systemic/incomplete learning trial; stop matrix")
        status = result["status"] + ":" + result["reason"]
        learning = session.snapshot()
        if result["status"] == "verified":
            program = Program.from_dict(result["program"])
        # Independent audit is outside actor budget, never fed back to actor.
        if program is not None:
            check = learning["checks"][-1]
            if len(check["actuals"]) != len(domain.points) or check["checked_count"] != len(domain.points):
                raise RuntimeError("incomplete verification audit")
            if (check["program_fingerprint"] != program.fingerprint or
                    check["domain_fingerprint"] != domain.fingerprint or
                    result["program"] != learning["candidate"] or len(check["receipts"]) != len(domain.points)):
                raise RuntimeError("verification certificate binding mismatch")
            audit_budget = WorkBudget(1000000)
            for index, actual in enumerate(learning["checks"][-1]["actuals"]):
                expected = _truth(problem, domain.point_dict(index))
                audit_actual = program.evaluate(domain.point_dict(index), audit_budget, problem.grammar)
                if (type(actual) is not type(expected) or actual != expected or
                        type(audit_actual) is not type(expected) or audit_actual != expected):
                    raise RuntimeError("independent verification audit failed")
    # Learning is frozen. All held-out predictions are made without exposing
    # target labels; gold is read only by the evaluator after each prediction.
    predictions = []
    for point in split["transfer"]:
        try:
            if arm == "MEASURE":
                budget.charge("transfer_measurement", 8)
                value = _truth(problem, point)
            elif program is None:
                predictions.append(dict(point=point, outcome="abstain"))
                continue
            else:
                grammar = problem.grammar
                if arm == "FORMULA":
                    grammar = Grammar(grammar.inputs, constants=grammar.constants,
                                      max_nodes=64, max_depth=16, max_abs=grammar.max_abs)
                value = program.evaluate(point, budget, grammar)
            budget.charge("transfer_effect", 8)
            gold = _truth(problem, point)
            predictions.append(dict(point=point, value=value, expected=gold,
                                    outcome="correct" if type(value) is type(gold) and value == gold else "wrong"))
        except BudgetStop:
            predictions.append(dict(point=point, outcome="budget"))
        except ValueError:
            if arm in ("MEASURE", "FORMULA"):
                raise
            predictions.append(dict(point=point, outcome="invalid"))
    return dict(task=problem.task_id, family=problem.family, seed=seed, arm=arm, status=status,
                budget=budget.snapshot(), learning=learning, predictions=predictions,
                external_reads=len(oracle.receipts) + (len(predictions) if arm == "MEASURE" else 0), wall_seconds=f"{time.monotonic() - start:.9f}")


def summarize(path, *, split):
    from refinement_score import summarize as audited_summary
    return audited_summary(path, split=split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=SEEDS, default="smoke")
    parser.add_argument("--protocol", type=Path, required=True, help="host-owned frozen experiment protocol")
    parser.add_argument("--final-claim", type=Path, help="one stable host-owned claim path shared across outputs")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(v=1, sources=source_manifest(args.protocol), seeds=SEEDS, arms=ARMS, limit=LIMIT,
                    tasks=[dict(task=p.task_id, family=p.family, grammar=p.grammar.to_dict(),
                                description=p.description, points=p.points,
                                formula=p.formula.to_dict()) for p in tasks()],
                    partitions={split: {str(seed): {p.task_id: partition(p, seed) for p in tasks()}
                                        for seed in seeds} for split, seeds in SEEDS.items()})
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.output / "manifest.json"
    if args.prepare:
        if manifest_path.exists():
            raise ValueError("manifest already exists")
        write(manifest_path, manifest)
        print("prepared", digest(manifest), flush=True)
        return
    if json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("source/config changed since manifest")
    raw = args.output / (args.split + ".jsonl")
    if args.summarize:
        write(args.output / (args.split + "_summary.json"), summarize(raw, split=args.split))
        return
    if args.split == "final":
        approval = json.loads((args.output / "FINAL_APPROVAL.json").read_text())
        dev = summarize(args.output / "dev.jsonl", split="dev")
        selected = max(("FIXED", "RANDOM"), key=lambda arm: dev["arms"][arm]["correct"])
        if (approval.get("manifest") != digest(manifest) or approval.get("approved") is not True
                or approval.get("comparator") != selected
                or approval.get("dev_raw_sha256") != dev["raw_sha256"]):
            raise ValueError("missing exact final approval")
        if args.final_claim is None:
            raise ValueError("final requires a stable host-owned --final-claim path")
        global_claim = args.final_claim.resolve()
        if (approval.get("final_claim") != str(global_claim)
                or approval.get("output") != str(args.output.resolve())):
            raise ValueError("approved final binding changed")
        global_claim.parent.mkdir(parents=True, exist_ok=True)
        binding = dict(manifest=digest(manifest), output=str(args.output.resolve()))
        if global_claim.exists():
            if json.loads(global_claim.read_text()) != binding:
                raise ValueError("final already claimed with another source/output")
        else:
            with global_claim.open("x") as stream:
                json.dump(binding, stream)
                stream.flush()
                os.fsync(stream.fileno())
    stop = args.output / (args.split + "_STOP.json")
    if stop.exists():
        raise ValueError("persistent STOP requires investigation; no new actors")
    planned = {(p.task_id, seed, arm) for p in tasks() for seed in SEEDS[args.split] for arm in ARMS}
    done = set()
    previous = "0" * 64
    if raw.exists():
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            seal = row.get("seal")
            body = {k: v for k, v in row.items() if k != "seal"}
            if seal != dict(manifest=digest(manifest), previous=previous, sha256=digest(body)):
                raise ValueError("raw checksum/manifest chain mismatch")
            previous = digest(row)
            key = (row["task"], row["seed"], row["arm"])
            if key in done or key not in planned:
                raise ValueError("duplicate/unplanned terminal trial")
            from refinement_score import audit_row
            problem = next(p for p in tasks() if p.task_id == key[0])
            audit_row(row, problem, key[1], key[2])
            done.add(key)
    claims = args.output / (args.split + "_claims")
    claims.mkdir(exist_ok=True)
    claimed = set()
    for claim_path in claims.glob("*.json"):
        claim = json.loads(claim_path.read_text())
        key = tuple(claim["task"])
        if claim["manifest"] != digest(manifest) or claim_path.stem != digest(list(key)) or key in claimed:
            raise ValueError("invalid/duplicate trial claim")
        claimed.add(key)
    if claimed != done:
        write(stop, dict(error="claimed/terminal mismatch", claimed=len(claimed), terminal=len(done)))
        raise ValueError("interrupted trial or unclaimed raw; no automatic actor retry")
    total = len(SEEDS[args.split]) * len(tasks()) * len(ARMS)
    for seed in SEEDS[args.split]:
        for problem in tasks():
            for arm in ARMS:
                key = (problem.task_id, seed, arm)
                if key in done:
                    continue
                claim = claims / (digest(list(key)) + ".json")
                with claim.open("x") as stream:
                    json.dump(dict(task=key, manifest=digest(manifest)), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    row = trial(problem, seed, arm)
                    from refinement_score import audit_row
                    audit_row(row, problem, seed, arm)
                except Exception as error:
                    write(args.output / (args.split + "_STOP.json"),
                          dict(task=list(key), error=repr(error), manifest=digest(manifest)))
                    raise
                row["seal"] = dict(manifest=digest(manifest), previous=previous, sha256=digest(row))
                with raw.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                previous = digest(row)
                done.add(key)
                print(f"{len(done)}/{total} {key} {row['status']} units={row['budget']['spent']}", flush=True)
    write(args.output / (args.split + "_summary.json"), summarize(raw, split=args.split))


if __name__ == "__main__":
    main()
