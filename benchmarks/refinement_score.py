"""Read-only audit of the frozen finite refinement matrix, without actor or oracle calls."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from refinement_bench import ARMS, SEEDS, partition
from refinement_tasks import tasks
from solvi.experimental.experience._common import BudgetStop, WorkBudget, canonical, digest
from solvi.experimental.experience.domain_checks import FiniteDomain
from solvi.experimental.experience.programs import Grammar, Program


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(left, right):
    return canonical(left) == canonical(right)


def _verified(state, problem, domain, work):
    program = Program.from_dict(state["result"]["program"])
    check = state["checks"][-1]
    _require(_same(state["candidate"], program.to_dict()), "candidate/result disagreement")
    _require(
        check["status"] == "verified"
        and check["reason"] == "complete_finite_domain"
        and check["program_fingerprint"] == program.fingerprint
        and check["grammar_fingerprint"] == problem.grammar.fingerprint
        and check["domain_fingerprint"] == domain.fingerprint,
        "certificate binding",
    )
    n = len(domain.points)
    _require(
        type(check["checked_count"]) is int
        and check["checked_count"] == n
        and len(check["receipts"]) == len(check["actuals"]) == n,
        "incomplete certificate",
    )
    evidence = []
    for i, receipt in enumerate(check["receipts"]):
        point = domain.point_dict(i)
        actual = program.evaluate(point, work, problem.grammar)
        expected = problem.reference(point)
        request = f"{state['config']['session_id']}:round:{state['rounds']}:{i}"
        _require(
            _same(state["receipts"].get(request), receipt)
            and receipt["request_id"] == request
            and receipt["input_digest"] == digest(point)
            and _same(receipt["expected"], expected)
            and _same(actual, expected)
            and _same(check["actuals"][i], actual),
            "verification AST/receipt",
        )
        evidence.append([digest(point), actual, receipt])
    proof = digest(
        dict(
            program=program.fingerprint,
            grammar=problem.grammar.fingerprint,
            domain=domain.fingerprint,
            evidence=evidence,
        )
    )
    _require(check["proof_id"] == proof, "verification proof digest")
    return program


def _learning(row, problem, roles, work):
    state = row["learning"]
    ledger = WorkBudget.restore(state["ledger"])
    revision = digest(dict(task=problem.task_id, grammar=problem.grammar.to_dict()))
    allowed = roles["initial"] + roles["pool"]
    domain = FiniteDomain(
        problem.task_id,
        problem.task_id,
        "typed-v1",
        problem.grammar,
        "bounded-reference",
        revision,
        roles["verification"],
        budget=work,
        input_types={k: type(v).__name__ for k, v in allowed[0].items()},
        result_type="bool" if problem.family == "boolean_policy" else "int",
    )
    arm, seed = row["arm"], row["seed"]
    session = f"{problem.task_id}:{seed}:{arm}"
    cfg = state["config"]
    _require(
        cfg["session_id"] == session
        and cfg["grammar"] == problem.grammar.fingerprint
        and cfg["domain"] == domain.fingerprint
        and _same(cfg["seed"], seed)
        and cfg["mode"]
        == {"ACTIVE": "active", "FIXED": "fixed", "RANDOM": "random", "SINGLE": "none", "REFINE": "none"}[arm]
        and _same(cfg["max_rounds"], 1 if arm == "SINGLE" else 6)
        and _same(cfg["options"], [dict(name=str(i), inputs=p, cost=8) for i, p in enumerate(allowed)])
        and cfg["initial"] == ["0", "1"]
        and _same(cfg["max_observations"], 3)
        and _same(cfg["committee_size"], 3),
        "learning config mismatch",
    )
    outcome = state["result"]
    _require(
        ledger.limit == 100000
        and state["phase"] == "done"
        and state["pending"] is None
        and outcome["domain"] == domain.fingerprint
        and state["status"] == outcome["status"]
        and state["reason"] == outcome["reason"]
        and row["status"] == state["status"] + ":" + state["reason"]
        and _same(outcome["rounds"], state["rounds"])
        and type(state["rounds"]) is int
        and 0 <= state["rounds"] <= cfg["max_rounds"],
        "inconsistent terminal state",
    )
    ver = [domain.point_dict(i) for i in range(len(domain.points))]
    receipts = state["receipts"]
    observation_reads = 0
    for rid, receipt in receipts.items():
        prefix = session + ":"
        _require(rid.startswith(prefix), "foreign request")
        parts = rid[len(prefix) :].split(":")
        if len(parts) == 2 and parts[0] == "observation" and parts[1] in {str(i) for i in range(len(allowed))}:
            point = allowed[int(parts[1])]
            observation_reads += 1
        else:
            _require(
                len(parts) == 3
                and parts[0] == "round"
                and parts[2] in {str(i) for i in range(len(ver))}
                and parts[1] in {str(i) for i in range(1, state["rounds"] + 1)},
                "invalid verification request",
            )
            point = ver[int(parts[2])]
        _require(
            _same(
                receipt,
                dict(
                    request_id=rid,
                    event_id="event:" + rid,
                    source_id="bounded-reference",
                    source_revision=revision,
                    input_digest=digest(point),
                    expected=problem.reference(point),
                ),
            ),
            "receipt point/source/truth",
        )
    _require(
        ledger.counts.get("refinement_observation", 0) == 8 * observation_reads
        and ledger.counts.get("domain_oracle", 0) == 8 * (len(receipts) - observation_reads),
        "unpaid oracle read",
    )
    _require(
        ledger.counts.get("domain_manifest", 0) == work.counts["domain_manifest"],
        "unpaid verification domain preparation",
    )
    origins = Counter(e["origin"] for e in state["examples"])
    _require(
        origins["initial"] == 2
        and origins["observation"] == state["observations"]
        and 0 <= state["observations"] <= (3 if arm in ("ACTIVE", "FIXED", "RANDOM") else 0)
        and origins["initial"] + origins["observation"] == observation_reads,
        "unexpected training/observation count",
    )
    witnesses = {
        check["receipts"][-1]["request_id"]
        for check in state["checks"]
        if check["reason"] == "counterexample"
        and check["receipts"]
        and _same(check["witness"]["expected"], check["receipts"][-1]["expected"])
        and not _same(check["witness"]["actual"], check["witness"]["expected"])
        and check["witness"]["request_id"] == check["receipts"][-1]["request_id"]
    }
    seen = set()
    for example in state["examples"]:
        origin, point, receipt = example["origin"], example["inputs"], example["receipt"]
        rid = receipt["request_id"]
        permitted = {"initial": roles["initial"], "observation": roles["pool"], "counterexample": ver}[origin]
        _require(
            digest(point) not in seen
            and any(_same(point, p) for p in permitted)
            and _same(receipts.get(rid), receipt)
            and receipt["input_digest"] == digest(point)
            and _same(example["expected"], receipt["expected"])
            and ((rid in witnesses) if origin == "counterexample" else rid.startswith(session + ":observation:")),
            "training origin/point/receipt",
        )
        seen.add(digest(point))
    if row["status"] == "verified:complete_verification_domain":
        program = _verified(state, problem, domain, work)
    else:
        _require(
            state["status"] == "unknown"
            and state["reason"]
            in {
                "budget",
                "no_examples",
                "round_limit",
                "candidate_limit",
                "exhausted",
                "repeated_counterexample",
                "invalid_program",
            },
            "unexpected terminal status",
        )
        program = None
    return program, ledger, len(receipts)


def audit_row(row, problem, seed, arm):
    """Validate one designated trial. Return independently recomputed outcome counts and audit units."""
    _require(
        row["task"] == problem.task_id
        and row["family"] == problem.family
        and _same(row["seed"], seed)
        and row["arm"] == arm,
        "unexpected trial identity/family",
    )
    roles = partition(problem, seed)
    final = WorkBudget.restore(row["budget"])
    _require(final.limit == 100000, "full budget limit")
    work = WorkBudget(10**12)
    if arm in ("MEASURE", "FORMULA"):
        _require(
            row["learning"] is None
            and row["status"] == ("known_formula" if arm == "FORMULA" else "direct_measurement"),
            "bad control status",
        )
        program, ledger, reads = problem.formula, WorkBudget(100000), 0
    else:
        program, ledger, reads = _learning(row, problem, roles, work)
    grammar = problem.grammar
    if arm == "FORMULA":
        grammar = Grammar(
            grammar.inputs, constants=grammar.constants, max_abs=grammar.max_abs, max_depth=16, max_nodes=64
        )
    _require(len(row["predictions"]) == len(roles["transfer"]) == 14, "transfer denominator")
    start = ledger.spent
    outcomes = Counter()
    for point, given in zip(roles["transfer"], row["predictions"], strict=True):
        predicted = dict(point=point)
        try:
            if arm == "MEASURE":
                ledger.charge("transfer_measurement", 8)
                value = problem.reference(point)
                reads += 1
            elif program is None:
                predicted["outcome"] = "abstain"
            else:
                try:
                    value = program.evaluate(point, ledger, grammar)
                except ValueError:
                    predicted["outcome"] = "invalid"
            if "outcome" not in predicted:
                ledger.charge("transfer_effect", 8)
                expected = problem.reference(point)
                predicted.update(
                    value=value, expected=expected, outcome="correct" if _same(value, expected) else "wrong"
                )
        except BudgetStop:
            predicted["outcome"] = "budget"
        _require(_same(given, predicted), "transfer point/answer/outcome")
        outcomes[predicted["outcome"]] += 1
    _require(_same(ledger.snapshot(), final.snapshot()), "terminal transfer ledger mismatch")
    _require(_same(row["external_reads"], reads), "external read count")
    return outcomes, work.spent + ledger.spent - start


def summarize(path, *, split):
    """Require the full preassigned matrix before returning aggregate results."""
    seeds = SEEDS[split]
    problems = {p.task_id: p for p in tasks()}
    expected = {(p, s, a) for p in problems for s in seeds for a in ARMS}
    seen, grouped = set(), {}
    audit_units = 0
    path = Path(path)
    checksum = hashlib.sha256()
    manifest_path = path.parent / "manifest.json"
    manifest_hash = digest(json.loads(manifest_path.read_text())) if manifest_path.exists() else None
    previous = "0" * 64
    claims = path.parent / (split + "_claims")
    _require(not (path.parent / (split + "_STOP.json")).exists(), "persistent STOP forbids scoring")
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            checksum.update(raw)
            try:
                row = json.loads(raw)
                key = (row["task"], row["seed"], row["arm"])
                _require(key in expected and key not in seen and type(row["seed"]) is int, "duplicate/foreign trial")
                if manifest_hash is not None or "seal" in row:
                    _require(manifest_hash is not None, "sealed raw without frozen manifest")
                    body = {k: v for k, v in row.items() if k != "seal"}
                    _require(
                        row.get("seal") == dict(manifest=manifest_hash, previous=previous, sha256=digest(body)),
                        "terminal seal/chain mismatch",
                    )
                    previous = digest(row)
                    claim = claims / (digest(list(key)) + ".json")
                    _require(
                        claim.is_file()
                        and _same(json.loads(claim.read_text()), dict(task=list(key), manifest=manifest_hash)),
                        "missing or mismatched claim",
                    )
                task, seed, arm = key
                outcomes, units = audit_row(row, problems[task], seed, arm)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                raise ValueError(f"invalid raw line {number}: {error}") from error
            seen.add(key)
            audit_units += units
            group = grouped.setdefault(
                arm,
                dict(
                    trials=0,
                    assigned=0,
                    correct=0,
                    wrong=0,
                    abstain=0,
                    budget=0,
                    invalid=0,
                    units=0,
                    reads=0,
                    verified=0,
                    families={},
                    mechanisms={},
                ),
            )
            group["trials"] += 1
            group["assigned"] += sum(outcomes.values())
            group["verified"] += int(row["status"].startswith("verified:"))
            group["units"] += row["budget"]["spent"]
            group["reads"] += row["external_reads"]
            for status, count in outcomes.items():
                group[status] += count
            for space, name in [("families", row["family"]), ("mechanisms", task)]:
                sub = group[space].setdefault(name, dict(assigned=0, correct=0))
                sub["assigned"] += sum(outcomes.values())
                sub["correct"] += outcomes["correct"]
    _require(seen == expected, f"missing {len(expected - seen)} mandatory trials")
    if manifest_hash is not None:
        _require(
            {p.stem for p in claims.glob("*.json")} == {digest(list(key)) for key in seen}, "foreign/unsettled claim"
        )
    return dict(trials=len(seen), arms=grouped, raw_sha256=checksum.hexdigest(), audit_units=audit_units)
