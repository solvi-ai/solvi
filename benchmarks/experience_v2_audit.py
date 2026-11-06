"""Scorer-only independent finite truth, AST execution and durable effect audit.

Pinned source bytes and paid traces detect inconsistencies in a trusted-host
record. They do not authenticate history against a host rewriting every artifact.
"""

from __future__ import annotations
import itertools
import hashlib
import json
from pathlib import Path
import sqlite3

from experience_v2 import Wire, roles
from solvi.experimental.experience._common import canonical, digest
from solvi.experimental.experience import _common

_AXES = {"inventory": (range(5),) * 3, "eligibility": (range(5), range(5), (False, True)), "schedule": (range(4),) * 4}


_SOURCE_SQL = {
    "inventory": "SELECT max(0, ? - ? - ?)",
    "eligibility": "SELECT (? < ?) AND NOT ?",
    "schedule": "SELECT max(? + ?, ?) + ?",
}


def same(left, right):
    return type(left) is type(right) and left == right


def truth(family, row):
    if family == "eligibility":
        if type(row[2]) is not bool or any(type(v) is not int or not 0 <= v <= 4 for v in row[:2]):
            raise ValueError("invalid eligibility input")
        return row[0] < row[1] and not row[2]
    bound = 4 if family == "inventory" else 3
    if any(type(v) is not int or not 0 <= v <= bound for v in row):
        raise ValueError("invalid numeric input")
    if family == "inventory":
        stock = [object() for _ in range(row[0])]
        for _ in range(row[1] + row[2]):
            if stock:
                stock.pop()
        return len(stock)
    if family != "schedule":
        raise ValueError("unknown family")
    # Independent tick simulation: a,b sequential; c parallel; then d.
    sequential, parallel, elapsed = row[0] + row[1], row[2], 0
    while sequential or parallel:
        sequential = max(0, sequential - 1)
        parallel = max(0, parallel - 1)
        elapsed += 1
    for _ in range(row[3]):
        elapsed += 1
    return elapsed


def evaluate(ast, inputs, grammar, work):
    """Strict independent eager interpreter. No Program/learner calls."""
    arity = {"add": 2, "sub": 2, "mul": 2, "min": 2, "max": 2, "lt": 2, "eq": 2, "and": 2, "or": 2, "not": 1, "if": 3}
    nodes = 0

    def scalar(v):
        if type(v) not in (int, bool) or (type(v) is int and abs(v) > grammar["max_abs"]):
            raise ValueError("invalid scalar/overflow")
        return v

    def visit(node, depth):
        nonlocal nodes
        work[0] += 1
        nodes += 1
        if depth > grammar["max_depth"] or nodes > grammar["max_nodes"] or type(node) is not list or not node:
            raise ValueError("invalid AST shape/bounds")
        op = node[0]
        if op in ("input", "const"):
            if len(node) != 2:
                raise ValueError("invalid terminal")
            if op == "input":
                if node[1] not in grammar["inputs"]:
                    raise ValueError("unknown input")
                return scalar(inputs[node[1]])
            if type(node[1]) is int and node[1] not in grammar["constants"]:
                raise ValueError("unknown constant")
            return scalar(node[1])
        if op not in grammar["operators"] or op not in arity or len(node) != arity[op] + 1:
            raise ValueError("invalid operator")
        values = [visit(child, depth + 1) for child in node[1:]]
        if op in ("add", "sub", "mul", "min", "max", "lt") and any(type(v) is not int for v in values):
            raise ValueError("numeric scalar type")
        if op in ("and", "or", "not") and any(type(v) is not bool for v in values):
            raise ValueError("boolean scalar type")
        if op == "eq" and type(values[0]) is not type(values[1]):
            raise ValueError("equality scalar type")
        if op == "if" and (type(values[0]) is not bool or type(values[1]) is not type(values[2])):
            raise ValueError("conditional scalar type")
        a = values[0]
        b = values[1] if len(values) > 1 else None
        if op == "add":
            value = a + b
        elif op == "sub":
            value = a - b
        elif op == "mul":
            value = a * b
        elif op == "min":
            value = min(values)
        elif op == "max":
            value = max(values)
        elif op == "lt":
            value = a < b
        elif op == "eq":
            value = a == b
        elif op == "and":
            value = a and b
        elif op == "or":
            value = a or b
        elif op == "not":
            value = not a
        else:
            value = b if a else values[2]
        return scalar(value)

    return visit(ast, 1)


def audit_trial(directory, result):
    """Return discrepancies without altering raw artifacts or trusting flags."""
    errors, work = [], [0]
    output = dict(errors=errors, correct_R=0, correct_H=0, useful_R=0, useful_H=0, verification_work=0)

    def require(ok, message):
        work[0] += 1
        if not ok:
            errors.append(message)
        return ok

    try:
        directory = Path(directory)
        family, seed, arm = result["family"], result["seed"], result["arm"]
        codec = Wire(seed, family, result["wire"])
        groups = roles(seed, family)
        assigned = {
            codec.identity("task", role + ":" + str(i)): (role, row)
            for role in ("transfer_R", "transfer_H")
            for i, row in enumerate(groups[role])
        }
        entries = result["transfers"]
        require(
            len(entries) == 32 and {e["task_id"] for e in entries} == set(assigned),
            "assigned transfer identities/count",
        )
        if result.get("technical_censor") and not (directory / "environment.db").exists():
            output["verification_work"] = work[0]
            return output
        with sqlite3.connect((directory / "environment.db").resolve().as_uri() + "?mode=ro", uri=True) as conn:
            tasks = {
                t: (f, json.loads(p), e)
                for t, f, p, e in conn.execute("SELECT task_id,family,inputs,effects FROM tabular_tasks")
            }
            raw_receipts = [
                (a, json.loads(i), json.loads(r))
                for a, i, r in conn.execute("SELECT attempt,intent,receipt FROM tabular_receipts")
            ]
        receipts = {r["event_id"]: (intent, r) for _, intent, r in raw_receipts}
        require(len(receipts) == len(raw_receipts), "duplicate SQL receipt event IDs")
        successes = {}
        for attempt, intent, receipt in raw_receipts:
            task = receipt["task_id"]
            require(
                receipt["attempt_id"] == attempt and intent["task_id"] == task and receipt["kind"] == intent["kind"],
                "SQL intent receipt identity",
            )
            require(receipt["source"] == "tabular" and receipt["mechanism"] == "v1", "SQL source/version")
            f, inputs, effects = tasks[task]
            expected = truth(f, [inputs[n] for n in codec.names])
            if receipt["kind"] == "act":
                proposed = intent["payload"]["answer"]
                correct = same(proposed, expected)
                success = correct and successes.get(task, 0) == 0
                require(same(receipt["value"]["answer"], expected), "SQL observed answer differs from simulator")
                require(same(receipt["value"]["ok"], success), "SQL receipt ok differs from actual transition")
                successes[task] = successes.get(task, 0) + int(success)
            else:
                require(same(receipt["value"]["value"], expected), "SQL observation differs from simulator")
        for task, (_, _, effects) in tasks.items():
            require(
                type(effects) is int and effects == successes.get(task, 0) and effects in (0, 1),
                "SQL effects not exactly once: " + task,
            )
        events = (
            [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
            if (directory / "events.jsonl").exists()
            else []
        )
        registry = result["registry"]
        search_ok = False
        if arm == "NEW_CHECKED" and (result.get("search") or {}).get("reason") == "found":
            search_errors = len(errors)
            trace_events = [e for e in events if e["kind"] in ("search_intent", "search_receipt", "search_terminal")]
            valid_shape = require(
                [e["kind"] for e in trace_events] == ["search_intent", "search_receipt", "search_terminal"],
                "missing paid actual synthesis trace",
            )
            search = result["search"]
            require(type(search.get("tried")) is int and search["tried"] > 0, "missing positive synthesis tried count")
            if valid_shape:
                begin, paid_result, terminal = [e["payload"] for e in trace_events]
                expected_engine = hashlib.sha256(
                    Path(_common.__file__).with_name("programs.py").read_bytes()
                ).hexdigest()
                require(
                    begin["engine_sha256"]
                    == paid_result["engine_sha256"]
                    == terminal["engine_sha256"]
                    == expected_engine,
                    "unpinned synthesis engine",
                )
                require(
                    type(paid_result["tried"]) is int
                    and paid_result["tried"] > 0
                    and paid_result["tried"] == search["tried"] == terminal["tried"],
                    "synthesis tried trace mismatch",
                )
                require(
                    paid_result["counts"].get("search_generate", 0) >= paid_result["tried"]
                    and result["counts"].get("search_generate", 0) >= paid_result["tried"],
                    "unpaid synthesis generation",
                )
                require(
                    paid_result["end"] - paid_result["start"] == sum(paid_result["counts"].values())
                    and trace_events[0]["spent"] == paid_result["start"]
                    and trace_events[1]["spent"] == paid_result["end"] + 1
                    and trace_events[2]["spent"] == paid_result["end"] + 2,
                    "synthesis debit trace mismatch",
                )
                require(
                    terminal["receipt_digest"] == digest(paid_result)
                    and terminal["reason"] == paid_result["reason"] == search["reason"]
                    and canonical(paid_result["ast"]) == canonical(search["ast"]),
                    "synthesis AST receipt mismatch",
                )
                require(
                    canonical(begin["proposal"]) == canonical(paid_result["proposal"]),
                    "synthesis proposal trace mismatch",
                )
                for example in begin["proposal"]:
                    observed = receipts.get(example["event_id"])
                    require(
                        observed is not None
                        and observed[1]["task_id"] == example["task_id"]
                        and observed[1]["kind"] == "observe"
                        and same(observed[1]["value"]["value"], example["expected"])
                        and canonical(tasks[example["task_id"]][1])
                        == canonical(dict(zip(codec.names, codec.decode(example["inputs"])))),
                        "synthesis actual proposal linkage",
                    )
            search_ok = len(errors) == search_errors
        grammar = registry["grammar"] if "grammar" in registry else None
        if grammar is None:
            # Grammar is pinned by the r2 contract, not inferred from AST content.
            from experience_v2 import FAMILIES

            grammar = dict(
                inputs=["w" + str(i) for i in range(len(codec.names))] if codec.enabled else list(codec.names),
                constants=[0, 1],
                operators=list(FAMILIES[family][2]),
                max_depth=4,
                max_nodes=7,
                max_abs=32,
                max_candidates=10000,
            )
        checked_ok = set()
        for cid, checked in registry.get("checked", {}).items():
            verified = checked.get("result")
            if not verified or verified["status"] != "verified":
                continue
            start_errors = len(errors)
            candidate = registry["candidates"][cid]
            program = candidate["program"]
            require(
                digest(candidate) == cid and digest(program["ast"]) == program["fingerprint"], "candidate/program hash"
            )
            binding, record = checked["binding"], checked["result"]
            domain = binding["domain"]
            expected_revision = "rule:" + digest(
                [family, list(codec.names), 3 if family == "schedule" else 4, _SOURCE_SQL[family], "v1"]
            )
            require(domain["source_revision"] == expected_revision, "unpinned source revision")
            require(
                candidate["grammar_fingerprint"] == digest(grammar)
                and record["grammar_fingerprint"] == digest(grammar)
                and domain["grammar_fingerprint"] == digest(grammar),
                "grammar binding hash",
            )
            require(
                record["program_fingerprint"] == program["fingerprint"]
                and record["domain_fingerprint"] == domain["fingerprint"],
                "checked result fingerprints",
            )
            domain_body = {k: v for k, v in domain.items() if k != "fingerprint"}
            require(digest(domain_body) == domain["fingerprint"], "domain hash")
            expected_points = {canonical(codec.encode(list(row))) for row in itertools.product(*_AXES[family])}
            actual_points = domain["points"]
            require(
                len(actual_points) == len(expected_points) and {canonical(p) for p in actual_points} == expected_points,
                "incomplete/duplicate domain",
            )
            require(
                binding["candidate_id"] == cid
                and domain["scope"] == family
                and domain["mechanism"] == "v1"
                and domain["schema"] == family + "-v1"
                and domain["source_id"] == "sql:" + family,
                "checked context/source binding",
            )
            require(
                record["checked_count"] == len(expected_points)
                and len(record["receipts"]) == len(expected_points)
                and len(record["actuals"]) == len(expected_points),
                "checked coverage counts",
            )
            evidence = []
            for i, (point, receipt, actual) in enumerate(zip(actual_points, record["receipts"], record["actuals"])):
                expected = truth(family, codec.decode(point))
                interpreted = evaluate(program["ast"], point, grammar, work)
                require(
                    same(expected, actual) and same(expected, receipt["expected"]) and same(expected, interpreted),
                    "checked wrong AST/actual/oracle",
                )
                require(
                    receipt["request_id"] == cid + ":" + str(i)
                    and receipt["input_digest"] == digest(point)
                    and receipt["source_id"] == domain["source_id"]
                    and receipt["source_revision"] == domain["source_revision"],
                    "checked receipt source/prefix/input",
                )
                evidence.append([digest(point), actual, receipt])
            require(
                record["proof_id"]
                == digest(
                    dict(
                        program=program["fingerprint"],
                        grammar=record["grammar_fingerprint"],
                        domain=domain["fingerprint"],
                        evidence=evidence,
                    )
                ),
                "domain proof hash",
            )
            require(
                checked["registry_proof_id"]
                == digest(
                    dict(binding=binding, batch_id=registry["admissions"][cid]["batch_id"], proof_id=record["proof_id"])
                ),
                "registry proof hash",
            )
            require(
                result["oracle_calls"] >= len(expected_points)
                and result["counts"].get("domain_oracle", 0) >= 8 * len(expected_points),
                "unpaid full oracle",
            )
            reads = [e["payload"] for e in events if e["kind"] == "domain_read_receipt"]
            intents = [e["payload"] for e in events if e["kind"] == "domain_read_intent"]
            require(
                len(reads) == len(record["receipts"]) and canonical(reads) == canonical(record["receipts"]),
                "domain receipt event coverage",
            )
            require(
                len(intents) == len(reads)
                and all(e["request_id"] == cid + ":" + str(i) for i, e in enumerate(intents)),
                "domain intent prefix coverage",
            )
            for point, intent in zip(actual_points, intents):
                require(
                    intent["source_id"] == domain["source_id"]
                    and intent["source_revision"] == domain["source_revision"]
                    and intent["input_digest"] == digest(point)
                    and canonical(intent["point"]) == canonical(point),
                    "domain intent source/input",
                )
            event_ids = [r["event_id"] for r in record["receipts"]]
            require(len(event_ids) == len(set(event_ids)), "duplicate domain receipt events")
            for split, examples in [
                ("proposal", candidate["proposal_examples"]),
                ("admission", [registry["events"][eid] for eid in registry["admissions"][cid]["event_ids"]]),
            ]:
                wanted_tasks = {codec.identity("task", split + ":" + str(i)) for i in range(8)}
                require(
                    len(examples) == 8 and {e["task_id"] for e in examples} == wanted_tasks, "training roles/identity"
                )
                for example in examples:
                    observed = receipts.get(example["event_id"])
                    require(
                        observed is not None
                        and observed[1]["kind"] == "observe"
                        and observed[1]["task_id"] == example["task_id"]
                        and example["split"] == split,
                        "training external receipt linkage",
                    )
                    target = truth(family, codec.decode(example["inputs"]))
                    require(
                        same(example["expected"], target)
                        and same(evaluate(program["ast"], example["inputs"], grammar, work), target),
                        "training AST/label mismatch",
                    )
            if len(errors) == start_errors:
                checked_ok.add(cid)
        sql_act_events = {
            r["event_id"] for _, intent, r in raw_receipts if intent["kind"] == "act" and r["task_id"] in assigned
        }
        represented_events = {e["receipt"]["event_id"] for e in entries if e.get("receipt") is not None}
        require(sql_act_events == represented_events, "physical SQL act/transfer receipt bijection")
        used_events, used_attempts = set(), set()
        for entry in entries:
            task = entry["task_id"]
            if task not in assigned:
                continue
            role, row = assigned[task]
            suffix = "R" if role == "transfer_R" else "H"
            require(entry["role"] == role, "transfer role mismatch")
            f, inputs, effects = tasks[task]
            require(
                f == family and canonical(inputs) == canonical(dict(zip(codec.names, row))), "SQL task input mismatch"
            )
            received = entry.get("receipt")
            if received is None:
                continue
            stored = receipts.get(received["event_id"])
            if not require(
                stored is not None and canonical(stored[1]) == canonical(received), "result receipt does not match SQL"
            ):
                continue
            intent, sql_receipt = stored
            unique = received["event_id"] not in used_events and received["attempt_id"] not in used_attempts
            used_events.add(received["event_id"])
            used_attempts.add(received["attempt_id"])
            if not require(
                unique and received["task_id"] == task and intent["task_id"] == task,
                "duplicate/cross-task transfer receipt",
            ):
                continue
            expected = truth(family, row)
            correct = intent["kind"] == "act" and same(intent["payload"]["answer"], expected) and effects == 1
            if correct:
                output["correct_" + suffix] += 1
            aid, cid = entry.get("application_id"), entry.get("candidate_id")
            if not aid:
                continue
            recovered = entry.get("technical_reconciliation") is True and entry.get("status") == "committed_unlinked"
            if recovered:
                # Count the independently checked physical action above, but an
                # interrupted paid revision cannot supply useful-program credit.
                continue
            receipt_linked = require(
                any(
                    e["kind"] == "application_receipt"
                    and e["payload"].get("task_id") == task
                    and e["payload"].get("application_id") == aid
                    and e["payload"].get("event_id") == received["event_id"]
                    for e in events
                ),
                "missing application receipt event linkage",
            )
            revision_linked = require(
                any(
                    e["kind"] == "registry_revision"
                    and e["payload"].get("application_id") == aid
                    and e["payload"].get("event_id") == received["event_id"]
                    for e in events
                ),
                "missing registry revision event linkage",
            )
            app = registry["applications"].get(aid)
            candidate = registry["candidates"].get(cid)
            linked = (
                require(app is not None and candidate is not None, "missing application/candidate")
                and receipt_linked
                and revision_linked
            )
            if not linked:
                continue
            linked &= require(
                digest(app) == aid
                and digest(candidate) == cid
                and app["competence_id"] == cid
                and app["task_id"] == task
                and canonical(app["inputs"]) == canonical(codec.encode(row)),
                "application identity/input binding",
            )
            predicted = evaluate(candidate["program"]["ast"], codec.encode(row), grammar, work)
            linked &= require(
                same(app["prediction"], predicted) and same(intent["payload"]["answer"], predicted),
                "application prediction differs from AST/SQL intent",
            )
            if (
                arm == "NEW_CHECKED"
                and correct
                and linked
                and cid in checked_ok
                and search_ok
                and result.get("search", {}).get("reason") == "found"
            ):
                linked &= require(
                    canonical(result["search"]["ast"]) == canonical(candidate["program"]), "search candidate linkage"
                )
                linked &= require(
                    entry["proof_id"] == registry["checked"][cid]["registry_proof_id"], "application proof linkage"
                )
                if linked:
                    output["useful_" + suffix] += 1
    except Exception as error:  # noqa: BLE001 - retain every audit failure without discarding raw trial artifacts
        errors.append(type(error).__name__ + ": " + str(error))
    output["verification_work"] = work[0]
    return output
