"""Read-only external scorer. No actor, verifier, Program or scheduler imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random

FAMILIES = ("reconcile", "invoice", "inventory")
ARMS = ("DIRECT", "DIRTY", "FIFO", "CHEAP", "DISPATCH", "NO_CACHE", "NO_LEARN")
LAYOUTS = {
    "reconcile": [("fold", "blend", "a", "b"), ("total", "add", "c", "d"),
                  ("delta", "sub", "fold", "total"), ("amount", "max", "delta", "e"),
                  ("flag", "lt", "amount", "f")],
    "invoice": [("rate", "blend", "b", "c"), ("line", "mul", "a", "rate"),
                ("discount", "add", "d", "e"), ("amount", "sub", "line", "discount"),
                ("flag", "lt", "amount", "f")],
    "inventory": [("reserved", "blend", "c", "d"), ("stock", "add", "a", "b"),
                  ("available", "sub", "stock", "reserved"), ("amount", "min", "available", "e"),
                  ("flag", "lt", "amount", "f")],
}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _same(a, b):
    return type(a) is type(b) and a == b


def _operation(kind, values):
    x, y = values
    if type(x) is not int or type(y) is not int:
        raise ValueError("reference expects integer ports")
    if kind == "blend":
        return x + pow(y, 2)
    if kind == "add":
        return sum((x, y))
    if kind == "sub":
        return sum((x, -y))
    if kind == "mul":
        return x * y
    if kind == "min":
        return sorted((x, y))[0]
    if kind == "max":
        return sorted((x, y))[-1]
    if kind == "lt":
        return x < y
    if kind == "eq":
        return x == y
    raise ValueError("unknown operation")


def gold(family, inputs):
    """Independent end formulas, not a replay through the actor's graph."""
    a, b, c, d, e, f = (inputs[k] for k in "abcdef")
    if family == "reconcile":
        amount = max(a + b * b - c - d, e)
    elif family == "invoice":
        amount = a * (b + c * c) - d - e
    elif family == "inventory":
        amount = min(a + b - c - d * d, e)
    else:
        raise ValueError("unknown family")
    return {"amount": amount, "flag": amount < f}


def fixture(seed, family):
    """Separately reconstruct roles, ordering and updates from the protocol."""
    rng = random.Random(seed)
    inputs = {k: rng.randrange(4) for k in "abcdef"}
    current, updates = dict(inputs), []
    for epoch in range(12):
        changes = {}
        if epoch not in (0, 4, 9):
            k = rng.choice(tuple(current))
            current[k] = (current[k] + rng.randrange(1, 4)) % 4
            changes[k] = current[k]
        updates.append({"epoch": epoch, "changes": changes,
                        "revision": {"kind": "add", "revision": "r2"} if epoch == 6 else None})
    layout = [*LAYOUTS[family], ("noise", "mul", "e", "f")]
    return {"spec": {"input_names": list("abcdef"), "nodes": [
        {"name": n, "kind": k, "deps": [x, y]} for n, k, x, y in layout],
        "outputs": ["amount", "flag"]}, "initial_inputs": inputs, "updates": updates}


def audit(row):
    """Validate raw ledger/events, then score every assigned output independently."""
    if row["family"] not in FAMILIES or row["arm"] not in ARMS:
        raise ValueError("unknown trial role")
    if type(row["seed"]) is not int or type(row["limit"]) is not int or row["limit"] <= 0:
        raise ValueError("invalid trial identity")
    expected = fixture(row["seed"], row["family"])
    for key, value in expected.items():
        if _encoded(row[key]) != _encoded(value):
            raise ValueError("fixture/role mismatch: " + key)
    ledger = row["ledger"]
    if set(ledger) != {"v", "limit", "spent", "counts", "digest"} or ledger["limit"] != row["limit"]:
        raise ValueError("invalid ledger")
    if ledger["v"] != 1 or type(ledger["spent"]) is not int or not 0 <= ledger["spent"] <= row["limit"]:
        raise ValueError("budget overrun")
    counts, spent = Counter(), 0
    for charge in row["charges"]:
        kind, amount = charge["kind"], charge["amount"]
        if type(kind) is not str or not kind or type(amount) is not int or amount <= 0:
            raise ValueError("invalid debit")
        counts[kind] += amount
        spent += amount
        if charge["spent"] != spent or charge["counts"] != dict(counts) or charge["limit"] != row["limit"]:
            raise ValueError("nonmonotonic or mismatched debit vector")
        body = {k: charge[k] for k in ("v", "limit", "spent", "counts")}
        if hashlib.sha256(_encoded(body).encode()).hexdigest() != charge["digest"]:
            raise ValueError("invalid charge hash")
    body = {k: ledger[k] for k in ("v", "limit", "spent", "counts")}
    if (spent != ledger["spent"] or dict(counts) != ledger["counts"] or
            hashlib.sha256(_encoded(body).encode()).hexdigest() != ledger["digest"]):
        raise ValueError("ledger/debits mismatch")
    attempt_sum, accepted_wrong, learned_calls, epoch_calls = 0, 0, 0, 0
    for event in row["callbackevents"]:
        kind, args = event["kind"], event["args"]
        if len(args) != 2 or type(event["accepted"]) is not bool:
            raise ValueError("callback schema")
        truth = _operation(kind, args)
        if event["accepted"] and not _same(truth, event["proposal"]):
            accepted_wrong += 1
        learned = event["opname"].startswith("learned:")
        if learned:
            # Admitted BLEND has5 AST nodes,2ports,16domainpoints,oneactivecompetence.
            if kind != "blend" or (event["cost"], event["verify_cost"]) != (28, 52):
                raise ValueError("learned call tariff mismatch")
        elif event["opname"] != kind or (event["cost"], event["verify_cost"]) != (
                (7, 8) if kind == "blend" else (5, 6)):
            raise ValueError("direct call tariff mismatch")
        attempt_sum += event["cost"] + event["verify_cost"] + 2
        if event["phase"].startswith("epoch:"):
            epoch_calls += 1
            learned_calls += learned and event["accepted"] and _same(truth, event["proposal"])
    if attempt_sum != counts["compute_attempt"] + counts["baseline_attempt"]:
        raise ValueError("attempt debit does not match executed/verified callbacks")
    if len(row["epochs"]) != 12:
        raise ValueError("assigned epochs missing")
    current, correct, wrong, abstain, reused = dict(row["initial_inputs"]), 0, 0, 0, 0
    previous_spent = 0
    per_epoch = []
    for i, (update, epoch) in enumerate(zip(row["updates"], row["epochs"])):
        if epoch["epoch"] != i or not previous_spent <= epoch["spent"] <= spent:
            raise ValueError("epoch order/ledger mismatch")
        previous_spent = epoch["spent"]
        current.update(update["changes"])
        outputs = epoch["outputs"]
        if not set(outputs) <= {"amount", "flag"}:
            raise ValueError("unrequested output")
        if epoch["status"] == "complete" and len(outputs) != 2:
            raise ValueError("incomplete completed epoch")
        truth = gold(row["family"], current)
        scores = {"correct": 0, "wrong": 0, "abstain": 0}
        for name in ("amount", "flag"):
            if name not in outputs:
                abstain += 1
                scores["abstain"] += 1
            elif _same(outputs[name], truth[name]):
                correct += 1
                scores["correct"] += 1
            else:
                wrong += 1
                scores["wrong"] += 1
        if epoch["status"] == "complete" and not any(e["phase"] == "epoch:" + str(i) for e in row["callbackevents"]):
            reused += len(outputs)
        per_epoch.append(scores)
    if correct + wrong + abstain != 24:
        raise ValueError("denominator mismatch")
    acquisition = row["acquisition"]
    if type(acquisition["spent"]) is not int or not 0 <= acquisition["spent"] <= spent:
        raise ValueError("acquisition budget")
    return {"seed": row["seed"], "family": row["family"], "limit": row["limit"], "arm": row["arm"],
            "correct": correct, "wrong": wrong, "abstain": abstain, "assigned": 24, "units": spent,
            "source_units": counts["source_read"], "executions": epoch_calls, "learned_calls": learned_calls,
            "wrong_accepted_callbacks": accepted_wrong, "reuse_without_execution": reused,
            "acquisition_units": acquisition["spent"], "acquisition_status": acquisition["status"],
            "per_epoch": per_epoch, "judge_outputs": 24}
