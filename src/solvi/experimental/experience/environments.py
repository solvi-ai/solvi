"""Finite host-owned mechanisms; no learned program participates in truth.

Snapshots contain private cases and belong to the host, never to the learner.
Digests detect corruption, not malicious rewriting by a trusted host.
"""

from __future__ import annotations

from ._common import canonical, digest, identifier, natural, portable
from .interfaces import ProbeSpec, Receipt, TaskSpec


class FiniteEnvironment:
    """Two tiny executable families with idempotent external attempts.

    Cases accept family, x, y, mechanism (v1/v2), announced_version, threshold
    (default 4), and noise (0/1). Resource cases require x >= y. The resource
    v2 interaction charges one extra unit when x reaches threshold. Workflow
    v1 stages run in parallel; v2 stages run sequentially.
    """

    def __init__(self, cases: dict[str, dict], *, identity="environment"):
        self.identity = identifier(identity, "environment identity")
        if type(cases) is not dict:
            raise ValueError("cases must be a dict")
        self._cases = {}
        for task_id, raw in portable(cases).items():
            identifier(task_id, "task id")
            if (
                type(raw) is not dict
                or set(raw) - {"family", "x", "y", "mechanism", "announced_version", "threshold", "noise"}
                or not {"family", "x", "y"} <= set(raw)
            ):
                raise ValueError("unsupported case configuration")
            c = {"mechanism": "v1", "announced_version": True, "threshold": 4, "noise": 0, **raw}
            if c["family"] not in ("resource", "workflow") or c["mechanism"] not in ("v1", "v2"):
                raise ValueError("unknown family or mechanism")
            for name in ("x", "y", "threshold", "noise"):
                natural(c[name], name)
            if max(c["x"], c["y"], c["threshold"]) > 31 or c["noise"] > 1:
                raise ValueError("case exceeds finite domain")
            if type(c["announced_version"]) is not bool:
                raise ValueError("announced_version must be bool")
            if c["family"] == "resource" and c["x"] < c["y"]:
                raise ValueError("resource cases require x >= y")
            self._cases[task_id] = c
        self._state = {k: self._initial(c) for k, c in self._cases.items()}
        self._intents = {}
        self._receipts = {}

    @staticmethod
    def _truth(c):
        # Physical semantics, independent of the learner's AST/evaluator.
        if c["family"] == "resource":
            inventory = list(range(c["x"]))
            for _ in range(c["y"]):
                inventory.pop()
            available = len(inventory)
            if c["mechanism"] == "v2" and c["x"] >= c["threshold"]:
                available -= 1
            return available
        elapsed = 0
        remaining = [c["x"], c["y"]]
        while any(remaining):
            elapsed += 1
            if c["mechanism"] == "v1":
                remaining = [n - 1 if n else 0 for n in remaining]
            else:
                for i, n in enumerate(remaining):
                    if n:
                        remaining[i] -= 1
                        break
        return elapsed

    @classmethod
    def _initial(cls, c):
        if c["family"] == "resource":
            return {"available": cls._truth(c), "reserved": 0, "effects": 0}
        return {"clock": 0, "completed": False, "stages_completed": [False, False], "effects": 0}

    def _case(self, task_id):
        identifier(task_id, "task id")
        if task_id not in self._cases:
            raise ValueError("unknown task")
        return self._cases[task_id]

    def task(self, task_id):
        c = self._case(task_id)
        x, y = c["x"], c["y"]
        # Candidate support depends only on public numbers, never on which
        # mechanism is actual. Neither ordering nor flags mark the true world.
        shifted = x - y - 1 if x >= c["threshold"] else x - y
        answers = sorted({0, x, y, x - y, x + y, max(x, y), min(x, y), shifted})
        worlds = tuple({"answer": a, "noise": n, "parity": a % 2} for a in answers for n in (0, 1))
        return TaskSpec(
            task_id=task_id,
            scope=c["family"],
            mechanism=c["mechanism"] if c["announced_version"] else "v1",
            schema="xy-int-v1",
            inputs={"x": x, "y": y},
            worlds=worlds,
            probes=(
                ProbeSpec("measure", "answer", 8),
                ProbeSpec("noise", "noise", 1),
                ProbeSpec("diagnose", "parity", 4),
            ),
        )

    @property
    def state(self):
        """Detached private host audit state; not learner input."""
        return portable(self._state)

    def _validate_intent(self, task_id, kind, payload):
        self._case(task_id)
        if type(payload) is not dict:
            raise ValueError("payload must be a dict")
        payload = portable(payload)
        if kind == "observe":
            if set(payload) != {"probe", "field"} or not any(
                payload == {"probe": p.name, "field": p.field} for p in self.task(task_id).probes
            ):
                raise ValueError("unknown probe or mismatched field")
        elif kind == "act":
            if set(payload) != {"answer"} or type(payload["answer"]) is not int:
                raise ValueError("action requires an integer answer")
        else:
            raise ValueError("unknown external action kind")
        return {"task_id": task_id, "kind": kind, "payload": payload}

    def _outcome(self, intent, state):
        task_id, kind, payload = intent["task_id"], intent["kind"], intent["payload"]
        c = self._cases[task_id]
        answer = self._truth(c)
        if kind == "observe":
            measured = {"answer": answer, "noise": c["noise"], "parity": answer % 2}
            return {"field": payload["field"], "value": measured[payload["field"]]}, state
        next_state = portable(state)
        ok = payload["answer"] == answer
        if c["family"] == "resource":
            ok = ok and state["available"] > 0
            if ok:
                next_state["available"] -= 1
                next_state["reserved"] += 1
        else:
            ok = ok and not state["completed"]
            if ok:
                next_state.update(clock=answer, completed=True, stages_completed=[True, True])
        if ok:
            next_state["effects"] += 1
        return {"answer": answer, "ok": ok, "refused": not ok, "state": next_state}, next_state

    def _receipt(self, attempt_id, intent, value):
        return Receipt(
            event_id="event:" + digest([self.identity, attempt_id]),
            attempt_id=attempt_id,
            task_id=intent["task_id"],
            kind=intent["kind"],
            source=self.identity,
            mechanism=self.task(intent["task_id"]).mechanism,
            value=value,
        )

    def execute(self, task_id, kind, payload, attempt_id):
        identifier(attempt_id, "attempt id")
        intent = self._validate_intent(task_id, kind, payload)
        if attempt_id in self._intents:
            if canonical(intent) != canonical(self._intents[attempt_id]):
                raise ValueError("attempt id conflicts with an existing intent")
            return self.reconcile(attempt_id)
        value, next_state = self._outcome(intent, self._state[task_id])
        receipt = self._receipt(attempt_id, intent, value)
        self._intents[attempt_id] = intent
        self._receipts[attempt_id] = receipt.to_dict()
        self._state[task_id] = next_state
        return Receipt.from_dict(receipt.to_dict())

    def reconcile(self, attempt_id):
        identifier(attempt_id, "attempt id")
        r = self._receipts.get(attempt_id)
        return None if r is None else Receipt.from_dict(r)

    def snapshot(self):
        body = {
            "v": 1,
            "identity": self.identity,
            "cases": portable(self._cases),
            "records": [
                {"attempt_id": k, "intent": portable(v), "receipt": portable(self._receipts[k])}
                for k, v in self._intents.items()
            ],
            "state": self.state,
        }
        return {**body, "digest": digest(body)}

    @classmethod
    def restore(cls, snapshot, *, identity=None, cases=None):
        """Validate history with pure transitions; never execute external work.

        Optional identity/cases pin the expected host configuration on reopen.
        """
        s = portable(snapshot)
        if type(s) is not dict or set(s) != {"v", "identity", "cases", "records", "state", "digest"}:
            raise ValueError("unsupported environment snapshot")
        if (
            type(s["v"]) is not int
            or s["v"] != 1
            or s["digest"] != digest({k: v for k, v in s.items() if k != "digest"})
        ):
            raise ValueError("corrupt environment snapshot")
        env = cls(s["cases"], identity=s["identity"])
        if identity is not None and identity != env.identity:
            raise ValueError("environment identity mismatch")
        if cases is not None and canonical(cls(cases, identity=env.identity)._cases) != canonical(env._cases):
            raise ValueError("environment case/version mismatch")
        if type(s["records"]) is not list:
            raise ValueError("invalid environment records")
        for record in s["records"]:
            if type(record) is not dict or set(record) != {"attempt_id", "intent", "receipt"}:
                raise ValueError("invalid environment record")
            key = identifier(record["attempt_id"], "attempt id")
            if key in env._intents:
                raise ValueError("duplicate environment attempt")
            raw = record["intent"]
            if type(raw) is not dict or set(raw) != {"task_id", "kind", "payload"}:
                raise ValueError("invalid recorded intent")
            intent = env._validate_intent(raw["task_id"], raw["kind"], raw["payload"])
            value, next_state = env._outcome(intent, env._state[intent["task_id"]])
            expected = env._receipt(key, intent, value).to_dict()
            if canonical(record["receipt"]) != canonical(expected):
                raise ValueError("receipt conflicts with case or history")
            env._intents[key] = intent
            env._receipts[key] = expected
            env._state[intent["task_id"]] = next_state
        if canonical(s["state"]) != canonical(env._state):
            raise ValueError("environment state conflicts with history")
        return env


__all__ = ["FiniteEnvironment"]
