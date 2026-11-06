"""Learn x+1 by counterexample, choose a useful observation, reopen a real journal.

Run from the repository: PYTHONPATH=src python examples/31_counterexample_learning.py
No network/model required. A finite verification result is not global correctness.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import (
    DomainReceipt, ExperienceJournal, FiniteDomain, Grammar, ObservationOption,
    RefinementSession, WorkBudget,
)
from solvi.experimental.experience._common import digest


class Oracle:
    source_id = "example-sql"
    source_revision = "increment-v1"

    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS receipts (request TEXT PRIMARY KEY, body TEXT)")

    def reconcile(self, request_id):
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT body FROM receipts WHERE request=?", (request_id,)).fetchone()
        return None if row is None else DomainReceipt(**json.loads(row[0]))

    def read(self, request_id, source_revision, point):
        if source_revision != self.source_revision or set(point) != {"x"} or type(point["x"]) is not int:
            raise ValueError("foreign oracle request")
        with sqlite3.connect(self.path) as conn:
            value = conn.execute("SELECT ? + 1", (point["x"],)).fetchone()[0]
            receipt = DomainReceipt(request_id, "receipt:" + request_id, self.source_id,
                                    self.source_revision, digest(point), value)
            conn.execute("INSERT INTO receipts VALUES (?,?)", (request_id, json.dumps(receipt.to_dict())))
        return receipt


def main():
    with TemporaryDirectory(prefix="solvi-refinement-") as directory:
        root = Path(directory)
        grammar = Grammar(("x",), operators=("add",), max_depth=2, max_nodes=3, max_candidates=100)
        for mode in ("none", "active"):
            budget = WorkBudget(10000)
            oracle = Oracle(root / (mode + "-oracle.db"))
            # Training, observation and verification inputs are disjoint.
            domain = FiniteDomain("increment", "v1", "x-int", grammar, oracle.source_id,
                                  oracle.source_revision, [{"x": 2}, {"x": 3}], budget=budget)
            options = (ObservationOption("initial", {"x": 0}, 8),
                       ObservationOption("probe", {"x": 1}, 8))
            store = KnowledgeStore(str(root / (mode + "-knowledge.db")))
            journal = ExperienceJournal(store, "example-" + mode, {})
            session = RefinementSession(grammar, domain, oracle, options, budget,
                                        session_id="example-" + mode, initial=("initial",),
                                        mode=mode, max_observations=1, journal=journal)
            result = session.run()
            if result["status"] != "verified":
                raise RuntimeError("example did not learn the rule")
            if mode == "none" and not session.state["rejected"]:
                raise RuntimeError("counterexample loop was not exercised")
            if mode == "active" and session.state["observations"] != 1:
                raise RuntimeError("useful observation was not exercised")
            reopened_store = KnowledgeStore(str(root / (mode + "-knowledge.db")))
            reopened = ExperienceJournal(reopened_store, "example-" + mode, {})
            restored = RefinementSession.restore(journal=reopened, grammar=grammar, domain=domain,
                                                 oracle=Oracle(oracle.path), options=options)
            before = restored.budget.snapshot()
            if restored.run() != result or restored.budget.snapshot() != before:
                raise RuntimeError("completed replay repeated work")
            print(json.dumps(dict(mode=mode, result=result, rounds=session.state["rounds"],
                                  observations=session.state["observations"],
                                  counterexamples=len(session.state["rejected"]),
                                  work=budget.spent, replay="unchanged"), indent=2))


if __name__ == "__main__":
    main()
