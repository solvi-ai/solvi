"""Learn a boolean CSV rule, check its full declared domain, apply it in SQLite.

Run: python examples/30_checked_csv_operations.py
This is a small API demonstration, not evidence of general learning superiority.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import (
    AutoCurriculum, CompetenceRegistry, ExperienceJournal, ExperienceRunner, Grammar,
    TabularEnvironment, TabularOracle, WorkBudget,
)


def main():
    # Informative public inputs; proposal/admission/transfer roles are assigned
    # before their external outcomes by AutoCurriculum, not by this example.
    points = [(0, 0, False), (0, 1, False), (1, 0, False), (1, 2, True),
              (2, 3, False), (3, 4, True), (4, 4, False), (0, 4, True),
              (1, 1, False), (1, 3, False), (2, 1, True), (2, 4, True),
              (3, 3, False), (3, 4, False), (4, 0, True), (4, 1, False),
              (0, 2, False), (0, 3, True)]
    with TemporaryDirectory(prefix="solvi-checked-") as directory:
        root = Path(directory)
        csv_path = root / "tasks.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["task_id", "family", "requested", "available", "expired"])
            for index, (requested, available, expired) in enumerate(points):
                writer.writerow([str(index), "eligibility", requested, available, int(expired)])
        environment = TabularEnvironment(root / "effects.db", csv_path)
        store = KnowledgeStore(str(root / "knowledge.db"))
        journal = ExperienceJournal(store, "checked-csv", {"example": "eligibility"})
        grammar = Grammar(("requested", "available", "expired"), constants=(), operators=("lt", "and", "not"),
                          max_depth=3, max_nodes=6, max_candidates=10000)
        budget = WorkBudget(200000)
        oracle = TabularOracle("eligibility")
        domain = oracle.domain(grammar, budget=budget)
        registry = CompetenceRegistry(grammar, store=store,
                                      allowed_sources=(environment.identity, oracle.source_id))
        contracts = {"eligibility": (domain, oracle)}
        runner = ExperienceRunner(environment, journal, budget, registry, mode="fixed",
                                  checked_domains=contracts, checked_store_policy="evidence_only")
        controller = AutoCurriculum(runner, [str(i) for i in range(len(points))],
                                    proposal_quota=8, admission_quota=8, max_attempts=1)
        results = controller.run()
        if controller.state["status"] != "done" or not all(row["ok"] for row in results):
            raise RuntimeError("CSV curriculum did not finish correctly")
        if not all(row["competence_id"] for row in results[16:]):
            raise RuntimeError("verified operation was not applied on transfer rows")
        restored = AutoCurriculum.restore(controller.snapshot(), environment=environment,
                                         journal=journal, registry=registry, checked_domains=contracts,
                                         checked_store_policy="evidence_only")
        spent = restored.runner.budget.spent
        if restored.run() != results or restored.runner.budget.spent != spent:
            raise RuntimeError("completed replay repeated work")
        print(json.dumps({"status": controller.state["status"], "domain_points": len(domain.points),
                          "transfer": results[16:], "work": runner.budget.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
