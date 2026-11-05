"""External experience -> admitted arithmetic -> unseen tasks -> pure replay.

Run: python examples/28_executable_experience.py
This is an API example on a declared finite mechanism, not a learning benchmark.
"""

from __future__ import annotations

import json

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import (
    CompetenceRegistry,
    ExperienceJournal,
    ExperienceRunner,
    ExternalExample,
    FiniteEnvironment,
    Grammar,
    WorkBudget,
)


def main():
    pairs = {
        "proposal-0": (8, 1),
        "proposal-1": (10, 3),
        "proposal-2": (9, 5),
        "proposal-3": (12, 2),
        "admission-0": (7, 2),
        "admission-1": (11, 4),
        "admission-2": (8, 3),
        "admission-3": (13, 6),
        "transfer-0": (16, 4),
        "transfer-1": (15, 7),
    }
    env = FiniteEnvironment({name: {"family": "resource", "x": x, "y": y} for name, (x, y) in pairs.items()})
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "example-experience", {"mechanism": "resource-v1"})
    grammar = Grammar(("x", "y"), operators=("sub", "add", "max"))
    registry = CompetenceRegistry(grammar, store=store)
    runner = ExperienceRunner(env, journal, WorkBudget(100_000), registry, mode="full")

    proposal_ids = [name for name in pairs if name.startswith("proposal-")]
    admission_ids = [name for name in pairs if name.startswith("admission-")]
    runner.run(proposal_ids + admission_ids)

    def external_examples(ids, split):
        examples = []
        for raw in runner.events.values():
            if raw["kind"] != "act" or raw["task_id"] not in ids:
                continue
            task = env.task(raw["task_id"])
            examples.append(
                ExternalExample(
                    raw["event_id"],
                    task.task_id,
                    raw["source"],
                    task.mechanism,
                    task.schema,
                    task.scope,
                    task.inputs,
                    raw["value"]["answer"],
                    split,
                )
            )
        return examples

    admission = runner.learn(external_examples(proposal_ids, "proposal"), external_examples(admission_ids, "admission"))
    if admission is None or not admission.accepted:
        raise RuntimeError("finite operation was not admitted")
    runner.run([name for name in pairs if name.startswith("transfer-")])
    transferred = [r for r in runner.results if r["task_id"].startswith("transfer-")]
    assert all(r["ok"] and r["competence_id"] for r in transferred)
    before = env.snapshot()
    snapshot = runner.snapshot()
    restored = ExperienceRunner.restore(snapshot, environment=env, journal=journal, registry=runner.registry)
    assert restored.replay()["results"] == runner.results and env.snapshot() == before
    print(
        json.dumps(
            {
                "admitted": admission.competence_id,
                "transfer": transferred,
                "work": restored.budget.snapshot(),
                "journal_valid": journal.verify(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
