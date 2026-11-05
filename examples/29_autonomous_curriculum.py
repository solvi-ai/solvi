"""Collect external experience, learn a composition and apply it without manual epochs.

Run: python examples/29_autonomous_curriculum.py
This finite resource example covers the x >= 4 branch only, not arbitrary transfer.
"""

from __future__ import annotations

import json

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience import (
    AutoCurriculum,
    CompetenceRegistry,
    ExperienceJournal,
    ExperienceRunner,
    FiniteEnvironment,
    Grammar,
    Program,
    WorkBudget,
)


def main():
    pairs = [(8, 1), (10, 3), (9, 5), (12, 2), (7, 2), (11, 4), (8, 3), (13, 6), (16, 4), (15, 7)]
    cases = {str(i): dict(family="resource", mechanism="v2", x=x, y=y) for i, (x, y) in enumerate(pairs)}
    env = FiniteEnvironment(cases)
    journal = ExperienceJournal(KnowledgeStore(), "automatic-example", {"domain": "resource-v2"})
    grammar = Grammar(("x", "y"), operators=("sub", "add", "max"))
    runner = ExperienceRunner(env, journal, WorkBudget(100_000), CompetenceRegistry(grammar), mode="full")
    controller = AutoCurriculum(runner, list(cases), max_attempts=2)
    results = controller.run()
    if controller.state["status"] != "done" or len(results) != len(cases):
        raise RuntimeError("finite curriculum did not finish")
    if not all(r["ok"] and r["competence_id"] for r in results[8:]):
        raise RuntimeError("acquired operation was not applied to unseen inputs")
    registry = controller.runner.registry.snapshot()
    learned = [c["program"] for c in registry["candidates"].values()]
    if not any(
        Program.from_dict(p).nodes >= 5 and set(Program.from_dict(p).input_names) == {"x", "y"} for p in learned
    ):
        raise RuntimeError("learner did not acquire a composite operation")
    before = env.snapshot()
    restored = AutoCurriculum.restore(
        controller.snapshot(), environment=env, journal=journal, registry=controller.runner.registry
    )
    if restored.run() != results or env.snapshot() != before:
        raise RuntimeError("completed curriculum replay changed effects")
    print(
        json.dumps(
            dict(
                status=controller.state["status"],
                attempts=controller.state["attempts"],
                programs=learned,
                transfer=results[8:],
                work=runner.budget.snapshot(),
                roles={key: value["used_as"] for key, value in controller.state["assignments"].items()},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
