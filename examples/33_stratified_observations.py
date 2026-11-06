"""Paid proposal-label stratification; the returned names authorize no reads."""

from solvi.experimental.experience import (
    ExternalExample,
    ObservationOption,
    WorkBudget,
    plan_stratified_observations,
)

pilot = [
    ExternalExample(
        f"pilot{i}", "task", "host", "observed", "bins", "demo", {"f0": 0 if i < 4 else 7}, i >= 4, "proposal"
    )
    for i in range(8)
]
# The host independently shuffles its unread pool before planning.
pool = [ObservationOption(f"row{i}", {"f0": i % 8}, 1) for i in range(16)]
budget = WorkBudget(10000)
plan = plan_stratified_observations(pilot, pool, budget)
print(plan)
print("Planning units:", budget.spent)
# A host must separately pay reads and independently validate any later claim.
