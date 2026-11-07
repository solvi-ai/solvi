"""A checked exclusion changes the next objective solver's residual."""
from solvi.experimental.experience import WorkBudget
from solvi.experimental.experience.cooperation import (
    ConstraintSolver, CooperativeSearch, FiniteProblem, ObjectiveSolver,
)


def main():
    budget = WorkBudget(200)
    problem = FiniteProblem(
        {"a_cheap": {"cost": 1, "capacity": 1},
         "b_balanced": {"cost": 4, "capacity": 5},
         "c_large": {"cost": 8, "capacity": 9}},
        {"capacity": lambda attrs: attrs["capacity"] >= 4},
        lambda attrs: attrs["cost"], revision="capacity-v1", budget=budget)
    search = CooperativeSearch(problem, budget)
    print("Initial:", search.residual())
    search.run([ConstraintSolver(problem), ObjectiveSolver(problem)])
    print("After exchange:", search.residual())
    restored = CooperativeSearch.restore(search.snapshot(), problem, budget, authority=search.authority)
    print("Independent result:", restored.certify())
    print("Total paid work:", restored.budget.snapshot())


if __name__ == "__main__":
    main()
