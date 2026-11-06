"""Compare one-pass committees and paid bounded selection on known scalar programs.

Run from the repository: PYTHONPATH=src python examples/32_bounded_observations.py
No model, external dataset, oracle or network is used. Printed work units are
declared operation prices, not a general performance or correctness claim.
"""
from __future__ import annotations

import json

from solvi.experimental.experience import (
    Grammar, ObservationOption, WorkBudget, select_observation,
    select_observation_bounded, synthesize, synthesize_committee,
)


def main():
    grammar = Grammar(("x",), constants=(0,), operators=("add",),
                      max_depth=2, max_nodes=3, max_candidates=64)
    examples = [({"x": 0}, 0)]
    repeated_budget = WorkBudget(10000)
    repeated = []
    for _ in range(3):
        result = synthesize(examples, grammar, repeated_budget,
                            exclude=[program.fingerprint for program in repeated])
        if result.candidate is None:
            raise RuntimeError("known committee did not fit")
        repeated.append(result.candidate)

    committee_budget = WorkBudget(10000)
    result = synthesize_committee(examples, grammar, committee_budget, size=3,
                                  max_additional_work=512, reserve=32)
    committee = result.candidates
    if [p.fingerprint for p in committee] != [p.fingerprint for p in repeated]:
        raise RuntimeError("one-pass committee changed enumeration order")
    check_budget = WorkBudget(1000)
    known = [[program.evaluate({"x": x}, check_budget, grammar) for x in (0, 1, 2)]
             for program in committee]
    if known != [[0, 1, 2], [0, 0, 0], [0, 2, 4]]:
        raise RuntimeError("known x, 0, 2*x answers changed")

    options = [ObservationOption(f"x={x}", {"x": x}, 8) for x in range(16)]
    full_budget, bounded_budget = WorkBudget(10000), WorkBudget(10000)
    full = select_observation(committee, options, grammar, full_budget)

    def predict(program, inputs, paid_budget):
        # The host's immutable options above already satisfy grammar bounds.
        # Program.evaluate charges every evaluated AST node to this same window.
        return program.evaluate(inputs, paid_budget, grammar)

    bounded = select_observation_bounded(committee, options, bounded_budget,
                                         predict=predict, max_work=512)
    if (full.name, full.pairs_separated) != ("x=1", 3):
        raise RuntimeError("full selector changed the known first maximal row")
    if (bounded.name, bounded.pairs_separated) != (full.name, full.pairs_separated):
        raise RuntimeError("bounded selector changed the known choice")
    if bounded_budget.spent > 512:
        raise RuntimeError("selector exceeded its work window")
    print(json.dumps(dict(
        ordered_fingerprints_equal=True, known_predictions=known,
        repeated_search_work=repeated_budget.spent, one_pass_search_work=committee_budget.spent,
        continuation_work=result.continuation_spent, known_answer_check_work=check_budget.spent,
        full_selection=dict(choice=full.name, pairs=full.pairs_separated,
                            reason=full.reason, work=full_budget.spent),
        bounded_selection=dict(choice=bounded.name, pairs=bounded.pairs_separated,
                               reason=bounded.reason, work=bounded_budget.spent),
        observation_purchase_work=0,
        scope="tiny known-answer example; no general performance or accuracy claim",
    ), indent=2))


if __name__ == "__main__":
    main()
