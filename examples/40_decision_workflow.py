"""Four directions in one paid workflow; no external effect or pretrained model.

The host/simulator supplies formalizations and outcome evidence. This is not
natural-language compilation, a field trial or a proof of learned-policy gain.
"""
from solvi.experimental.experience import Grammar, Program, WorkBudget
from solvi.experimental.experience.preferences import Alternative, PreferenceSession, Requirement, check_regret_certificate
from solvi.experimental.experience.teaching import TeachingSession
from solvi.experimental.experience.cooperation import FiniteProblem, CooperativeSearch, ConstraintSolver, ObjectiveSolver
from solvi.experimental.experience.decision_learning import (
    DecisionAction, DecisionContext, DecisionLearner, DecisionPolicy, DecisionObjective, OutcomeFeedback,
)


def main():
    budget = WorkBudget(200000)
    configurations = {'economy': {'capacity': 2, 'cost': 1}, 'balanced': {'capacity': 6, 'cost': 4},
                      'large': {'capacity': 9, 'cost': 8}}
    preferences = PreferenceSession([Alternative(k, (-v['cost'], v['capacity'], 1)) for k, v in configurations.items()],
                                    [(1, 1, 0), (1, 3, 0)], requirements=[Requirement('safe', 2, 1)], budget=budget)
    question = preferences.next_question()
    if question:
        # Simulated author supplies a preference; it is not invented by the model.
        a, b = sorted((question.left, question.right), key=lambda k: 3*configurations[k]['capacity']-configurations[k]['cost'], reverse=True)
        preferences.answer('answer1', a, b, source='simulated-author')
    certificate = preferences.choose()
    assert check_regret_certificate(preferences, certificate, budget=budget)

    assert certificate.regret == 0 and len(certificate.surviving_weights) == 1
    cost_weight, reward_weight, _ = preferences.weights[certificate.surviving_weights[0]]
    objective = DecisionObjective(reward_weight=reward_weight, cost_weight=cost_weight)

    grammar = Grammar(('capacity',), constants=(0, 1, 3, 5), max_depth=4)
    hypotheses = [Program(('lt', ('const', threshold), ('input', 'capacity'))) for threshold in (3, 5)]
    teacher = TeachingSession(grammar, hypotheses, [{'capacity': 4}], budget=budget)
    q = teacher.question()
    teacher.answer(q, False)  # User says capacity4 does not suffice.
    requirement = teacher.admit('adequate-capacity', '1', [{'capacity': v['capacity']} for v in configurations.values()],
                                lambda p: p['capacity'] > 5, source_id='independent-capacity-reference', source_revision='1')
    allowed = {k: teacher.apply(requirement, {'capacity': v['capacity']}).value is True for k, v in configurations.items()}
    assignments = {k: {**v, 'admitted': allowed[k]} for k, v in configurations.items()}
    problem = FiniteProblem(assignments, {'learned-capacity': lambda p: p['admitted']},
                            lambda p: cost_weight*p['cost']-reward_weight*p['capacity'],
                            revision='formal-host-v1', budget=budget)
    cooperative = CooperativeSearch(problem, budget)
    cooperative.run([ConstraintSolver(problem), ObjectiveSolver(problem)], rounds=3)
    decision = cooperative.certify()
    assert decision['status'] == 'optimal'

    selected = decision['solution'][1]
    def external_outcome(name, demand):
        # The external simulator owns consequences. Solvers do not provide labels.
        capacity = configurations[name]['capacity']
        return capacity - 3*max(0, demand-capacity)
    observed = {'action': selected, 'demand': 7, 'reward': external_outcome(selected, 7)}
    budget.charge('workflow.external_outcome', 8)
    learner = DecisionLearner(budget)
    def outcomes(split):
        rows = []
        demands = (3, 5, 8, 10) if split == 'train' else (4, 6, 9, 11)
        for demand in demands:
            actions = tuple(DecisionAction(name, (v['capacity']-demand, 1), cost=v['cost'], allowed=allowed[name])
                            for name, v in configurations.items())
            context = DecisionContext(f'{split}-{demand}', 'outcomes-v1', actions)
            budget.charge('workflow.external_full_outcomes', 8*len(actions))
            values = tuple((a.name, external_outcome(a.name, demand)) for a in actions)
            rows.append(OutcomeFeedback(f'{split}-receipt-{demand}', 'external-outcome-simulator', context, values, split))
        return rows
    training = outcomes('train')
    # The simulator's selected-action outcome is retained alongside the full-feedback data;
    # the host additionally observes alternatives in this simulator (not free bandit labels).
    observed_context = DecisionContext('observed-task', 'outcomes-v1', tuple(
        DecisionAction(name, (v['capacity']-7, 1), cost=v['cost'], allowed=allowed[name])
        for name, v in configurations.items()))
    budget.charge('workflow.external_alternative_outcomes', 8*(len(configurations)-1))
    training.append(OutcomeFeedback('observed-task-receipt', 'external-outcome-simulator', observed_context,
                                   tuple((name, observed['reward'] if name == selected else external_outcome(name, 7))
                                         for name in configurations), 'train'))
    candidate = learner.train(training, objective=objective)
    admission = learner.admit(candidate, outcomes('validation'), DecisionPolicy((0, 0), objective))
    learner.activate(admission['receipt'])
    next_context = DecisionContext('next-task', 'outcomes-v1', tuple(
        DecisionAction(name, (v['capacity']-12, 1), cost=v['cost'], allowed=allowed[name])
        for name, v in configurations.items()))
    result = {'preference': certificate.choice, 'taught_feasibility': allowed, 'verified_solution': decision,
              'observed_external_outcome': observed, 'next_learned_proposal': learner.choose(next_context), 'work': budget.snapshot()}
    print(result)
    return result


if __name__ == '__main__':
    main()
