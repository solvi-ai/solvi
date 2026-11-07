"""Train on downstream regret, validate separately, then explicitly activate."""
from solvi.experimental.experience import WorkBudget
from solvi.experimental.experience.decision_learning import (
    DecisionAction, DecisionContext, DecisionLearner, DecisionPolicy, OutcomeFeedback,
    LoggedFeedback, evaluate_logged_policy,
)


def feedback(split):
    return [OutcomeFeedback(f'{split}-{x}', 'external-simulator-v1',
                            DecisionContext(f'{split}-{x}', '1', (
                                DecisionAction('left', (x, 1)), DecisionAction('right', (-x, 1)))),
                            (('left', 3*x), ('right', -3*x)), split) for x in ((-2, -1, 1, 2) if split == 'train' else (-5, -3, 3, 5))]


def main():
    budget = WorkBudget(100000)
    learner = DecisionLearner(budget)
    candidate = learner.train(feedback('train'), loss='decision')
    receipt = learner.admit(candidate, feedback('validation'), DecisionPolicy((0, 0)))
    print('Finite empirical validation:', receipt['accepted'])
    learner.activate(receipt['receipt'])
    context = DecisionContext('new', '1', (DecisionAction('left', (-3, 1)), DecisionAction('right', (3, 1))))
    print('Activated small linear ranker selects:', learner.choose(context))
    saved = learner.snapshot()
    restored = DecisionLearner.restore(saved, budget, authority=learner.authority)
    print('Paid replay:', restored.choose(context))
    logging = (('left', (1, 2)), ('right', (1, 2)))
    logs = [LoggedFeedback(str(i), context, name, reward, logging, 'randomized-host')
            for i, (name, reward) in enumerate((('left', -9), ('right', 9), ('left', -9), ('right', 9)))]
    estimate = evaluate_logged_policy(logs, {context.fingerprint: {'left': 0, 'right': 1}}, budget,
                                      reward_model={context.fingerprint: {'left': -9, 'right': 9}})
    print('IPS/SNIPS/DR and ESS (not confidence bounds):', estimate.to_dict())
    print('Full work:', budget.snapshot())


if __name__ == '__main__':
    main()
