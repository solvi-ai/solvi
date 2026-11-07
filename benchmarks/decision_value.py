"""Full-cost, authored finite resource-allocation decision-learning trials."""
from itertools import product
from random import Random

from solvi.experimental.experience._common import WorkBudget, BudgetStop, natural
from solvi.experimental.experience.decision_learning import (
    DecisionAction, DecisionContext, DecisionObjective, DecisionPolicy, OutcomeFeedback, fit_decision_policy, decision_regret,
)

ARMS = ('decision', 'prediction', 'direct_search', 'constant')


def _reward(features, family):
    # Independent host outcome simulator, never calls the model's predictor.
    x, y, _bias = features
    if family == 0:
        return 3*x - y
    if family == 1:
        return 2*x + y - 12*int(x < 0 and y > 1)
    return 5*min(x,y) + 3*int(x+y > 2)


def _data(seed, budget):
    rng = Random(seed)
    family = seed % 3
    result = {}
    for split, count in (('train', 12), ('validation', 8), ('evaluation', 12)):
        rows = []
        for i in range(count):
            budget.charge('value.data', 1+4*5)
            # Disjoint feature regimes, not simply new seeds with identical rows.
            span = 4 if split == 'train' else 6 if split == 'validation' else 9
            actions = tuple(DecisionAction(f'a{j}', (rng.randrange(-span,span+1), rng.randrange(-span,span+1), 1),
                                           cost=rng.randrange(0,5), allowed=(j != 3)) for j in range(4))
            ctx = DecisionContext(f'{split}-{seed}-{i}', 'external-simulator-v1', actions)
            outcomes = tuple((a.name,_reward(a.features,family)) for a in actions)
            rows.append(OutcomeFeedback(f'outcome-{split}-{seed}-{i}', 'separate-numeric-simulator', ctx, outcomes, split))
        result[split] = rows
    return result


def _regret(policy, rows, budget):
    """Independent finite utility scorer; no tested decision_regret helper."""
    total, wrong, predictions = 0, 0, []
    for row in rows:
        choice = policy.choose(row.context,budget)
        raw = dict(row.outcomes)
        scores = {}
        for a in row.context.actions:
            budget.charge('value.score',2)
            if a.allowed:
                scores[a.name] = raw[a.name]-a.cost
        if choice is None or choice not in scores:
            wrong += 1
            predictions.append(None)
        else:
            total += max(scores.values())-scores[choice]
            predictions.append(choice)
    return total,wrong,predictions


def run_trial(seed, arm, limit):
    natural(seed)
    if arm not in ARMS:
        raise ValueError('unknown decision-learning arm')
    budget = WorkBudget(limit)
    result = {'seed':seed,'family':seed%3,'arm':arm,'status':'unknown','regret':None,
              'hardviolations':None,'assigned':12,'correct':None,'weights':None}
    try:
        data = _data(seed,budget)
        if arm in ('decision','prediction'):
            candidate = fit_decision_policy(data['train'],budget,loss=arm,epochs=4,steps=(2,1))
            policy = candidate.policy
        elif arm == 'direct_search':
            best = None
            for pair in product(range(-12,13),repeat=2):
                weights = (*pair, 0)
                budget.charge('value.direct.candidate')  # bias cancels in every action ranking
                candidate = DecisionPolicy(weights,DecisionObjective())
                score = sum(decision_regret(candidate,row,budget) for row in data['train'])
                if best is None or (score,weights) < best[0]:
                    best = ((score,weights),candidate)
            policy = best[1]
        else:
            budget.charge('value.constant')
            policy = DecisionPolicy((0,0,0))
        # Same separately labelled validation and safe baseline for every arm.
        baseline = DecisionPolicy((0,0,0))
        candidate_validation,violations,_ = _regret(policy,data['validation'],budget)
        baseline_validation,baseline_violations,_ = _regret(baseline,data['validation'],budget)
        budget.charge('value.admission',1)
        accepted = not violations and not baseline_violations and candidate_validation <= baseline_validation
        if not accepted:
            policy = baseline
        regret,violations,predictions = _regret(policy,data['evaluation'],budget)
        correct = 0
        for row,choice in zip(data['evaluation'],predictions):
            raw = dict(row.outcomes)
            scores = []
            for a in row.context.actions:
                budget.charge('value.verify',2)
                if a.allowed:
                    scores.append((raw[a.name]-a.cost,a.name))
            best_score = max(x[0] for x in scores)
            correct += int(any(a == choice and s == best_score for s,a in scores))
        result['evaluation'] = [{'context':row.context.to_dict(), 'outcomes':[list(v) for v in row.outcomes], 'chosen':choice}
                                for row,choice in zip(data['evaluation'],predictions)]
        result.update(status='complete',regret=regret,hardviolations=violations,correct=correct,
                      weights=list(policy.weights),validation_accepted=accepted,predictions=predictions)
    except BudgetStop:
        result['reason'] = 'full budget exhausted including data, learning, validation and scoring'
    result['work'] = budget.snapshot()
    return result
