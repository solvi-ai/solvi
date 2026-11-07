"""Finite preference trials; no evaluation runs execute at import time.

Observation fee: 8 work units for every obtained pairwise answer. The direct
weights arm has privileged information and is only an oracle upper bound.
Planning/solver/checking work uses the same shared ledger, without equalizing
away the actual cost of the minimax query optimizer.
"""
from itertools import combinations
from random import Random

from solvi.experimental.experience._common import BudgetStop, WorkBudget, natural, portable
from solvi.experimental.experience.preferences import (
    Alternative, PreferenceSession, Requirement, check_regret_certificate,
)


ARMS = ('active', 'fixed', 'random', 'direct_weights')
OBSERVATION_FEE = 8
MAX_QUESTIONS = 3


def _problem(seed, budget):
    # Local RNG is seeded before any model/problem construction.
    rng = Random(seed)
    family = seed % 3
    dimensions = family + 3
    budget.charge('trial.setup', 8 * dimensions + 4 * dimensions + 1)
    # Authored families: axis priorities; paired priorities; mixed opposing costs.
    families = (
        ((4,0,0), (0,4,0), (0,0,4), (1,1,1)),
        ((3,1,0,0), (0,3,1,0), (0,0,3,1), (1,0,0,3)),
        ((3,-1,0,0,1), (0,3,-1,0,1), (0,0,3,-1,1), (-1,0,0,3,1)),
    )
    alternatives = []
    for i in range(8):
        attributes = tuple(rng.randrange(0, 61) for _ in range(dimensions))
        # Keep at least one feasible alternative without rejection sampling.
        if i == 0:
            attributes = (max(10, attributes[0]), *attributes[1:])
        alternatives.append(Alternative(f'a{i}', attributes))
    return alternatives, families[family], (Requirement('minimum-first',0,10,True),), rng.randrange(4)


def _score(alternatives, weights, hidden, requirements, answers, choice, budget):
    """Independent dot/max scorer; does not call session.choose or its internals."""
    feasible = []
    for a in alternatives:
        budget.charge('trial.score.feasibility', max(1,len(requirements)))
        if all(a.attributes[r.attribute] >= r.minimum for r in requirements):
            feasible.append(a)
    table = {}
    for i, w in enumerate(weights):
        for a in alternatives:
            budget.charge('trial.score.utility', len(w))
            table[i,a.id] = sum(w[j]*a.attributes[j] for j in range(len(w)))
    survivors = []
    for i in range(len(weights)):
        valid = True
        for preferred, other, relation in answers:
            budget.charge('trial.score.answer')
            left, right = table[i,preferred], table[i,other]
            if not (left == right if relation == 'tie' else left > right):
                valid = False
        if valid:
            survivors.append(i)
    regret = {}
    for a in feasible:
        worst = 0
        for i in survivors:
            for b in feasible:
                budget.charge('trial.score.regret')
                worst = max(worst,table[i,b.id]-table[i,a.id])
        regret[a.id] = worst
    budget.charge('trial.score.realized',len(feasible))
    realized = max(table[hidden,b.id] for b in feasible)-table[hidden,choice]
    violations = 0
    selected = next(a for a in alternatives if a.id == choice)
    for r in requirements:
        budget.charge('trial.score.hard')
        violations += int(r.hard and selected.attributes[r.attribute] < r.minimum)
    return realized, regret[choice], violations, min(regret.items(),key=lambda p:(p[1],p[0]))


def run_trial(seed, arm, limit):
    """Return portable raw metrics, including unknown on any budget exhaustion."""
    natural(seed, 'seed')
    natural(limit, 'limit')
    if arm not in ARMS:
        raise ValueError('unknown preference trial arm')
    budget = WorkBudget(limit)
    result = dict(seed=seed, family=seed%3, arm=arm, status='unknown',
                  realizedregret=None, worstregret=None, chosen=None, queries=0,
                  hardviolations=None, unequalinfo=arm == 'direct_weights',
                  information='exact hidden weights' if arm == 'direct_weights' else 'pairwise answers')
    answers = []
    try:
        alternatives, weights, requirements, hidden = _problem(seed,budget)
        result['fixture'] = {'alternatives':[{'id':a.id,'attributes':list(a.attributes)} for a in alternatives],
                             'weights':[list(w) for w in weights], 'hidden':hidden,
                             'requirements':[{'id':r.id,'attribute':r.attribute,'minimum':r.minimum,'hard':r.hard} for r in requirements]}
        budget.charge('trial.session.setup',1)
        supplied_weights = (weights[hidden],) if arm == 'direct_weights' else weights
        session = PreferenceSession(alternatives,supplied_weights,requirements=requirements,budget=budget)
        feasible = []
        for a in alternatives:
            budget.charge('trial.question.feasibility',len(requirements))
            if all(a.attributes[r.attribute] >= r.minimum for r in requirements):
                feasible.append(a)
        pairs = []
        for a,b in combinations(feasible,2):
            budget.charge('trial.question.pair')
            pairs.append((a.id,b.id))
        random = Random(seed + 104729)
        if arm == 'random':
            budget.charge('trial.question.shuffle',len(pairs))
            random.shuffle(pairs)
        if arm != 'direct_weights':
            for step in range(MAX_QUESTIONS):
                budget.charge('trial.question.plan')
                if arm == 'active':
                    question = session.next_question(cost=OBSERVATION_FEE)
                    if question is None:
                        break
                    left,right = question.left,question.right
                else:
                    if step >= len(pairs):
                        break
                    budget.charge('trial.question.scan',step+1)
                    left,right = pairs[step]
                budget.charge('trial.observation',OBSERVATION_FEE)
                result['queries'] += 1
                # Hidden author's response generation is paid, too.
                values = {}
                for a in alternatives:
                    budget.charge('trial.author.lookup')
                    if a.id in (left,right):
                        budget.charge('trial.author.utility',len(weights[hidden]))
                        values[a.id] = sum(x*y for x,y in zip(a.attributes,weights[hidden]))
                relation = 'tie' if values[left] == values[right] else 'strict'
                preferred,other = (left,right) if values[left] >= values[right] else (right,left)
                answers.append((preferred,other,relation))
                session.answer(f'q{step}',preferred,other,source='simulated-hidden-author',relation=relation)
        certificate = session.choose()
        if certificate.status != 'ok':
            result['status'] = certificate.status
        elif not check_regret_certificate(session,certificate,budget=budget):
            result['status'] = 'invalid_certificate'
        else:
            # Oracle scorer uses its declared singleton information set; all
            # other arms reconstruct survivors independently from observations.
            scoring_weights = supplied_weights if arm == 'direct_weights' else weights
            scoring_hidden = 0 if arm == 'direct_weights' else hidden
            realized,worst,violations,best = _score(alternatives,scoring_weights,scoring_hidden,
                requirements,answers,certificate.choice,budget)
            result.update(status='ok' if (certificate.choice,certificate.regret) == best else 'invalid_choice',
                          realizedregret=realized,worstregret=worst,chosen=certificate.choice,
                          hardviolations=violations)
    except BudgetStop:
        result['status'] = 'unknown'
        result['reason'] = 'budget'
    result['answers'] = [list(a) for a in answers]
    result['workledger'] = budget.snapshot()
    return portable(result)
