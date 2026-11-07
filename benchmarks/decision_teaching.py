"""Finite known-class teaching comparison; no final evaluation on import.

Tariff: initial/proposal reference labels cost 8 actual pre-debited units.
Task manifests, query selection, finite enumeration, Program nodes, independent
admission/composition and apply are charged to one WorkBudget. Reference eval
labels cost one unit. Exhaustion emits an explicit partial/unknown row. Costs
are operation units, not time. Fixed and direct use the same informative plan
and stop at a singleton, as active does. At most three query labels are obtained;
all_examples is a marked different-information upper control with a full sweep.

The seeded generator varies numeric boundaries, exception positions, axis,
polarity, and known AND/OR composition templates. Held-out instances are from
these known classes; no unseen grammar/operator or OOD claim is supported.
Semantic fingerprints hash the actual target AST with commutative AND/OR
children sorted, without seed or
point roles. Metadata-only split checks do not run policies. Proposal, query,
admission and evaluation points are distinct. Composition predictions use the
actual independently accepted composition receipt AST, scored separately from
finite-domain verified application, which stays unknown on unseen points.
"""
from __future__ import annotations

import random

from solvi.experimental.experience._common import BudgetStop, WorkBudget, canonical, digest, natural, portable
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.teaching import ConceptRef, TeachingSession

ARMS = ('active', 'fixed', 'random', 'all_examples', 'direct_enumerator')


def _task(seed):
    natural(seed, 'seed')
    rng = random.Random(seed)
    family = seed % 3
    axis, other = ('x', 'y') if rng.randrange(2) else ('y', 'x')
    a, b = rng.randint(-4000, 4000), rng.randint(-4000, 4000)
    gap, gap_y = rng.randint(2, 11), rng.randint(2, 11)
    flip = bool(rng.randrange(2))
    operator = ('and', 'or')[rng.randrange(2)]
    low, middle, high = a - gap, a, a + 2 * gap
    low_y, middle_y, high_y = b - gap_y, b, b + 2 * gap_y
    if family == 0:
        parameters = [(a + j * gap,) for j in range(3)]
        asts = [('lt', ('input', axis), ('const', t)) for t, in parameters]
        coordinates = [(a + gap, low_y), (a, low_y), (low, b), (high, b),
                       (a + gap, high_y), (a, high_y)]
        initial_pair = (low - 1, low_y - 1)
    elif family == 1:
        parameters = [(t, e) for t in (a, a + gap) for e in (b, b + gap_y)]
        asts = [('and', ('lt', ('input', axis), ('const', t)),
                 ('not', ('eq', ('input', other), ('const', e)))) for t, e in parameters]
        coordinates = [(low, b), (middle, high_y), (low, b + gap_y),
                       (high, b), (middle, b), (middle, b + gap_y)]
        initial_pair = (low - 1, low_y - 1)
    else:
        parameters = [(t, u) for t in (a, a + gap) for u in (b, b + gap_y)]
        asts = [(operator, ('lt', ('input', axis), ('const', t)),
                 ('lt', ('input', other), ('const', u))) for t, u in parameters]
        # Isolate one utility-independent predicate bit in each first question.
        isolate_y = low_y if operator == 'and' else high_y
        isolate_x = low if operator == 'and' else high
        coordinates = [(middle, isolate_y), (isolate_x, middle_y),
                       (middle, middle_y), (low, high_y), (high, low_y), (high, high_y)]
        initial_pair = (low - 1, low_y - 1) if operator == 'and' else (high + 1, high_y + 1)
    if flip:
        asts = [('not', ast) for ast in asts]
    target = rng.randrange(len(parameters))
    parameter = parameters[target]

    def reference(point):
        x, y = point[axis], point[other]
        if family == 0:
            value = x < parameter[0]
        elif family == 1:
            value = x < parameter[0] and y != parameter[1]
        elif operator == 'and':
            value = x < parameter[0] and y < parameter[1]
        else:
            value = x < parameter[0] or y < parameter[1]
        return not value if flip else value

    def point(pair):
        return {axis: pair[0], other: pair[1]}

    def unique(pairs, excluded=()):
        seen = set(excluded)
        result = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                result.append(point(pair))
        return result

    initial = point(initial_pair)
    query = unique(coordinates, (initial_pair,))
    excluded = {initial_pair, *coordinates}
    admission_pairs = [(x, y) for x in (a - 1, a + gap - 1, a + gap + 1)
                       for y in (b - 1, b, b + gap_y, b + gap_y + 1)]
    admission = unique(admission_pairs, excluded)
    excluded.update((p[axis], p[other]) for p in admission)
    evaluation = unique([(x, y) for x in (low - 2, a, a + gap, high + 2)
                         for y in (low_y - 2, b, b + gap_y, high_y + 2)], excluded)
    constants = tuple(sorted({v for ast_parameter in parameters for v in ast_parameter}))
    grammar = Grammar(('x', 'y'), constants=constants, max_depth=8, max_nodes=64, max_abs=1000000)
    library = tuple(Program(ast) for ast in asts)
    def semantic_normal_form(ast):
        if ast[0] in ('input', 'const'):
            return ast
        children = [semantic_normal_form(child) for child in ast[1:]]
        if ast[0] in ('and', 'or'):
            children.sort(key=canonical)
        return [ast[0], *children]

    semantic_fingerprint = digest(semantic_normal_form(library[target].to_dict()['ast']))
    manifest = dict(family=family, axis=axis, polarity=flip,
                    operator=operator if family == 2 else None,
                    semantic_fingerprint=semantic_fingerprint,
                    library=[p.fingerprint for p in library], initial=initial,
                    query=query, admission=admission, evaluation=evaluation)
    manifest['task_fingerprint'] = digest(manifest)
    return dict(family=family, library=library, grammar=grammar, reference=reference,
                query=query, admission=admission, evaluation=evaluation,
                initial=initial, metadata=manifest,
                is_exception=lambda point: family == 1 and point[axis] < parameter[0] and point[other] == parameter[1])


def task_metadata(seed):
    """Generate bounded task metadata only; does not teach/query/score a policy."""
    return portable(_task(seed)['metadata'])


def check_task_separation(dev_seeds, final_seeds):
    """Metadata-only pre-freeze check; fail on duplicate target or full task."""
    for seeds in (dev_seeds, final_seeds):
        if type(seeds) not in (tuple, list) or len(seeds) > 256:
            raise ValueError('bounded concrete seed sequence required')
    dev = [task_metadata(seed) for seed in dev_seeds]
    final = [task_metadata(seed) for seed in final_seeds]
    semantic_overlap = sorted({m['semantic_fingerprint'] for m in dev} &
                              {m['semantic_fingerprint'] for m in final})
    task_overlap = sorted({m['task_fingerprint'] for m in dev} & {m['task_fingerprint'] for m in final})
    return portable(dict(status='pass' if not semantic_overlap and not task_overlap else 'fail',
                         dev_seeds=list(dev_seeds), final_seeds=list(final_seeds),
                         semantic_overlap=semantic_overlap, task_overlap=task_overlap,
                         dev=dev, final=final, claim='held-out instances of known rule classes'))


def run_trial(seed, arm, limit):
    """Return one portable raw row; caller owns preregistration/freeze/output."""
    natural(seed, 'seed')
    natural(limit, 'limit')
    if arm not in ARMS:
        raise ValueError('unknown teaching arm')
    rng = random.Random(seed)
    task = _task(seed)
    family, library, grammar, reference, query, admission, evaluation = (
        task[k] for k in ('family', 'library', 'grammar', 'reference', 'query', 'admission', 'evaluation'))
    budget = WorkBudget(limit)
    result = dict(seed=seed, arm=arm, family=family, limit=limit, task_metadata=task['metadata'],
                  information='different_information_upper_control' if arm == 'all_examples' else 'initial_plus_at_most_three',
                  proposalerrors=0, unknowns=0, verifiedcount=0, queries=0,
                  selectedcandidate=None, status='unknown', admission=False,
                  composition_checked=False, composition_verified=False,
                  proposal_evaluated=0, evaluation_total=len(evaluation),
                  proposal_unknowns=0, verifiederrors=0, role_overlap=False,
                  query_inputs=[], initial_examples=0, exception_examples=0,
                  admission_verifiedcount=0, composition_proposalerrors=0,
                  composition_proposal_evaluated=0, composition_proposal_unknowns=0,
                  composition_unknowns=0, composition_verifiedcount=0,
                  composition_verifiederrors=0, selectedcomposition=None)
    session, selected, ref, composition_ref, composition_program = None, None, None, None, None
    direct_verified = False

    def label(point, *, initial=False):
        budget.charge('benchmark.teaching.label', 8)
        value = reference(point)
        if initial:
            result['initial_examples'] += 1
        else:
            result['queries'] += 1
            result['query_inputs'].append(dict(point))
        # Explicit exception labels are identified by their semantic role.
        if family == 1:
            result['exception_examples'] += int(task['is_exception'](point))
        return value

    def filter_programs(programs, point, expected):
        survivors = []
        for program in programs:
            budget.charge('benchmark.teaching.enumerate')
            actual = program.evaluate(point, budget, grammar)
            if type(actual) is bool and actual == expected:
                survivors.append(program)
        return survivors

    def direct_admit(program, negate=False):
        for point in admission:
            budget.charge('benchmark.teaching.admission_oracle')
            expected = reference(point)
            actual = program.evaluate(point, budget, grammar)
            if type(actual) is not bool or actual != (not expected if negate else expected):
                return False
        return True

    try:
        budget.charge('benchmark.teaching.task_manifest', len(library) + len(query) + len(admission) + len(evaluation) + 1)
        initial = task['initial']
        if arm == 'direct_enumerator':
            survivors = filter_programs(library, initial, label(initial, initial=True))
            for point in query[:3]:
                if len(survivors) <= 1:
                    break
                survivors = filter_programs(survivors, point, label(point))
            selected = survivors[0] if len(survivors) == 1 else None
            if selected is not None:
                result['selectedcandidate'] = selected.fingerprint
                direct_verified = direct_admit(selected)
                result['admission'] = direct_verified
                result['admission_verifiedcount'] = len(admission) if direct_verified else 0
                if direct_verified:
                    budget.charge('benchmark.teaching.compose', selected.nodes + 1)
                    proposed = Program(('not', selected.ast))
                    result['composition_checked'] = True
                    result['composition_verified'] = direct_admit(proposed, negate=True)
                    if result['composition_verified']:
                        composition_program = proposed
        else:
            session = TeachingSession(grammar, library, query, budget=budget, max_items=64, result_type='bool')
            session.teach(initial, label(initial, initial=True))
            order = list(query)
            if arm == 'random':
                rng.shuffle(order)
            for index in range(len(query) if arm == 'all_examples' else min(3, len(query))):
                if arm != 'all_examples' and len(session.candidates) <= 1:
                    break
                if arm == 'active':
                    question = session.question()
                    if question is None:
                        break
                    point = question.inputs
                    value = label(point)
                    if task['is_exception'](point):
                        session.teach(point, value, exception=True)
                    else:
                        session.answer(question, value)
                else:
                    point = order[index]
                    value = label(point)
                    session.teach(point, value, exception=bool(task['is_exception'](point)))
            selected = session.candidates[0] if len(session.candidates) == 1 else None
            if selected is not None:
                result['selectedcandidate'] = selected.fingerprint
            ref = session.admit('learned', '1', admission, reference,
                                source_id='benchmark-independent-reference', source_revision=f'seed-{seed}')
            result['admission'] = ref is not None
            result['admission_verifiedcount'] = len(admission) if ref is not None else 0
            if ref is not None:
                result['composition_checked'] = True
                composition_ref = session.compose('complement', '1', Program(('not', ('input', 'learned'))),
                                                  {'learned': ref}, admission, lambda p: not reference(p),
                                                  source_id='benchmark-independent-complement', source_revision=f'seed-{seed}')
                if composition_ref is not None:
                    checked = session.apply(composition_ref, admission[0])
                    result['composition_verified'] = checked.status == 'verified'
                    # Extract the actual accepted production receipt, not a guessed tree.
                    saved = session.snapshot()
                    for record in saved['body']['concepts'].values():
                        budget.charge('benchmark.teaching.receipt_lookup')
                        if record['ref'] == [composition_ref.name, composition_ref.version]:
                            budget.charge('benchmark.teaching.receipt_decode', selected.nodes + 1)
                            composition_program = Program.from_dict(record['program'])
                            break
        if composition_program is not None:
            result['selectedcomposition'] = composition_program.fingerprint
        for point in evaluation:
            budget.charge('benchmark.teaching.eval_reference')
            expected = reference(point)
            if selected is None:
                result['proposal_unknowns'] += 1
            else:
                prediction = selected.evaluate(point, budget, grammar)
                result['proposalerrors'] += int(type(prediction) is not bool or prediction != expected)
                result['proposal_evaluated'] += 1
            if composition_program is None:
                result['composition_proposal_unknowns'] += 1
            else:
                prediction = composition_program.evaluate(point, budget, grammar)
                result['composition_proposalerrors'] += int(type(prediction) is not bool or prediction != (not expected))
                result['composition_proposal_evaluated'] += 1
            if session is not None:
                applied = session.apply(ref or ConceptRef('unavailable', '1'), point)
                composed = session.apply(composition_ref or ConceptRef('unavailable-composition', '1'), point)
                for response, count, errors, unknown in (
                    (applied, 'verifiedcount', 'verifiederrors', 'unknowns'),
                    (composed, 'composition_verifiedcount', 'composition_verifiederrors', 'composition_unknowns')):
                    if response.status == 'verified':
                        result[count] += 1
                        result[errors] += int(response.value != (not expected if count.startswith('composition') else expected))
                    else:
                        result[unknown] += 1
            else:
                # Direct baseline has the same finite admission scope, checked independently.
                for unknown in ('unknowns', 'composition_unknowns'):
                    budget.charge('benchmark.teaching.apply')
                    if direct_verified:
                        for checked in admission:
                            budget.charge('benchmark.teaching.membership')
                            if point == checked:
                                raise ValueError('role overlap in benchmark task')
                    result[unknown] += 1
        result['status'] = 'accepted' if result['admission'] else 'ambiguous' if selected is None else 'admission_failed'
    except BudgetStop:
        result['status'] = 'budget_exhausted'
    result['eval_completed'] = result['unknowns'] + result['verifiedcount']
    result['workledger'] = budget.snapshot()
    result['cost'] = budget.spent
    return portable(result)
