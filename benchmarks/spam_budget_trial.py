"""Single-fit, prefix-uniform paid acquisition experiment."""
from __future__ import annotations
import time
import json
from dataclasses import asdict
from spam_budget_data import prepare
from active_real_trial import TraceBudget
from active_real_models import fit_forest
from solvi.experimental.experience._common import BudgetStop, digest, portable
from solvi.experimental.experience.competences import ExternalExample
from solvi.experimental.experience.observations import ObservationOption

ARMS = ('RANDOM48', 'RANDOM40', 'RANDOM32', 'MIXED40', 'MIXED32', 'FULL_LABEL')
PRICES = (128, 8192)
LIMIT = 4000000

def real_trial(dataset, cache_dir, seed, split, arm, price, *, roles_path=None, emit=None, prepared=None):
    if arm not in ARMS or price not in PRICES:
        raise ValueError('outside frozen experiment configuration')
    data = prepare(dataset, cache_dir, seed, split, roles_path) if prepared is None else prepared
    if (data['name'], data['seed'], data['split']) != (dataset, seed, split):
        raise ValueError('foreign prepared data')
    if len(data['pool']) != 256 or len(data['test']) != 512:
        raise ValueError('incomplete benchmark partition')
    if len({str(row['id']) for row in data['pool']}) != 256:
        raise ValueError('duplicate pool IDs')
    started = time.monotonic()
    budget = TraceBudget(LIMIT)
    receipts, acquired, fits = [], [], []
    model, plan = None, None
    reason = 'label_quota'
    sequence = f'{dataset}:{seed}:{split}:{arm}:{price}'
    def event(kind, payload):
        if emit is not None:
            emit(kind, portable(payload))
    def read(row):
        budget.charge('real_route', 1)
        rid = str(row['id'])
        if any(r['id'] == rid for r in receipts):
            raise ValueError('repeated label')
        budget.charge('real_label', price)
        request = f'{sequence}:label:{len(receipts)}'
        event('label_intent', dict(request=request, id=rid, ledger=budget.snapshot()))
        receipt = dict(request=request, id=rid, expected=row['label'], source=data['source_sha'])
        event('label_receipt', receipt)
        receipts.append(receipt)
        acquired.append(dict(id=rid, inputs=portable(row['inputs']), label=receipt['expected']))
    try:
        for kind, units in data['preprocess_costs'].items():
            budget.charge(kind, units)
        p = len(data['feature_names'])
        budget.charge('real_pool_setup', (p + 1) * 256)
        budget.charge('real_order', 256)
        count = 256 if arm == 'FULL_LABEL' else int(arm[-2:])
        pilot = count - 8 if arm.startswith('MIXED') else count
        for row in data['pool'][:pilot]:
            read(row)
        if arm.startswith('MIXED'):
            from solvi.experimental.experience.stratified import plan_stratified_observations
            budget.charge('real_examples', (p + 1) * pilot)
            examples = tuple(ExternalExample(
                event_id=r['id'], task_id=dataset, source=data['source_sha'], mechanism='paid_label',
                schema='quantized8', scope='pool', inputs=r['inputs'], expected=r['label'], split='proposal'
            ) for r in acquired)
            budget.charge('real_options', (p + 1) * 64)
            rows = data['pool'][pilot:pilot + 64]
            options = tuple(ObservationOption(str(r['id']), r['inputs'], price) for r in rows)
            before = budget.snapshot()
            event('plan_intent', dict(ids=[r['id'] for r in acquired], options=[o.name for o in options], ledger=before))
            chosen = plan_stratified_observations(examples, options, budget, limit=8, candidate_limit=64)
            plan = dict(json.loads(json.dumps(asdict(chosen))), before=before, after=budget.snapshot())
            event('plan_commit', plan)
            budget.charge('real_candidate_index', 64)
            by_id = {str(r['id']): r for r in rows}
            if len(chosen.names) != 8 or len(set(chosen.names)) != 8 or not set(chosen.names) <= by_id.keys():
                raise ValueError('planner returned invalid acquisition IDs')
            for rid in chosen.names:
                read(by_id[rid])
        fit_seed = seed * 100000 + (0 if arm == 'FULL_LABEL' else 800)
        before = budget.snapshot()
        event('fit_intent', dict(index=0, ids=[r['id'] for r in acquired], seed=fit_seed, ledger=before))
        proposed = fit_forest(acquired, budget, seed=fit_seed)
        record = dict(index=0, logical_round=0 if arm == 'FULL_LABEL' else 8, seed=fit_seed,
                      acquired_ids=[r['id'] for r in acquired], before=before, after=budget.snapshot(), model=proposed.to_dict())
        event('fit_commit', record)
        fits.append(record)
        model = proposed
    except BudgetStop:
        reason = 'budget'
        model = None
    frozen = dict(reason=reason, model=None if model is None else model.to_dict(), ledger=budget.snapshot(),
                  labels=len(receipts), fit_count=len(fits))
    event('frozen', frozen)
    predictions = []
    for row in data['test']:
        item = dict(id=str(row['id']), expected=row['label'])
        if model is None:
            item['outcome'] = 'abstain'
        else:
            try:
                value = model.predict(row['inputs'], budget)
            except BudgetStop:
                item['outcome'] = 'budget'
            else:
                item.update(value=value, outcome='correct' if type(value) is bool and value == row['label'] else 'wrong')
        predictions.append(item)
    return dict(tier='budget_real', dataset=dataset, seed=seed, split=split, arm=arm, price=price,
                quota=256 if arm == 'FULL_LABEL' else int(arm[-2:]), data_sha=data['source_sha'], partition_digest=digest(data['group_ids']), thresholds=data['thresholds'],
                rawcounts=data['rawcounts'], quantized_collisions=data['quantized_test_rows_matching_pool'],
                receipts=receipts, fits=fits, plan=plan, frozen=frozen, predictions=predictions,
                budget=budget.snapshot(), trace=budget.trace, wall_seconds=f'{time.monotonic() - started:.9f}')

budget_trial = real_trial
