"""Single-fit, prefix-uniform paid acquisition experiment."""
from __future__ import annotations
import time
from packed_spam_data import prepare
from active_real_trial import TraceBudget
from active_real_models import fit_forest
from solvi.experimental.experience._common import BudgetStop, digest, portable

ARMS = ('RANDOM48', 'RANDOM40', 'FULL_LABEL', 'LEGACY40', 'MEDIAN40', 'CUT40', 'ADAPT40')
PRICES = (128, 8192)
LIMIT = 4000000

def real_trial(dataset, cache_dir, seed, split, arm, price, *, fold="f0", roles_path=None, emit=None, prepared=None):
    if arm not in ARMS or price not in PRICES:
        raise ValueError('outside frozen experiment configuration')
    data = prepare(dataset, cache_dir, seed, split, fold, roles_path) if prepared is None else prepared
    if (data['name'], data['seed'], data['split']) != (dataset, seed, split):
        raise ValueError('foreign prepared data')
    if len(data['pool']) != 256 or len(data['test']) != 512:
        raise ValueError('incomplete benchmark partition')
    if len({str(row['id']) for row in data['pool']}) != 256:
        raise ValueError('duplicate pool IDs')
    if data.get('fold', fold) != fold:
        raise ValueError('foreign prepared fold')
    if arm == 'LEGACY40':
        from spam_budget_trial import real_trial as legacy_trial
        record = legacy_trial(dataset, cache_dir, seed, split, 'MIXED40', price, prepared=data, emit=emit)
        return dict(tier='packed_real', dataset=dataset, seed=seed, split=split, fold=fold,
                    arm=arm, price=price, record=record)
    started = time.monotonic()
    budget = TraceBudget(LIMIT)
    receipts, acquired, fits = [], [], []
    model, controller = None, None
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
        acquired.append(dict(id=rid, inputs=row['inputs'], label=receipt['expected']))
    try:
        for kind, units in data['preprocess_costs'].items():
            budget.charge(kind, units)
        p = len(data['feature_names'])
        budget.charge('real_pool_setup', (p + 1) * 256)
        budget.charge('real_order', 256)
        count = 256 if arm == 'FULL_LABEL' else int(arm[-2:])
        pilot = 32 if arm in ('MEDIAN40', 'CUT40', 'ADAPT40') else count
        for row in data['pool'][:pilot]:
            read(row)
        if arm in ('MEDIAN40', 'CUT40', 'ADAPT40'):
            from solvi.experimental.experience.packed import PackedObservationFrame, PackedStratifiedSampler
            strategy = 'adaptive' if arm == 'ADAPT40' else 'round_robin'
            cut_mode = 'median' if arm == 'MEDIAN40' else 'all'
            controller = dict(strategy=strategy, cut_mode=cut_mode, max_steps=8, steps=[])
            before = budget.snapshot()
            controller['init_before'] = before
            event('controller_intent', dict(config=dict(strategy=strategy, cut_mode=cut_mode, max_steps=8), ledger=before))
            budget.charge('real_frame_route', 96)
            records = tuple((str(r['id']), r['inputs']) for r in data['pool'][:96])
            source = f"{dataset}:{data['source_sha']}:{fold}:{split}:{seed}"
            frame = PackedObservationFrame(records, budget, source=source)
            budget.charge('real_pilot_labels', 32)
            paid = tuple((r['id'], r['expected']) for r in receipts)
            sampler = PackedStratifiedSampler(frame, paid, budget, strategy=strategy, cut_mode=cut_mode, max_steps=8)
            budget.charge('real_candidate_index', 64)
            by_id = {str(r['id']): r for r in data['pool'][32:96]}
            controller.update(init_after=budget.snapshot(), source=source,
                              features=list(sampler.features), thresholds=list(sampler.thresholds),
                              reason=getattr(sampler, 'reason', 'stratified' if sampler.features else 'single_class'))
            event('controller_commit', {k:v for k,v in controller.items() if k != 'steps'})
            for index in range(8):
                step = dict(index=index, before=budget.snapshot())
                event('choice_intent', step)
                rid = sampler.choose(budget)
                step.update(id=rid, after_choose=budget.snapshot())
                event('choice_commit', step)
                if rid not in by_id:
                    raise ValueError('controller returned foreign or missing candidate')
                read(by_id[rid])
                sampler.observe(rid, receipts[-1]['expected'], budget)
                step.update(observed=receipts[-1]['expected'], after_observe=budget.snapshot())
                controller['steps'].append(step)
                event('observe_commit', step)
            controller['summary'] = dict(received=len(sampler.received), features=list(sampler.features),
                                         thresholds=list(sampler.thresholds), reason=controller['reason'])
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
    record = dict(tier='packed_native', dataset=dataset, seed=seed, split=split, arm=arm, price=price,
                quota=256 if arm == 'FULL_LABEL' else int(arm[-2:]), data_sha=data['source_sha'], partition_digest=digest(data['group_ids']), thresholds=data['thresholds'],
                rawcounts=data['rawcounts'], quantized_collisions=data['quantized_test_rows_matching_pool'],
                receipts=receipts, fits=fits, controller=controller, frozen=frozen, predictions=predictions,
                budget=budget.snapshot(), trace=budget.trace, wall_seconds=f'{time.monotonic() - started:.9f}')

    return dict(tier='packed_real', dataset=dataset, seed=seed, split=split, fold=fold,
                arm=arm, price=price, record=record)

budget_trial = real_trial
