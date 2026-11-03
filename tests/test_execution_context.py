import asyncio
import pytest
from solvi import Catalog
from solvi.core.runtime import Flow, Step, _ExecutionContext, execute, aexecute, MISSING


def flow(cat, names, q='q'):
    return Flow([Step(cat.parts[n]) for n in names], {q: names}, {}, {})


@pytest.mark.parametrize('asynchronous', [False, True])
def test_shared_computation_and_chain(asynchronous):
    cat, calls = Catalog(), []

    @cat.fn
    def shared(raw):
        calls.append('shared')
        return raw + 1

    @cat.fn
    def downstream(shared):
        return shared * 2

    @cat.check
    def valid(shared):
        calls.append('valid')
        return shared > 0

    ctx = _ExecutionContext()
    first = flow(cat, ['shared'])
    second = flow(cat, ['shared', 'valid', 'downstream'])
    async def run():
        await aexecute(cat, first, {'raw': 2}, context=ctx)
        return await aexecute(cat, second, {'raw': 2}, context=ctx)
    if asynchronous:
        trace, values = asyncio.run(run())
    else:
        execute(cat, first, {'raw': 2}, context=ctx)
        trace, values = execute(cat, second, {'raw': 2}, context=ctx)
    assert calls == ['shared', 'valid']
    assert values['downstream'] == 6
    assert trace.init == {'raw': 2}
    assert [r.step for r in trace.records] == [1, 2, 3]
    assert trace.replay(cat, ctx.flow)['ok']
    assert ctx.outputs['shared'].value == 3


def test_missing_is_cached_and_input_rejected():
    cat, calls = Catalog(), []
    @cat.fn
    def failed(raw):
        calls.append(raw)
        raise ValueError('failed')
    ctx = _ExecutionContext()
    f = flow(cat, ['failed'])
    execute(cat, f, {'raw': 1}, context=ctx)
    execute(cat, f, {'raw': 1}, context=ctx)
    assert calls == [1]
    assert ctx.outputs['failed'].value is MISSING
    assert 'failed' not in ctx.vals
    with pytest.raises(ValueError):
        execute(cat, f, {'raw': 2}, context=ctx)
    assert not ctx._busy


def test_skipped_step_can_run_later():
    cat, calls = Catalog(), []
    @cat.check(hard=True)
    def deny(raw):
        return False
    @cat.fn
    def later(raw):
        calls.append(raw)
        return raw
    ctx = _ExecutionContext()
    execute(cat, flow(cat, ['deny', 'later']), {'raw': 1}, context=ctx)
    assert not calls
    execute(cat, flow(cat, ['later'], 'other'), {'raw': 1}, context=ctx)
    assert calls == [1]
    assert ctx.trace.replay(cat, ctx.flow)['ok']


def test_lease_precedes_input_hashing(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from solvi.core import runtime
    entered, resume = threading.Event(), threading.Event()
    original = runtime.vhash
    def blocked(value):
        entered.set()
        assert resume.wait(5)
        return original(value)
    monkeypatch.setattr(runtime, 'vhash', blocked)
    ctx, cat = _ExecutionContext(), Catalog()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ctx.acquire, cat, {'raw': 1}, True)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match='already in use'):
                execute(cat, flow(cat, []), {'raw': 1}, context=ctx)
            assert ctx._busy
        finally:
            resume.set()
        future.result(timeout=5)
    ctx.release()
    assert not ctx._busy


@pytest.mark.parametrize('hashes', [False, True])
def test_catalog_definition_changes_rejected(hashes):
    cat = Catalog()
    @cat.fn
    def f(raw):
        return raw + 1
    ctx = _ExecutionContext()
    execute(cat, flow(cat, ['f']), {'raw': 1}, hashes=hashes, context=ctx)
    cat.parts['f'].func = lambda raw: raw + 100
    with pytest.raises(ValueError, match='catalog definition'):
        execute(cat, flow(cat, ['f']), {'raw': 1}, hashes=hashes, context=ctx)
    assert not ctx._busy


def test_constructor_failure_releases_lease():
    from solvi.core.runtime import _Run
    ctx, cat = _ExecutionContext(), Catalog()
    with pytest.raises(AttributeError):
        _Run(cat, None, {}, context=ctx)
    assert not ctx._busy
    execute(cat, flow(cat, []), {}, context=ctx)


def test_execution_failure_releases_lease(monkeypatch):
    from solvi.core.runtime import _Run
    ctx, cat = _ExecutionContext(), Catalog()
    def fail(self):
        raise RuntimeError('driver failed')
    with monkeypatch.context() as patch:
        patch.setattr(_Run, 'phases', fail)
        with pytest.raises(RuntimeError, match='driver failed'):
            execute(cat, flow(cat, []), {}, context=ctx)
    assert not ctx._busy
    execute(cat, flow(cat, []), {}, context=ctx)


def test_async_cancellation_releases_lease():
    ctx, cat = _ExecutionContext(), Catalog()
    async def run():
        entered = asyncio.Event()
        @cat.fn
        async def f(raw):
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(aexecute(cat, flow(cat, ['f']), {'raw': 1}, context=ctx))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ctx._busy
        await aexecute(cat, flow(cat, []), {'raw': 1}, context=ctx)
    asyncio.run(run())


def test_reused_outputs_are_excluded_from_forward_batch():
    from solvi.core.runtime import _Run
    cat = Catalog()
    @cat.fn
    def a(raw):
        return raw + 1
    @cat.fn
    def b(raw):
        return raw + 2
    @cat.fn
    def c(raw):
        return raw + 3
    ctx = _ExecutionContext()
    execute(cat, flow(cat, ['a']), {'raw': 1}, context=ctx)
    fresh = flow(cat, ['a', 'b', 'c'])
    fresh.batches = [['a', 'b', 'c']]
    run = _Run(cat, fresh, {'raw': 1}, context=ctx)
    try:
        assert 'a' not in run.batch_of
        assert run.batch_of['b'][0] == ['b', 'c']
        assert run.batch_of['b'][1] == [cat.parts['b'].func, cat.parts['c'].func]
    finally:
        ctx.release()


def test_lean_context_still_guards_raw_content():
    cat, ctx = Catalog(), _ExecutionContext()
    raw = {'items': [1]}
    trace, _ = execute(cat, flow(cat, []), raw, hashes=False, context=ctx)
    assert trace.init_hash == ''
    raw['items'].append(2)
    with pytest.raises(ValueError, match='raw input'):
        execute(cat, flow(cat, []), raw, hashes=False, context=ctx)
    assert not ctx._busy


def test_single_fresh_batch_member_runs_without_cached_member():
    cat, ctx, calls = Catalog(), _ExecutionContext(), []
    @cat.fn
    def a(raw):
        calls.append('a')
        return raw + 1
    @cat.fn
    def b(raw):
        calls.append('b')
        return raw + 2
    def shared_pass(parts, args, names):
        calls.extend(names)
        raise AssertionError('cached a must not be scored again')
    cat.parts['b'].func.in_pass = shared_pass
    execute(cat, flow(cat, ['a']), {'raw': 1}, context=ctx)
    both = flow(cat, ['a', 'b'])
    both.batches = [['a', 'b']]
    _, values = execute(cat, both, {'raw': 1}, context=ctx)
    assert values['b'] == 3
    assert calls == ['a', 'b']


def test_continuation_keeps_steps_after_appended_acceptance_metadata():
    from solvi.core.chain import append
    from solvi.core.runtime import Record, vhash
    cat = Catalog()

    @cat.fn
    def first(raw):
        return raw + 1

    @cat.fn
    def second(first):
        return first * 2

    ctx = _ExecutionContext()
    execute(cat, flow(cat, ['first']), {'raw': 1}, context=ctx)
    # Answer/plan records can be appended after a logical flow with skipped steps.
    append(ctx.trace, Record(step=0, kind='head', name='answer:source', inputs={'first': vhash(2)},
                             value='yes', confidence=.9, provenance='learned'), 10)
    trace, _ = execute(cat, flow(cat, ['first', 'second'], 'next'), {'raw': 1}, context=ctx)
    assert [r.step for r in trace.records] == [1, 11, 12]
    previous = trace.init_hash
    for record in trace.records:
        assert record.prev == previous and record.hash == vhash(record.body())
        previous = record.hash
