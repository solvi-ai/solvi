import csv
import sqlite3

import pytest

from solvi.experimental.experience.tabular import TabularEnvironment, TabularOracle


def make_csv(tmp_path):
    path = tmp_path / 'tasks.csv'
    columns = ['task_id', 'family', 'stock', 'reserved', 'damaged', 'requested', 'available', 'expired', 'a', 'b', 'c', 'd']
    rows = [dict(task_id='zero', family='inventory', stock=1, reserved=2, damaged=1),
            dict(task_id='yes', family='eligibility', requested=1, available=3, expired=0),
            dict(task_id='late', family='eligibility', requested=1, available=3, expired=1),
            dict(task_id='plan', family='schedule', a=2, b=3, c=1, d=2)]
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_sql_effects_replay_and_reopen(tmp_path):
    csv_path = make_csv(tmp_path)
    path = tmp_path / 'state.db'
    env = TabularEnvironment(path, csv_path)
    for task, answer in [('zero', 0), ('yes', True), ('late', False), ('plan', 7)]:
        measured = env.execute(task, 'observe', {'probe': 'measure', 'field': 'answer'}, 'probe:' + task)
        assert type(measured.value['value']) is type(answer)
        assert measured.value['value'] == answer
        receipt = env.execute(task, 'act', {'answer': answer}, 'act:' + task)
        assert receipt.value['ok']
        reopened = TabularEnvironment(path)
        assert reopened.reconcile('act:' + task) == receipt
        assert reopened.execute(task, 'act', {'answer': answer}, 'act:' + task) == receipt
        assert not reopened.execute(task, 'act', {'answer': answer}, 'repeat:' + task).value['ok']
    assert all(s['effects'] == 1 for s in env.state.values())
    with pytest.raises(ValueError):
        env.execute('plan', 'act', {'answer': 6}, 'act:plan')


def test_strict_types_and_public_support(tmp_path):
    env = TabularEnvironment(tmp_path / 'state.db', make_csv(tmp_path))
    for task, invalid in [('zero', False), ('yes', 1), ('plan', True), ('late', 0)]:
        with pytest.raises(ValueError):
            env.execute(task, 'act', {'answer': invalid}, 'wrong:' + task)
    assert env.task('yes').worlds == env.task('late').worlds
    assert len(env.task('zero').worlds) == 5
    assert len(env.task('plan').worlds) == 10
    assert env.task('yes').probes[0].cost == 8
    assert not env.execute('plan', 'act', {'answer': 8}, 'bad').value['ok']
    assert env.state['plan']['effects'] == 0


def test_pinned_identity_revision_csv_and_sql(tmp_path):
    csv_path = make_csv(tmp_path)
    path = tmp_path / 'state.db'
    TabularEnvironment(path, csv_path)
    for kwargs in [dict(identity='other'), dict(source_revision='v2')]:
        with pytest.raises(ValueError):
            TabularEnvironment(path, **kwargs)
    original = csv_path.read_bytes()
    csv_path.write_bytes(original + b'\n')
    with pytest.raises(ValueError):
        TabularEnvironment(path)
    csv_path.write_bytes(original)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE tabular_tasks SET inputs='{}' WHERE task_id='zero'")
    with pytest.raises(ValueError):
        TabularEnvironment(path)


def test_sql_boolean_zero_and_csv_newline_substitutions_are_detected(tmp_path):
    csv_path = make_csv(tmp_path)
    path = tmp_path / 'state.db'
    TabularEnvironment(path, csv_path)
    original = csv_path.read_bytes()
    assert b'\r\n' in original
    csv_path.write_bytes(original.replace(b'\r\n', b'\n'))
    with pytest.raises(ValueError, match='changed'):
        TabularEnvironment(path)
    csv_path.write_bytes(original)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE tabular_tasks SET inputs=json_set(inputs,'$.expired',0) WHERE task_id='yes'")
    with pytest.raises(ValueError, match='changed'):
        TabularEnvironment(path)


@pytest.mark.parametrize('family,point,expected', [
    ('inventory', dict(stock=4, reserved=1, damaged=2), 1),
    ('eligibility', dict(requested=2, available=2, expired=False), False),
    ('eligibility', dict(requested=0, available=4, expired=False), True),
    ('schedule', dict(a=1, b=1, c=3, d=2), 5),
])
def test_independent_oracle(family, point, expected):
    oracle = TabularOracle(family)
    receipt = oracle.read('req', oracle.source_revision, point)
    assert type(receipt.expected) is type(expected)
    assert receipt.expected == expected
    assert receipt == oracle.read('req', oracle.source_revision, point)
    with pytest.raises(ValueError):
        oracle.read('req', 'other', point)
    bad = dict(point)
    bad[next(iter(bad))] = True
    with pytest.raises(ValueError):
        oracle.read('req', oracle.source_revision, bad)


@pytest.mark.parametrize('family,names,count', [
    ('inventory', ('stock', 'reserved', 'damaged'), 125),
    ('eligibility', ('requested', 'available', 'expired'), 50),
    ('schedule', ('a', 'b', 'c', 'd'), 256),
])
def test_paid_manifest_covers_complete_domain(family, names, count):
    from solvi.experimental.experience.programs import Grammar
    from solvi.experimental.experience._common import WorkBudget, BudgetStop
    oracle = TabularOracle(family)
    grammar = Grammar(names)
    with pytest.raises(BudgetStop):
        oracle.domain(grammar, budget=WorkBudget(0))
    budget = WorkBudget(100000)
    domain = oracle.domain(grammar, budget=budget)
    assert budget.spent > count
    assert len(domain.points) == count
    for index in range(count):
        receipt = oracle.read('r:' + str(index), domain.source_revision, domain.point_dict(index))
        assert type(receipt.expected) is (bool if family == 'eligibility' else int)
