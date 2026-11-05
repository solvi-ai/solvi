import cProfile
import json
import pstats
import tempfile
import warnings
from pathlib import Path

from solvi import Answer, Catalog, Question, System
from solvi.core.runtime import trace_hash
from solvi.core.store import JSONLStorage, run_context
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.process import Process

ROOT = Path(__file__).parent

def build(n, folder):
    p = Process('native-chain', storage=JSONLStorage(folder / 'process.jsonl'))
    systems = []
    for i in range(n):
        cat = Catalog()
        @cat.rule('q')
        def increment(x: int) -> int:
            return x + 1
        s = System(cat, [Question('q', '?', Answer.choice(list(range(100))))],
                   storage=JSONLStorage(folder / f'n{i}.jsonl'))
        systems.append(s)
        p.node(f'n{i}', s, reads=['x'], writes={'x': 'answer'}, end=i == n - 1)
        if i:
            p.edge(f'n{i-1}', f'n{i}')
    return p, systems


summary = []
for n in (3, 10):
    folder = Path(tempfile.mkdtemp(prefix=f'native_runtime_{n}_'))
    p, systems = build(n, folder)
    records = []
    prof = cProfile.Profile()
    for j in range(35):
        if j == 5:
            prof.enable()
        result = p.run({'x': 1}, run=f'run-{j}')
        records.append({'request': j, 'status': result.status, 'answer': result.answer,
                        'x': result.facts['x'], 'path': len(result.path),
                        'nodes': [(st['node'], st['by'], st['status']) for st in result.path]})
        (ROOT / f'outputs_{n}.json').write_text(json.dumps(records, indent=2))
    prof.disable()
    prof.dump_stats(str(ROOT / f'profile_{n}.pstats'))
    with (ROOT / f'profile_{n}.txt').open('w') as handle:
        pstats.Stats(prof, stream=handle).sort_stats('cumulative').print_stats(35)
    # Admission parity outside the profile, same run/parent context.
    x, parent, parity = 1, None, []
    for i, s in enumerate(systems):
        with run_context(run='run-34', parent=parent):
            direct = s.ask({'x': x})
        x = direct['q'].answer
        parity.append(trace_hash(direct) == result.nodes[f'n{i}'].trace_hash)
        parent = direct.stored_id
    verify = [s.storage.verify()['ok'] for s in systems] + [p._store.verify()['ok']]
    cell = {'n': n, 'known_answers': all(r['x'] == n+1 and r['status'] == 'decided' and r['path'] == n for r in records),
            'trace_parity': parity, 'journals_ok': verify, 'journal_dir': str(folder)}
    summary.append(cell)
    (ROOT / 'checks.json').write_text(json.dumps(summary, indent=2))
    assert cell['known_answers'] and all(parity) and all(verify), cell
print(json.dumps(summary))
