import ast
import json
import statistics
import tempfile
import time
from pathlib import Path

from solvi.core import store

root = Path(__file__).parent
# Reuse only fixture imports/definitions, not the previous experiment loop.
tree = ast.parse((root / 'profile_runtime.py').read_text())
namespace = {'__file__': str(root / 'profile_runtime.py')}
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.FunctionDef))], type_ignores=[]), 'fixture', 'exec'), namespace)
tests = ast.parse((root.parent.parent / 'tests/test_store_record_body_membership.py').read_text())
ns = {'store': store}
exec(compile(ast.Module(body=[node for node in tests.body if isinstance(node, ast.FunctionDef) and node.name == 'reference'], type_ignores=[]), 'oracle', 'exec'), ns)
candidate = store.record_body
pairs = []
for i in range(7):
    cell = {}
    for name, fn in [('old', ns['reference']), ('new', candidate)][::1 if i % 2 == 0 else -1]:
        store.record_body = fn
        process, systems = namespace['build'](10, Path(tempfile.mkdtemp(prefix='runtime_pair_')))
        checks = []
        for j in range(5):
            process.run({'x': 1}, run=f'warm{j}')
        start = time.perf_counter()
        for j in range(30):
            r = process.run({'x': 1}, run=f'pair{j}')
            checks.append(r.status == 'decided' and r.facts['x'] == 11 and len(r.path) == 10)
        cell[name] = time.perf_counter() - start
        cell[name + '_checks'] = all(checks) and all(s.storage.verify()['ok'] for s in systems) and process._store.verify()['ok']
    pairs.append(cell)
    (root / 'pairs.json').write_text(json.dumps({'pairs': pairs}, indent=2))
store.record_body = candidate
result = {'pairs': pairs, 'median_new_old': statistics.median(p['new'] / p['old'] for p in pairs)}
(root / 'pairs.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
assert all(p['old_checks'] and p['new_checks'] for p in pairs)
