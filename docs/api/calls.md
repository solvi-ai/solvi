# Ordinary function calls

`solvi.trace(fn, storage=store)` wraps an eager synchronous function. The wrapper returns its original result and
records JSON inputs, output or exception, function identity, and the parent traced call in the existing storage chain.
Use `@solvi.trace(storage=store)` for decorator syntax. These are `kind="call"` records, separate from decisions.
Parent UUIDs follow traced nesting across storage objects; a parent can live in a different store. Verifying a store's
hash chain does not establish that every parent is present locally or that all calls have been captured.

```python
import solvi
from solvi.core.store import JSONLStorage
from solvi.core.store.calls import cases_from_calls, replay_call

store = JSONLStorage("calls.jsonl")

@solvi.trace(storage=store)
def total(price, count):
    return price * count

assert total(12, 3) == 36
case = cases_from_calls(store)[0]
assert replay_call(case, total)["ok"]
```

Call capture accepts finite JSON data with string object keys. It snapshots arguments before the function can mutate
them. Generator, coroutine and async-generator functions are refused; arbitrary Python objects and lazy iterator
consumption are outside this API's current scope. Calls may have side effects, and storage failure does not undo them.

Replay requires an explicitly supplied callable. It executes that callable and can repeat its side effects; storage
data never chooses or imports executable code. Ordinary decision `replay_all` does not replay function-call records.
The helpers below replay and compare calls explicitly and export JSON call cases separately from decision test cases.
For a store containing several function names, `diff_calls(store, replacement, name="original-name")` explicitly
selects the original function. An ambiguous unfiltered comparison is refused before executing any replacement calls.
`cases_from_calls` returns JSON data; after a JSON roundtrip, `replay_call(case, function)` checks the exported case.
These fixtures are separate from `solvi.testing` decision cases. Stored-record hash checks establish local consistency,
not authenticity: retain a trusted external store head/anchor when tampering is in scope.

::: solvi.core.store.calls
