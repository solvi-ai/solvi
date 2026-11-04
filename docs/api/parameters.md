# Parameter views and sequential batches

`System.with_params(mapping=None, **facts)` returns an immutable **parameter view**
over the existing System runtime. It does not construct an independent System.

```python
agent = system.with_params({"config": {"offset": 1}}, permission=True)
a = agent.ask({"x": 1})
b = agent.ask({"x": 1, "config": {"offset": 2}})
responses = agent.ask_many([{"x": 1}, {"x": 2}], max_items=100)
```

Parameter values must be finite JSON: string-keyed dictionaries, lists, strings,
numbers, booleans and null, nested at most 32 levels. The mapping is privately
copied; mutating the original or the detached `agent.params` property cannot
change it. Each call receives a fresh copy. Keyword defaults override the mapping;
caller's facts override defaults, including explicit null. These are data defaults,
not a security policy. Unknown external facts are allowed, as on ordinary asks.
Computed part names and the reserved hypothetical-probe marker are refused.

The backend **is shared**: catalog, heads, calibration, guards and their history,
learning policies, statistics, cache and storage. `agent.shared_system` exposes
that same System explicitly. Learning or recalibrating it affects every view;
concurrent independent agents and isolated guard histories are not provided.
The view exposes ask, aask, ask_many, params, shared_system and fingerprint; it
does not forward arbitrary methods such as fit, guarantee or rename. Native
Process binding of a parameter view currently raises TypeError; use a System
with explicit process facts instead.

The backend's ordinary input model, contracts, hard checks and guarantees still
apply after merging. Requests accept dictionaries or BaseModel instances. A
view converts BaseModel fields to a dictionary before merging, so the backend
input model, when configured, validates the merged facts. This is useful when
required input-model fields are supplied by defaults. Caller-provided invalid
values are handled by native input validation; defaults are not fallback values
for rejected overrides. A later catalog edit that turns a default into a computed
part is rejected on the next request.

`agent.fingerprint()` includes the current backend fingerprint and private defaults.
Different defaults have different view fingerprints; views do not freeze backend
code or calibration. Response traces retain the actual merged input and normal
backend fingerprint. Replay with `response.trace.replay(agent.shared_system)` or
the backend store uses those recorded facts, including overrides. A view fingerprint
is not an additional trace field or a separate calibration certificate.

`System.ask_many(states, questions=None, max_items=1000, **ask_options)` and the
view's matching method first materialize a bounded iterable. Containers, fact
keys and the item bound are checked for the whole batch before any call; this
also means an iterator failure during materialization produces no asks. The
same selected questions and ordinary ask options apply to every item. A mapping,
string or bytes object cannot serve as the outer iterable. `max_items` is a
positive integer; bool and float are refused. An empty batch returns an empty list.

Execution is sequential and uses ordinary ask. Each response has its own trace
and, when storage is enabled, its own record; the existing cache and planner keep
their normal eligibility and reuse rules. No implicit concurrency, vectorized
execution or throughput improvement is promised. Runtime validation and backend
errors propagate normally: earlier calls, storage records and learning effects
remain; later inputs are not run. There is no batch transaction or separate
partial-results exception. With store=False, already-created responses are not
returned when a later call raises; choose individual asks when the caller needs
to retain each response despite failures.

No population memory or throughput measurement is supplied by this API change.
