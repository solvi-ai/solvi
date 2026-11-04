# Binding a check to its actual source facts

A rule can read a cached tax rate while its invariant reads a fresh database
rate, even when the invariant's Python argument is named `tax`:

```python
@catalog.check(hard=True, then={"pay": "no"}, reads_from={"tax": "database_tax"})
def rate_matches(invoice_tax: int, tax: int) -> bool:
    return invoice_tax == tax
```

Only this check's argument `tax` is bound to `database_tax`. Other rules reading
`tax` keep their existing dependencies. The original Python function and its
ordinary calling signature are retained. `then` functions have their own inputs;
the check's mapping does not redirect them.

The planner, typed producer/consumer compatibility checks, input schemas, trace
hashes, provenance paths, and replay all name `database_tax`. Values are validated
under that fact name before the original check is called with `tax=...`. A source
can be an ordinary given fact or a computed fact. A given input cannot override
an existing computation; normal catalog collision and missing-input guards apply.

`reads_from` is a dict from existing function argument names to Python fact
identifiers. The effective inputs must be distinct: mapping one argument onto
another mapped or unmapped argument's source is refused. Self-dependencies and
cycles through already-declared computations are refused before registration.
Existing planner/lint cycle checks still apply to declarations added later.
Model-backed and shared-pass check calls are deliberately unsupported and refused
before any model call; their joint call/input semantics require a separate contract.

Bindings are copied into immutable pairs when registered, so editing the caller's
dict cannot change a catalog. They are included in part/catalog fingerprints and
the fingerprint cache key. A replacement binding changes cache/replay identity;
mutating a Part into an inconsistent declaration fails closed rather than using
another source silently. Re-register a check to change its dependencies and types
coherently. Boundary aliases remain ordinary graph computations.

`solvi check` reports `check_shared_source` only for checks that opt into this
binding and share a syntactic given-source dependency with their question's rule.
The warning names those actual given facts, including dependencies reached through
computed parts and aliases. Existing catalogs without bindings gain no such warning.

This is source binding, not a statistical independence certificate. Two different
fact names can read one stale backend, correlated models, or a copied answer. The
lint warning detects shared declared graph roots; its absence proves no backend,
causal, or sampling independence. Supply and audit the intended data sources.
