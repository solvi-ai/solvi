# Explicit coexistence and validity contracts

The store's ordinary `(scope, subject, relation)` key has succession semantics.
A later start can close an earlier holder. Different values at the same start
can compete, including a writer replacing its own earlier value. Keep that
contract for relations with one holder at a time.

For a relation and subject known to allow simultaneous holders, declare a
separate scope key for each holder. This is the existing scoped-key contract;
it does not add `multi=True` or change the default resolver.

```python
from solvi import Knowledge

km = Knowledge(writers={"wikidata": "outcome"})
for holder in ("captain A", "captain B"):
    km.tell({"s": "SM", "r": "P35", "o": holder},
            scope={"holder": holder}, source="outcome", by="wikidata",
            valid=("2020-04-01", "2020-10-01"))
snapshot = km.snapshot(about="SM", relation="P35", valid_at="2020-06-01")
holders = Knowledge.values(snapshot, "SM", "P35")  # both, order is not precedence
```

`Knowledge.value` raises `ValueError` when distinct matching values remain.
Use `values` for a plural relation or narrow the singular read with
`scope={"holder": ...}`. A scalar reader does not pick a holder by hash.
A decision consuming the plural result must accept a collection explicitly.

Scopes and half-open validity intervals are ordinary journal data, so the same
meaning survives reopening, rebuild, and isolated writable branches. Writer
registration, source gates, withdrawal, and pending re-test semantics still
apply. Reopen with the intended write gates and writer registry as usual;
those Python policies are not exported by the journal. A holder key does not
close when a *different* holder starts. Callers must supply real term ends:
missing end dates mean continuing membership. Do not apply holder scopes to
all relations, since doing so changes single-holder succession into coexistence.
Do not mix numeric and ISO-date world clocks within a time query; restrict
snapshot subjects/relations before querying a store containing different clocks.

## Undated versus explicitly unbounded assertions

`valid=None` means no explicit temporal assertion was supplied. On keys whose
live assertions are all undated, legacy journal-order resolution is retained.
In particular, a later direct outcome can replace another writer's outcome.

`valid=(None, None)` explicitly asserts validity from the beginning to the end
of world time. Two independent writers asserting different values at that same
start can dispute. A disputed value is unavailable as a fact; a plural reader
is not a mechanism for accepting conflicting claims as true.

When a live explicit-valid assertion shares a key with an undated one, the
undated assertion participates as an unbounded beginning-of-time claim in the
existing temporal resolver. Thus an undated base and a bounded `[10, 20)` value
read as base before 10, the bounded value from 10 through 19, and base again
at 20, in either insertion order. An undated claim and an explicit unbounded
claim by different writers with different values dispute at all times.

This is a per-assertion contract, not a persisted global mode switch. If every
explicit-valid assertion is withdrawn, the remaining all-undated live key uses
its legacy behavior again. For a consistently temporal application, supply
explicit validity on every assertion, using `(None, None)` when unbounded.
Nothing silently migrates an old undated journal into a new interpretation.
