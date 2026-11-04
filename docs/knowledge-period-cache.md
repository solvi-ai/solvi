# Effective interval read cache

KnowledgeStore reuses effective intervals across repeated reads of the same settled state. Only top-level computations with empty recursion context and empty local memo may enter the shared cache. Recursive/seen-dependent results and populated per-query memo results are never promoted into it; cyclic dependencies keep the existing evaluation semantics.

Each cache entry stores a scalar resolved-state epoch and immutable tuple interval pairs. A completed affected-state flush increments the scalar epoch; writes do not clear or traverse the whole cache. Entries are bounded by existing item IDs, including invalid old-epoch entries until that ID is read again. A missing item is never cached. Public periods() returns fresh interval lists, including on cache hits; callers cannot mutate the cached list structure. None retains its prior meaning (no effective validity restriction), not an empty interval set.

This is an internal computation cache, not a new source of knowledge or a change to withdrawal, expiry, flags, derived support, world-time or replay rules. Direct mutation of internal items/state remains unsupported. No performance/SLA gain is claimed without a separately registered matched measurement; the original S266 negative corpus result is unchanged.
