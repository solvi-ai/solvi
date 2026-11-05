# Scoring a batch of candidate laws

`lawset.score_many(candidates, buffer, exclude=None)` returns the same ordered
triples as sequential `score` calls, including duplicates. It does not admit laws
or count statistical tests. The default preserves lazy iterator interleaving.

When callbacks and inputs are stable throughout the call, use
`lawset.score_many(candidates, buffer, stable=True)` to build the active-law view
once per action. This is an explicit owner promise: callbacks must not mutate
closure state, laws, candidates or evidence during the call. Views are local to
one call; mutable callbacks need the default sequential path. Existing optional
`memo=True` semantics are unchanged. No speed or quality gain is guaranteed.
