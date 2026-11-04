# Optional bounded withdrawal protection

`Knowledge(revival_window=N)` and `KnowledgeStore(revival_window=N)` enable a
finite nonnegative window in logical `tick` units. The owner chooses their meaning
(e.g. one tick/day); these are not dates inferred from `valid` or wall time.
`None` preserves the existing overlap-based, unbounded withdrawal guard.

With the option enabled, a new direct dated assertion of withdrawn content on the
same scoped key is checked against withdrawals on **any validity period**. An
assertion written at elapsed clock <= N remains subject to the existing copy/retest
rules. A genuinely new direct assertion after N may vouch anew, without bypassing
write/source gates, flags, retirement or consulted/premise dependencies. Existing
independent clean support and explicit retest/reassert semantics remain applicable.
Time passing alone never clears old held claims: decisions use the clock recorded
on the new assertion and on the withdrawal, not the current clock at query time.

Replay, known-journal prefixes, verify and branch preserve the declared policy and
reconstruct entry-position-to-clock data. Reopen a persistent journal with the same
explicit revival_window; the policy is supplied by the owner, not inferred from its
contents. The public property is read-only. No authentication, automatic truth,
legitimate-return identification, wall-time expiry, or learned window is promised.
