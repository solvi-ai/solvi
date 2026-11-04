# Same-writer pending version continuity (1.5 breaking change)

A standing newer value published by a writer can require rechecking after its
independent supporting source is withdrawn. The value remains a hint; the store
does not trust that pending publication. On the newer publication's validity
span, however, the same writer's older different value no longer certifies an
answer. This creates a temporal gap, rather than silently serving the obsolete
value. A pending assertion from a different writer cannot create this gap in
another writer's clean support.

For example, product_team first says warranty=36 from day0, then says24 from day120
while an independently checked vendor supports24. If the vendor is withdrawn and
the team copy becomes pending, a read at126 now returns no fact from either claim.
It still returns36 before120. The caller can obtain a fresh independent verdict
and call native `retest(id, ok=True, by=...)` to make24 eligible. No label, identity,
permission, trust or re-test is fabricated automatically.

The gap only covers the standing pending version's bounded interval. If that
later assertion is actually withdrawn, the old version reopens as before. Old
claims from an independent writer remain eligible under existing resolution.
Flags, retirement, premise/consulted taints and source/write gates are unchanged.
`periods`, `usable(valid_at=...)`, time-specific snapshots, derived periods, rebuild
and branches use the same effective intervals. Reads without a world time retain
their documented status-only behavior; applications needing temporal answers must
supply `valid_at` or the store's explicit world clock.

This intentionally differs from 1.4's older-value fallback. It fixes an unsafe
host answer, not the truth of the later copy: abstention is still different from
the known gold value. The frozen corpus performance/quality result remains
negative; this focused migration does not establish a new corpus pass.

Continuity clipping changes occupancy only: a fragment reopening at the end of a
gap retains its ORIGINAL version-start priority. It cannot outrank an independent
clean version published between the original start and the gap end. Undated clean
support is preserved when no matching same-writer restriction applies. Derived,
inferred and consulted older contributions are not direct declarations and are
not clipped by this continuity rule; their existing dependency gates decide them.
