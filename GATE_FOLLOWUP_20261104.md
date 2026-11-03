# Frozen gate follow-up — 2026-11-04

New lane growth_library_gate_followup_20261104, base 41d4232528934014ca7fa340de802396611bba69. Frozen source remains untouched. S306 documentation-only commit 057c6ed was fetched by its branch/full hash and cherry-picked as a75d179.

Reproduced the requested gate subset unchanged: 4 failed, 3 passed, 2 errors in 3.34s (gate_followup_repro_20261104.log). Default strict=True now rejects the deliberately failing calibration in example 24 and golden auto_built; the latter prevents fingerprint collection and causes the module-relocation failures. The guide's System signature also omitted allow_unreachable_hard_checks.

Migrated only these diagnostic callers to explicit strict=False. Seed, calibration examples/data and risk bars remain identical. The runtime's existing private human latch and infinite thresholds remain active: the example test now verifies all 500 decisions are human when the reported calibration fails. The default strict constructor/build behavior is unchanged. The guide now prints the actual additional System keyword.

After caller migration, the only changed golden entry was golden/auto_built.questions: a3991f3a2f09250d → c1e75316400a7f7e. The failed-calibration human fallback intentionally changes its installed question guarantee threshold to infinity, hence the fingerprint. Catalog, parts, models and every other pinned entry were unchanged. Only this single JSON value was updated; the golden fixture was not regenerated.

Final requested subset: 9 passed, 7 warnings in 4.44s, including module relocation invariants. Expected warnings preserve the explicit diagnostic calibration failure; none were suppressed. Ruff PASS, git diff-check PASS. Logs live in root research_growth/growth_2026_11: gate_followup_repro_20261104.log, gate_followup_firstfix_20261104.log, gate_followup_final_20261104.log, gate_followup_ruff_20261104.log. Python runs used the memory wrapper with OMP/BLAS=4. No new full suite, benchmark, stochastic quality eval or push. Other failures from the ongoing frozen full suite are outside this patch's proof.
