# Explicit priority of failed hard checks

`Catalog.check(hard=True, precedence=10, then={"q": "review"})` gives a hard
check an integer priority. Larger numbers win among the **proven false checks
that govern the same question**. A passing check never overrides a failed one.
Negative integers are allowed. `None` and `0` are equivalent defaults. Booleans,
floats (including NaN and infinity), strings, and priorities on ordinary checks
are refused before registration.

For several failed checks, the runtime first selects the group with the largest
priority. Ties retain `Catalog(combine="first")` declaration order or the existing
`strictest` answer-option order, including the existing abstention and callable
fallback rules. Priority therefore expresses the importance of a cause without
requiring its forced answer to come last in the question's options.

This does not introduce dependencies between checks or choose which checks may
be omitted. With a nonzero priority among governing checks, an unavailable
mandatory check makes the question abstain: a more permissive failure response
cannot be selected from an incomplete set of checks. The original error or
missing input is retained in the reason. Without a nonzero priority the existing
legacy handling of hard failures and unavailable evidence is retained.

The default hard-check phase already evaluates all governing checks. A learned
schedule also waits for all governing checks when nonzero priorities apply,
including with `combine="first"`; evaluation order may change, the selected
answer does not. `early_exit=False` gives the same selection. This is not a
performance claim. The winning check's `then` may be static, a function of facts,
an explicit abstention, or absent (a governing check then abstains).

Nonzero priorities are part of the part/catalog fingerprints, decision cache
key, and trace replay identity. Default/zero priorities preserve the existing
fingerprints. Renaming boundary facts or question names retains priorities.
A stored trace made with different priorities is not replay-compatible.
