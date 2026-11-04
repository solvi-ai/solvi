# A fixed reference null before a regime change

`LawSet(null=<number>)` already supports a fixed null. `null="base"` instead
recomputes the effect's rate on the allowed buffer rows of the candidate's action,
excluding the proposing group when requested. New-regime rows can raise that
rate. This changes the comparison baseline without changing the candidate's
matched support or observed hits.

An application can compute and record a descriptive reference **before** adding
new-regime rows, then pass that number explicitly:

```python
reference = descriptive.base_rate(prechange_buffer, action, effect)
frozen = LawSet(grammar, base=base_predictor, null=reference,
                min_precision=0.85, retire_below=0.7, delta=0.05, space=space)
```

Here `descriptive` is an existing `LawSet` with the intended action/effect
interpretation, and `prechange_buffer` is a separate reference buffer retained
for provenance. The recipe uses existing APIs and does not create a new null
mode, complement estimator, window selector, or automatic change-point detector.
Record the reference value, source rows/groups, action, effect, applicability
conditions, support mode, `delta`, `space`, and held-out split before evaluation.
Do not choose the baseline window or null after seeing which candidate passes.
The same scalar applies to every candidate tested by this particular `LawSet`;
use separately declared sets if different action/effect hypotheses need different
reference values and account for their testing budgets explicitly.

`base_rate` is descriptive: it counts rows of the action, not just rows matching
the candidate's condition. It uses the configured effect predictor, excludes the
specified group, floors at 0.05, and caps at `min_precision` (0.95 when precision
is 1). It returns the floor for an unseen action. Therefore the output is neither
an exact raw empirical frequency in every case nor an automatic valid upper
confidence bound on the null success probability.

Freezing an **estimated** frequency only freezes the numerical comparator. It
does not establish nominal alpha control, remove estimation uncertainty, or
justify a global marginal rate for a selected condition or changed population.
A binomial-tail interpretation requires a genuinely known null probability or
a justified upper bound for success on the relevant independent held-out rows,
together with the existing assumptions on candidate selection and multiplicity.
If a fixed null is independently justified, pass it directly; calling
`base_rate` is not required. This recipe supplies no new calibration guarantee.

`why_not(candidate, buffer)` reports the selected null, support, hits, and need
without changing the tested-candidate counter. With a numeric null, added regime
rows can change support/hits/net and whether admission passes; they do not change
the null itself. `need_at` and `passes` expose the existing fixed-null arithmetic.
Observed `min_precision`, support, net, and other admission gates remain in force.
No adaptive complement baseline or empirical quality improvement is claimed.
