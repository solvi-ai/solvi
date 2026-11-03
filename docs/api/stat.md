# `solvi.stat`

Sequential evidence and multiple testing, without model or scientific-computing dependencies.
Existing experimental imports of the shared processes remain available and refer to the same classes.

```python
from solvi.stat import MeanBound, ELOND, e_bh

test = MeanBound(0.5)  # H0: conditional mean of each bounded outcome is at most 0.5
for outcome in [0.8, 0.7, 0.9]:
    test.update(outcome)
saved = test.dump()
resumed = MeanBound(0.5).restore(saved)

ledger = ELOND(q=0.1, schedule="uniform", horizon=50)
allocation = ledger.reserve()  # before inspecting this hypothesis's outcomes
rejected = ledger.finish(allocation.index, 2.0)  # a valid e-value, not an arbitrary score
family_rejections = e_bh([0.0, 25.0, 30.0], q=0.1)
```

`MeanBound.update` returns **log evidence**; `ELOND.finish` and `e_bh` take e-values on the ordinary
nonnegative scale. A marginal average-risk bound is insufficient for an anytime process: the stated null
must hold conditionally on the past. Selecting a promising score and calling it an e-value does not establish
that null validity. Data-only checkpoints preserve arithmetic; they do not authenticate the data's origin.

`Mixture` and `MirrorBound` certify low conditional mean loss, in the opposite direction to `MeanBound`.
`Certificate` configures `EProcess`, including restart budgets and a separate change detector. A detector
alarm has the bound stated in `Certificate`; it is not itself a level-delta certification test.

`DeltaBudget(schedule="horizon", horizon=T)` allocates `delta/T` at each of exactly `T` steps, then refuses
more. The default Basel schedule remains `6*delta/(pi*pi*v*v)` indefinitely. Reserve a budget before
observing the corresponding test; there is no implicit refund or reset.

`ELOND` supports `schedule="uniform", horizon=T` and default `schedule="basel"`. Its minimal sequential
interface requires alternating `reserve` and `finish`: only one reservation may be pending. Resolve an
abandoned test with e-value zero to consume its index conservatively. Do not create a new ledger to reuse
the same epoch's budget. Its FDR statement applies to fixed test prefixes under valid null e-values;
arbitrary stopping of an FDR report needs separate justification. `e_bh` is a fixed-family procedure.
Both allow positive infinity, reject negative values and NaN, and leave online state intact on invalid input.

References: [Wang and Ramdas, e-BH](https://sas.uwaterloo.ca/~wang/papers/2021Wang-Ramdas-JRSSB.pdf),
[Xu and Ramdas, e-LOND](https://proceedings.mlr.press/v238/xu24a/xu24a.pdf).

`agreement_diagnostics(rows, checkers=[...])` describes fixed, caller-normalized answers. Each row supplies `gold`,
`primary` and a `checkers` mapping. Answer tokens are exact strings; `None` and missing checker entries mean missing
data. The report includes agreement frequencies conditional on primary correctness, conditional pair tables/phi,
and the fraction of matching answers when both predictors were wrong. Denominators and missing counts are explicit;
undefined rates and correlations are `None`. This descriptive report does not fit weights, establish independence
or issue a certificate.

::: solvi.stat
