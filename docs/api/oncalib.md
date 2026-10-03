# `solvi.experimental.oncalib`

Calibration on the fly from the outcomes an agent sees (experimental: importing it warns `ExperimentalWarning`). A
guarantee recalibrated continuously does not keep its promise — read the risk note below before using it. The stable
way is explicit: `System.outcome(...)` stores the labels, `System.guarantee(..., corrections=True)` recalibrates on them
when you decide to.

::: solvi.experimental.oncalib
