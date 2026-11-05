# Availability of audit labels

`AuditWatch.report()` returns lifetime `seen`, `audit_requests`, `audited`
(observed labels), `missing`, `audit_errors`, `failed` and `observed_coverage`.
Coverage is observed labels divided by selected requests, or `None` before any
request. An unselected decision is not missing. A check returning `None` or NaN
adds one missing label; `observe()` also returns `missing=True` for that decision.
As before, `passed=None` requests the configured check; without a check this
raises rather than declaring a missing label. Exceptions propagate and count as
errors, not missing labels. False is an observed failing label.

`reset()` clears the detector and latched flag, preserving lifetime counters,
alarms and reference; replacing the reference also preserves lifetime counters.
These diagnostics do not certify random missingness, bound risk, impute outcomes,
or identify the cause of missing labels. Detector updates and thresholds are unchanged.

Labels are explicitly boolean (`bool`/NumPy boolean) or numeric scalars equal to
0 or 1. `None` and Python/NumPy floating NaN represent missing labels (reference
samples skip them). Arrays of every shape, strings, arbitrary objects, infinities
and other numbers raise `ValueError`; arbitrary truthiness is not evaluated.
A reference may be a vector of scalar labels, but array-valued elements are invalid.
Invalid reference labels leave the fitted reference and detector unchanged.
