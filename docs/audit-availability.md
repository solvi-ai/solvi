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
