# `solvi.experimental.audit`

`AuditWatch`: a check that is not run on every decision, run on a random share of decisions as well
(`Dispatcher(audit=)`) and followed with a Bernoulli CUSUM whose threshold is simulated with the audit share and the
reference's sampling error. A flag says the check disagrees with the system more often, not which of them broke (see
[experimental](../experimental.md#audit)).

::: solvi.experimental.audit
