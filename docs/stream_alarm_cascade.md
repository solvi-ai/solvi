# Dependent additive layers after an alarm

`AdapterLoop(..., alarm_cascade="downstream")` explicitly declares that later
additive layers depend on every upstream routing identity and regime. The
existing default, `alarm_cascade="layer"`, retains layer-local recertification.
The downstream option requires additive mode.

A non-ignored alarm restarts the alarmed layer and every later layer, including
their usefulness certificates, clears their old samples, resets downstream risk
detectors, and starts their configured probation. An active candidate is discarded
and its trial boundary removed before old-zone evidence can be reused. Retained
layers keep their frozen routing boundaries. Existing rollback actions detach the
requested suffix; detached models cannot return through additive rollback.

Restarts use the existing shared delta ledger and lazy charging. They do not lower
risk levels, evidence floors, or promotion gates. Checkpoints retain this option,
probation, and certificate state; older checkpoints default to layer-local behavior.
Replay uses the same declared option. Ignored alarms remain ignored by policy.

This option responds to observed alarms. It does not detect an arbitrary hidden
model swap or establish that an unchanged provider has an unchanged distribution.
