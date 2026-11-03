# Provider identity leases (opt-in candidate)

`DecideModel(scorer, identity_safe=True, identity_lease=lease_factory)` enables
technical abstention when the deployment identity or fitted calibration is not
confirmed. Existing models without `identity_safe=True` retain their behavior.
No built-in remote adapter currently provides a deployment lease.

The factory must return a context manager yielding a nonnegative integer generation.
The provider must pin one generation for the **whole deployment and router**, from
entry through cache reads, inflight scoring, final decision processing and exit.
Generations must increase on every deployment change, including partial routing;
they must not be reused. This is an explicit provider trust contract. Model names,
served-by names, fixed canary answers and configuration fingerprints cannot supply it.
Missing leases, invalid generations and entry/exit exceptions cause abstention.

Generation changes invalidate cache and fitted adaptations. A new successful
`part.calibrate_for`, `part.act_guard` or `part.conformal` under the lease recovers
the fitted Part binding. Failure restores previous settings for inspection but
leaves them stale. Each successful fit replaces prior threshold, grouped and
conformal configuration; compose those fits only when a later API supports separate
bindings. Canary `accept()` never resets unknown/stale deployment identity.

Checkpoint-level calibrated metadata is unsupported in this mode, including
temperature, thresholds and learned act calibration. Such models remain fail-closed:
fresh Part calibration cannot recover them. Start from an uncalibrated scorer and
fit a Part under the supported lease contract instead.

Calibration files record generation/epoch only as audit provenance. Loading always
requires fresh calibration, including `strict=False`; matching generation numbers
do not establish a deployment namespace across processes.

## Mutation and composition boundaries

With identity-safe mode enabled, model and Part `adapt`, `fit`, `teach`, `reset`,
model `load_adaptations`, `adaptation(create=True)` and Part adapter loading raise
`ValueError` before scoring, consuming example iterators or modifying fitted state.
Use a new uncalibrated model and fresh lease-bound Part calibration to migrate.
`adaptation(create=False)` remains an inspection API; mutating the returned object
or public attributes/dictionaries directly is unsupported and must not be used to
change a live identity-safe model. Private raw scoring/training APIs and external
experimental adapter mutation are outside this contract.

Combinations containing identity-safe leaves are currently unsupported. Their
`decide`/call methods abstain before leaf scoring; low-level `state`/`final`, fitting,
calibration, reset and loading APIs raise `ValueError` before scoring or mutation.
Use an individual Part until a whole-combination lease transaction and binding are
implemented. These boundaries do not imply detection of hidden provider changes
behind a false or incomplete lease token, nor global sensitivity of fixed probes.
