# Serving certified layers during an additive trial

`AdapterLoop(..., trial_serving="certified")` is an explicit alternative to the
default `trial_serving="escalate"`. It requires additive stream mode. The default
still escalates every input in a candidate's trial zone and reads all its labels.

The opt-in lets an existing layer answer inside that zone when its live risk
promise is certified, it is outside probation, and its score meets its current
live threshold λ. This includes the deepest layer when λ has loosened below the
frozen routing threshold Λ. The usual top-down answering order is retained. The
candidate never answers before promotion; fixed Λ boundaries and its trial zone
do not change when a live threshold loosens.

Candidate evidence changes at the same time as serving. Its risk and usefulness
promises and its actual promotion samples read **only** decisions in the entire
fixed zone whose input-independent base coin wins: `coin(n) < rho_base`.
The rate is computed from pre-decision history, before any current label or
candidate outcome. This is the existing uniform, predictable-rate sampling path
already used by certified layers, not an input-dependent estimator or IPW scheme.
The base coin is shared with those layer audits; a separate independent coin is
not introduced. Its independence requirement concerns the current input/label.

Every selected candidate sample requires a real oracle label. Missing or refused
labels remain losses; the model's own answer is never used as a label. If another
reason escalates an unsampled input and provides a label, the candidate does not
learn from it. Outside-zone base audits do not feed it either. Unlabelled scores
may still be observed to initialize its grid, as before. Existing layers retain
their existing base/owned-regime learning rules.

The stream record retains `base` and `rho_base` and, with this option, also writes
`trial_sampled` and `rho_trial` (None outside the trial zone). These record the
actual candidate selection and its known propensity. Replay recomputes the same
coin, route, threshold and selection; checkpoints persist the option. Historical
checkpoints without it retain default escalation behavior. Attached Dispatchers
use the same pre-decision route and audit gate.
The path's final threshold remains the capped owner/sampling bound `max(λ, Λ)`;
`answer_thr` records the actual serving threshold, which may be the lower live λ.

Minimum support, novel/probe/usefulness floors, risk levels, shared delta charging,
promotion gates and decision-age deadlines are unchanged. Counts are actual
samples, never estimates inflated by inverse propensity. A candidate may take
longer to certify or hit its unchanged trial deadline. This option makes no
measured quality, coverage or cost improvement claim and does not sandbox an
adaptive oracle or external model changes.

Focused regressions use a genuinely certified live base plus a conservative
synthetic historical frozen boundary to isolate the λ < score < Λ band; they do
not repeat the historical multi-seed report. Separate unmodified-route tests
cover exact checkpoint continuation and replay, and attached Dispatcher tests
exercise the same sampled/unsampled gate with normal native dispatch records.
