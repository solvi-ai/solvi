# Selecting identity probes

Clear probes reduce provider noise but can agree across harmful model changes. Repeating the same probes against a deterministic provider adds no input coverage. Stable boundary probes can reveal a different set of changes, but instability on those probes also causes false alarms. Neither selection nor the default tolerance estimates detection power.

```python
from solvi.core.guarantees.canary import IdentityCanary

report = IdentityCanary.pick_probes(
    model, candidate_questions, repeats=3,
    top_probability=(0.5, 0.9), max_repeat_tv=0.02, max_probes=20,
)
if report["probes"]:
    canary = IdentityCanary(report["probes"], every=500).attach(model)
    canary.probe()  # establish the serving reference explicitly
```

Candidates have the ordinary canary probe format: `(text, task, options)` or a dict with those fields and optional descriptions. Supply at most 4096 candidates and 2–32 repeat rounds against one frozen reference provider. No evaluation labels or changed provider are used. The picker asks the raw provider, bypassing local calibration/adaptation through the same provider path as ordinary probes. It preserves any existing attached canary, its reference and events.

A candidate is eligible only if every round answers, answers and reported provider names agree, every top probability lies in the half-open configured range, and each repeat's probability vector is within `max_repeat_tv` total variation of the first round. Eligible candidates are ordered by the first round's top-two probability gap, with input order breaking ties; `max_probes` truncates that list. Returned probe tuples can be passed directly to `IdentityCanary`.

The report includes each candidate's rejection reasons, answered rounds, top-probability range, reference gap, maximum repeat distance and selection status. `calls` counts attempted probe questions, not lower-level retries or provider billing. `answered_fraction` measures observed reference availability; `selection_fraction` measures candidate retention. An empty selection reports `has_stable_boundary_probes=False`; do not silently replace it with clear probes and call that informative. `detection_power=None` is intentional: selection reports do not measure sensitivity to an unknown change. Stability in these repeated calls is not a future noise guarantee.

Every regular canary event already reports `very_clear_reference_count` (reference top probability at least 0.99), probability gaps and shifts. It also reports reference answered count, comparable count/fraction and the range of answered reference top probabilities. A very clear probe can still change; the count is a warning about the chosen reference set, not a proof that it cannot move. Unanswered probes reduce comparison coverage. No value here describes unprobed input coverage or guarantees detection of arbitrary hidden swaps.

The picker spends `repeats * len(candidates)` attempted probe questions. Recheck stability when the serving setup changes. Freeze the selected list and tolerance before evaluating a particular change; adapting the selection to a changed model would evaluate a different question.
