# Readable rule conditions

`System.learn_rule(..., max_literals=3, ngrams=2, negations=True)` opts into richer decision-list conditions. `RuleList` accepts the same keywords. Defaults preserve the existing one-literal learner.

Conditions contain at most `max_literals` positive or negative literals (1–3). String literals optionally include ordered contiguous word ngrams up to `ngrams` (1–3). Negative literals mean that a known literal is absent; rules expose positive `all` and absent `unless` tuples alongside the readable `if` string. Ordered evaluation gives specific exceptions a way to precede generic rules.

Learning uses deterministic greedy covering and a beam of at most32 conditions per depth. Literal supports are indexed once per fit, then intersected with uncovered rows. This is a bounded search heuristic, not an exhaustive search or an implementation of every CN2/RIPPER rule. It may miss a useful conjunction. Existing support/minimum precision and maximum-rule truncation warning still apply. Training precision is smoothed support, not held-out calibrated confidence.

The learned list remains a deterministic ordinary catalog rule with a default answer when no condition matches. These condition features do not by themselves add a calibrated `act` signal, rule/model agreement mode or traffic guarantee. A held-out calibration/guarantee path must be supplied separately before claiming a certified learned-rule service. No historical corpus coverage or speed result follows from the richer grammar.

## Held-out quality and native guarantees

`system.learn_rule(question, training, features=..., calibrated=True, calibration=held_out)` installs an explicit `Decision` route. The training structure never reads the held-out labels. Each first-firing rule (and the default bucket) records correct/total on the caller's independent held-out data and uses `(correct+1)/(total+2)` as its quality signal. This smoothed precision is an estimate, not a certificate or a validated probability-calibration theorem. A bucket with no held-out observations abstains with confidence/act0; no training precision is substituted as confidence1.

`calibration=None` is allowed on this route, but every bucket abstains until `rule_list.calibrate(rows, answers)` supplies held-out support. Existing calls without `calibrated=True` retain their ordinary-rule behavior. Refit clears every quality count; the calibrated route then abstains again. Fingerprints include rule conditions and quality, so old traces detect a changed learned model.

Native `Decision.extra['act']` carries the quality estimate. On a third independent labeled set, use the existing `system.guarantee(question, guarantee_examples, signal='act', max_error=..., ...)` calibration path. The guarantee's method and assumptions remain unchanged. The example-free `promise=` structural act-head shortcut is not implemented for this ordinary-rule wrapper. Ordinary question minimum confidence can also abstain below the quality signal.

The caller must collect training, quality estimation and guarantee calibration separately and declare their independence honestly. Equal feature values can occur in fresh independent observations, while different feature hashes do not prove independence; this API does not infer independence from content. Never reuse training or quality labels as guarantee calibration evidence. After refitting or changing quality data, recalibrate the native guarantee on new independent observations before relying on it. No generic guarantee/corrections channel was changed for this feature.
