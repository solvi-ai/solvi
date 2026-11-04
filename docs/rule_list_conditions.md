# Readable rule conditions

`System.learn_rule(..., max_literals=3, ngrams=2, negations=True)` opts into richer decision-list conditions. `RuleList` accepts the same keywords. Defaults preserve the existing one-literal learner.

Conditions contain at most `max_literals` positive or negative literals (1–3). String literals optionally include ordered contiguous word ngrams up to `ngrams` (1–3). Negative literals mean that a known literal is absent; rules expose positive `all` and absent `unless` tuples alongside the readable `if` string. Ordered evaluation gives specific exceptions a way to precede generic rules.

Learning uses deterministic greedy covering and a beam of at most32 conditions per depth. Literal supports are indexed once per fit, then intersected with uncovered rows. This is a bounded search heuristic, not an exhaustive search or an implementation of every CN2/RIPPER rule. It may miss a useful conjunction. Existing support/minimum precision and maximum-rule truncation warning still apply. Training precision is smoothed support, not held-out calibrated confidence.

The learned list remains a deterministic ordinary catalog rule with a default answer when no condition matches. These condition features do not by themselves add a calibrated `act` signal, rule/model agreement mode or traffic guarantee. A held-out calibration/guarantee path must be supplied separately before claiming a certified learned-rule service. No historical corpus coverage or speed result follows from the richer grammar.
