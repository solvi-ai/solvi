# `solvi.experimental.relations`

Relations over a state and rules found over them (experimental): packed-bit `Relations`, the relation algebra, a
bottom-up enumerator with observational equivalence (`enumerate_programs`, `synthesize`, `top_programs`), a `Library` of
accepted programs, compilation into plain set code or index arrays with a differential check (`compile_program`,
`check_program`), encoders (`encode`, `encode_paths`), `RelAlgProposer` for the laws' held-out test, and `CompositionTable` (a composition
table of relations learned by counting, derivation over every bracketing, refusal of what it cannot derive).

## Composites with a library

`solvi.experimental.relations.library.find_rules(examples, targets, base=..., size=..., budget=..., library=(),
test=HeldOut(other_worlds, targets))` enumerates small programs over relations, admits only what passes the held-out
test, and builds the next rules on the admitted ones (`Library` entries as new terminals); `run(program, world,
library=)` runs a found rule on any world. The search is this package's enumerator and the programs are its tuples;
`library=` takes a `Library` (the collection of accepted programs) or a sequence of `Entry` (one accepted program:
program, name, examples, test), and admitted rules are `Entry` objects. `include_universal=False` leaves the universal
relation U out of the enumerator's terminals (opt-in; the default keeps it).

Operators: unary `T` (converse), `NOT` (bounded complement), `PLUS` (transitive closure), `CNT2` / `CNT3` (the nodes
with at least 2 / 3 successors, as a diagonal) and `CNT1` (at least one successor) — `CNT1` is opt-in: known to the
algebra, the enumerator and both compile backends, but not in the default `OPS` (pass it in `ops=`; `find_rules` uses
it through `counts=(1,)`); binary `OR`, `AND`, `MINUS`, `XOR`, `COMP`, `C2`, `C3`.

## Composition tables over kinship chains

`CompositionTable(majority=0.9, support=2)` learns which relation a chain of two relations composes to
(father ∘ sister → aunt) by exact counting, derives longer chains by composing over every bracketing, and refuses what
the table cannot derive.

    from solvi.experimental.relations import CompositionTable, base_rates, sized_admit

    examples = [(("father", "sister"), "aunt"), (("father", "sister"), "aunt"), (("aunt", "son"), "cousin"), ...]
    t = CompositionTable(majority=0.9, support=2).fit(examples)   # chains of length 2 vote; length 3 fill residuals
    d = t.derive(("father", "sister", "son"))
    d.relation, d.reason          # "cousin", "derived" — or None, "underivable" / "disagree" / "identity"
    d.candidates, d.cells         # bracketings per relation; the cells (left, right, relation, hits, support) used
    t.derive_paths([chain_a, chain_b])                            # one query reached by several chains
    t.to_knowledge(store, {"domain": "kinship"})                  # "rule" items with their support

- **Fit.** A chain (r1, r2) labelled r is a vote for the cell r1 ∘ r2 → r; a cell is admitted when its majority
  relation has a share ≥ `majority` and at least `support` votes. **Residual fill:** a chain (a, b, c) labelled r with
  a ∘ b = x in the table votes for x ∘ c → r, and with b ∘ c = y for a ∘ y → r; the residual votes are recomputed from
  the current table each round (a chain counts once per round) until the table stops changing (`rounds=` caps it).
  `fit` is deterministic and independent of the order of the examples; other chain lengths are counted in `ignored`.
- **Derive.** CYK over the chain: each span keeps the relations it composes to and how many bracketings give each; a
  bracketing that meets an empty cell gives nothing. `agree="all"` (default): answered only when every bracketing that
  gives a relation gives the same one; `agree="majority"`: the relation of strictly the most bracketings. Refusals:
  `"underivable"` (no bracketing composes), `"disagree"` (two relations), `"identity"` (only a learned identity).
  `explain(chain, relation=None)` returns the cells used.
- **Identity (opt-in).** `identity="self"`: a chain (a, b, c) labelled c votes a ∘ b → self where the pairs admit no
  cell; self composes as neutral and is never an answer. In domains where a cell depends on an attribute the chain
  does not carry (whose son's mother is "self" depends on the subject's gender), a learned identity adds wrong answers.
- **Admission hook.** `admit=(cell, relation, hits, support, tested) → bool` is an extra gate after the thresholds;
  `sized_admit(base_rates(examples), delta=0.1)` is the sized acceptance of `solvi.experimental.laws` with a base-rate
  null: P[Binomial(support, base_rate(r)) ≥ hits] ≤ δ / N, N the cells tested.

::: solvi.experimental.relations.CompositionTable

::: solvi.experimental.relations.Derivation

::: solvi.experimental.relations.Cell

::: solvi.experimental.relations

::: solvi.experimental.relations.library
