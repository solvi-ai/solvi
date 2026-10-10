# `solvi.experimental.relations`

Relations over a state and rules found over them (experimental): packed-bit `Relations`, the relation algebra, a
bottom-up enumerator with observational equivalence (`enumerate_programs`, `synthesize`, `top_programs`), a `Library` of
accepted programs, compilation into plain set code or index arrays with a differential check (`compile_program`,
`check_program`), encoders (`encode`, `encode_paths`) and `RelAlgProposer` for the laws' held-out test.

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

::: solvi.experimental.relations

::: solvi.experimental.relations.library
