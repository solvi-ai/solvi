# `solvi.experimental.relations`

Relations over a state and rules found over them (experimental): packed-bit `Relations`, the relation algebra, a
bottom-up enumerator with observational equivalence (`enumerate_programs`, `synthesize`, `top_programs`), a `Library` of
accepted programs, compilation into plain set code or index arrays with a differential check (`compile_program`,
`check_program`), encoders (`encode`, `encode_paths`) and `RelAlgProposer` for the laws' held-out test.

## Composites with a library

`solvi.experimental.relations.library.find_rules(examples, targets, base=..., size=..., budget=..., library=(),
test=HeldOut(other_worlds, targets))` enumerates small programs over relations, admits only what passes the held-out
test, and builds the next rules on the admitted ones (`Library` entries as new terminals); `run(program, world,
library=)` runs a found rule on any world. The enumerator inside is a minimal one until the full relation enumerator
replaces it; the program format and the `Library` fields stay.

::: solvi.experimental.relations

::: solvi.experimental.relations.library
