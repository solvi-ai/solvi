# `solvi.experimental.relations`

Rules over relations (experimental): programs over the relations of a state — a path tree, a family forest — found by
enumeration and judged by a held-out test.

## Composites with a library

`solvi.experimental.relations.library.find_rules(examples, targets, base=..., size=..., budget=..., library=(),
test=HeldOut(other_worlds, targets))` enumerates small programs over relations, admits only what passes the held-out
test, and builds the next rules on the admitted ones (`Library` entries as new terminals); `run(program, world,
library=)` runs a found rule on any world. The enumerator inside is a minimal one until the full relation enumerator
replaces it; the program format and the `Library` fields stay.

::: solvi.experimental.relations

::: solvi.experimental.relations.library
